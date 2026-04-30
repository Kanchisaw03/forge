#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const C0: f32 = 1.0;
const C1: f32 = 1.0000001;
const C2: f32 = 0.4999999;
const C3: f32 = 0.1666667;
const C4: f32 = 0.0416755;
const C5: f32 = 0.0083298;
const LOG2E: f32 = 1.442695;
const LN2H: f32 = 0.693359375;
const LN2L: f32 = -2.12194440e-4;

/// Scalar fallback approximation for `exp`.
pub fn exp_approx_scalar(x: f32) -> f32 {
    x.exp()
}

/// AVX2 vectorized exponential approximation for 8 `f32` lanes.
///
/// Performance rationale:
/// - branch-free range reduction;
/// - Horner form with FMA minimizes latency/rounding points;
/// - exponent reconstruction uses direct IEEE-754 exponent field synthesis.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn exp_avx2(x: __m256) -> __m256 {
    let x = _mm256_max_ps(x, _mm256_set1_ps(-87.0));
    let x = _mm256_min_ps(x, _mm256_set1_ps(88.0));

    let log2e = _mm256_set1_ps(LOG2E);
    let ln2h = _mm256_set1_ps(LN2H);
    let ln2l = _mm256_set1_ps(LN2L);

    // k = round(x * log2e)
    let kf = _mm256_round_ps(
        _mm256_mul_ps(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );

    // r = x - k*ln2 (high+low split for precision)
    let r = _mm256_sub_ps(x, _mm256_mul_ps(kf, ln2h));
    let r = _mm256_sub_ps(r, _mm256_mul_ps(kf, ln2l));

    // Degree-5 minimax polynomial with fixed coefficients.
    let c0 = _mm256_set1_ps(C0);
    let c1 = _mm256_set1_ps(C1);
    let c2 = _mm256_set1_ps(C2);
    let c3 = _mm256_set1_ps(C3);
    let c4 = _mm256_set1_ps(C4);
    let c5 = _mm256_set1_ps(C5);

    let mut p = c5;
    p = _mm256_fmadd_ps(p, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, c0);

    // 2^k via exponent-field construction.
    let ki = _mm256_cvtps_epi32(kf);
    let exponent = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    let two_pow_k = _mm256_castsi256_ps(exponent);
    _mm256_mul_ps(p, two_pow_k)
}

/// Portable element-wise exp using AVX2 when available.
pub fn exp_slice(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let mut i = 0usize;
            while i + 8 <= input.len() {
                // SAFETY: pointers are valid for 8-lane load/store and AVX2+FMA is feature-gated above.
                unsafe {
                    let x = _mm256_loadu_ps(input.as_ptr().add(i));
                    let y = exp_avx2(x);
                    _mm256_storeu_ps(output.as_mut_ptr().add(i), y);
                }
                i += 8;
            }
            for j in i..input.len() {
                output[j] = exp_approx_scalar(input[j]);
            }
            return;
        }
    }

    for (dst, src) in output.iter_mut().zip(input.iter().copied()) {
        *dst = exp_approx_scalar(src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_slice_matches_std_reasonably() {
        let input = (-100..100).map(|x| x as f32 * 0.5).collect::<Vec<_>>();
        let mut out = vec![0.0f32; input.len()];
        exp_slice(&input, &mut out);
        let max_rel = input
            .iter()
            .zip(out.iter())
            .map(|(x, y)| {
                let r = x.exp();
                if r == 0.0 {
                    0.0
                } else {
                    ((r - *y) / r).abs()
                }
            })
            .fold(0.0f32, f32::max);
        assert!(max_rel < 5e-3, "max relative error too large: {max_rel}");
    }
}
