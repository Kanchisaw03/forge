use rayon::prelude::*;

const TILE_M: usize = 32;
const TILE_N: usize = 32;
const TILE_K: usize = 32;
const MR: usize = 6;
const NR: usize = 8;

/// SGEMM kernel: `C = alpha * A * B + beta * C` (row-major).
///
/// Invariants:
/// - `a.len() == m * k`
/// - `b.len() == k * n`
/// - `c.len() == m * n`
pub fn sgemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if beta == 0.0 {
            c.fill(0.0);
        } else if beta != 1.0 {
            c.iter_mut().for_each(|v| *v *= beta);
        }
        return;
    }

    let row_block_elems = n * TILE_M;
    c.par_chunks_mut(row_block_elems)
        .enumerate()
        .for_each(|(tile_i, c_rows)| {
            let i0 = tile_i * TILE_M;
            let rows = c_rows.len() / n;
            let i_max = i0 + rows;

            for j0 in (0..n).step_by(TILE_N) {
                let j_max = (j0 + TILE_N).min(n);
                for k0 in (0..k).step_by(TILE_K) {
                    let k_max = (k0 + TILE_K).min(k);
                    let first_k_tile = k0 == 0;

                    let mut ii = i0;
                    while ii + MR <= i_max {
                        let mut jj = j0;
                        while jj + NR <= j_max {
                            microkernel_6x8(
                                a,
                                b,
                                c_rows,
                                i0,
                                n,
                                k,
                                ii,
                                jj,
                                k0,
                                k_max,
                                alpha,
                                if first_k_tile { beta } else { 1.0 },
                            );
                            jj += NR;
                        }

                        if jj < j_max {
                            scalar_microkernel(
                                a, b, c_rows, i0, n, k, ii, jj, ii + MR, j_max, k0, k_max, alpha,
                                if first_k_tile { beta } else { 1.0 },
                            );
                        }

                        ii += MR;
                    }

                    if ii < i_max {
                        scalar_microkernel(
                            a, b, c_rows, i0, n, k, ii, j0, i_max, j_max, k0, k_max, alpha,
                            if first_k_tile { beta } else { 1.0 },
                        );
                    }
                }
            }
        });
}

/// Compatibility wrapper for plain GEMM where `beta=0`.
///
/// Invariants:
/// - same as [`sgemm`]
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    sgemm(a, b, c, m, n, k, 1.0, 0.0);
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn scalar_microkernel(
    a: &[f32],
    b: &[f32],
    c_rows: &mut [f32],
    i0: usize,
    n: usize,
    k: usize,
    ii_start: usize,
    jj_start: usize,
    ii_end: usize,
    jj_end: usize,
    k0: usize,
    k_max: usize,
    alpha: f32,
    beta_effective: f32,
) {
    for ii in ii_start..ii_end {
        let ci = (ii - i0) * n;
        for jj in jj_start..jj_end {
            let mut acc = 0.0f32;
            for kk in k0..k_max {
                acc += a[ii * k + kk] * b[kk * n + jj];
            }
            let dst = &mut c_rows[ci + jj];
            if beta_effective == 0.0 {
                *dst = alpha * acc;
            } else {
                *dst = alpha * acc + beta_effective * *dst;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn microkernel_6x8(
    a: &[f32],
    b: &[f32],
    c_rows: &mut [f32],
    i0: usize,
    n: usize,
    k: usize,
    ii: usize,
    jj: usize,
    k0: usize,
    k_max: usize,
    alpha: f32,
    beta_effective: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: feature detection guarantees AVX2/FMA availability. All pointer arithmetic
            // stays within `a`, `b`, and `c_rows` bounds for a full 6x8 tile.
            unsafe {
                use std::arch::x86_64::*;

                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();
                let mut acc4 = _mm256_setzero_ps();
                let mut acc5 = _mm256_setzero_ps();

                for kk in k0..k_max {
                    let vb = _mm256_loadu_ps(b.as_ptr().add(kk * n + jj));
                    let a0 = _mm256_set1_ps(a[ii * k + kk]);
                    let a1 = _mm256_set1_ps(a[(ii + 1) * k + kk]);
                    let a2 = _mm256_set1_ps(a[(ii + 2) * k + kk]);
                    let a3 = _mm256_set1_ps(a[(ii + 3) * k + kk]);
                    let a4 = _mm256_set1_ps(a[(ii + 4) * k + kk]);
                    let a5 = _mm256_set1_ps(a[(ii + 5) * k + kk]);

                    acc0 = _mm256_fmadd_ps(a0, vb, acc0);
                    acc1 = _mm256_fmadd_ps(a1, vb, acc1);
                    acc2 = _mm256_fmadd_ps(a2, vb, acc2);
                    acc3 = _mm256_fmadd_ps(a3, vb, acc3);
                    acc4 = _mm256_fmadd_ps(a4, vb, acc4);
                    acc5 = _mm256_fmadd_ps(a5, vb, acc5);
                }

                let alpha_v = _mm256_set1_ps(alpha);
                acc0 = _mm256_mul_ps(acc0, alpha_v);
                acc1 = _mm256_mul_ps(acc1, alpha_v);
                acc2 = _mm256_mul_ps(acc2, alpha_v);
                acc3 = _mm256_mul_ps(acc3, alpha_v);
                acc4 = _mm256_mul_ps(acc4, alpha_v);
                acc5 = _mm256_mul_ps(acc5, alpha_v);

                for (row_off, acc) in [acc0, acc1, acc2, acc3, acc4, acc5].into_iter().enumerate() {
                    let c_ptr = c_rows
                        .as_mut_ptr()
                        .add((ii + row_off - i0) * n + jj);
                    if beta_effective == 0.0 {
                        _mm256_storeu_ps(c_ptr, acc);
                    } else {
                        let old = _mm256_loadu_ps(c_ptr);
                        let merged = if beta_effective == 1.0 {
                            _mm256_add_ps(acc, old)
                        } else {
                            let beta_v = _mm256_set1_ps(beta_effective);
                            _mm256_fmadd_ps(beta_v, old, acc)
                        };
                        _mm256_storeu_ps(c_ptr, merged);
                    }
                }
                return;
            }
        }
    }

    scalar_microkernel(
        a,
        b,
        c_rows,
        i0,
        n,
        k,
        ii,
        jj,
        ii + MR,
        jj + NR,
        k0,
        k_max,
        alpha,
        beta_effective,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(
        a: &[f32],
        b: &[f32],
        mut c: Vec<f32>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> Vec<f32> {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = alpha * sum + beta * c[i * n + j];
            }
        }
        c
    }

    #[test]
    fn matmul_matches_reference() {
        let m = 64;
        let n = 64;
        let k = 64;
        let a = (0..m * k).map(|i| (i % 17) as f32 * 0.1).collect::<Vec<_>>();
        let b = (0..k * n).map(|i| (i % 13) as f32 * 0.2).collect::<Vec<_>>();
        let mut c = vec![0.0f32; m * n];
        matmul(&a, &b, &mut c, m, n, k);
        let ref_c = reference(&a, &b, vec![0.0; m * n], m, n, k, 1.0, 0.0);
        let max_err = c
            .iter()
            .zip(ref_c.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "max abs error too large: {max_err}");
    }

    #[test]
    fn sgemm_alpha_beta_matches_reference() {
        let m = 37;
        let n = 29;
        let k = 23;
        let alpha = 0.75f32;
        let beta = -0.25f32;
        let a = (0..m * k).map(|i| ((i % 11) as f32 - 3.0) * 0.2).collect::<Vec<_>>();
        let b = (0..k * n).map(|i| ((i % 7) as f32 - 2.0) * 0.3).collect::<Vec<_>>();
        let c_init = (0..m * n).map(|i| ((i % 5) as f32 - 1.0) * 0.1).collect::<Vec<_>>();
        let mut c = c_init.clone();
        sgemm(&a, &b, &mut c, m, n, k, alpha, beta);
        let ref_c = reference(&a, &b, c_init, m, n, k, alpha, beta);
        let max_err = c
            .iter()
            .zip(ref_c.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 2e-3, "max abs error too large: {max_err}");
    }
}
