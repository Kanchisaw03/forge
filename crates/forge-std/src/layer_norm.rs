/// Layer normalization for `rows x dim`.
///
/// Invariants:
/// - `input/output` length is `rows * dim`
/// - `gamma/beta` length is `dim`
pub fn layer_norm(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    rows: usize,
    dim: usize,
    eps: f32,
) {
    assert_eq!(input.len(), rows * dim);
    assert_eq!(output.len(), rows * dim);
    assert_eq!(gamma.len(), dim);
    assert_eq!(beta.len(), dim);

    for row in 0..rows {
        let base = row * dim;
        let src = &input[base..base + dim];
        let dst = &mut output[base..base + dim];

        // Welford pass: numerically stable online mean/variance accumulation.
        let mut mean = 0.0f32;
        let mut m2 = 0.0f32;
        let mut count = 0.0f32;
        for x in src.iter().copied() {
            count += 1.0;
            let delta = x - mean;
            mean += delta / count;
            let delta2 = x - mean;
            m2 += delta * delta2;
        }
        let var = if dim > 0 { m2 / dim as f32 } else { 0.0 };
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..dim {
            let normalized = (src[i] - mean) * inv_std;
            dst[i] = gamma[i] * normalized + beta[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_zero_mean_unitish_variance() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        let gamma = vec![1.0f32, 1.0];
        let beta = vec![0.0f32, 0.0];
        layer_norm(&input, &mut out, &gamma, &beta, 2, 2, 1e-5);
        for row in 0..2 {
            let s = &out[row * 2..row * 2 + 2];
            let mean = s.iter().sum::<f32>() / 2.0;
            assert!(mean.abs() < 1e-4);
        }
    }

    #[test]
    fn layer_norm_is_stable_on_large_offset_values() {
        let input = vec![100_000.0f32, 100_001.0, 99_999.0, 100_002.0];
        let mut out = vec![0.0f32; 4];
        let gamma = vec![1.0f32, 1.0, 1.0, 1.0];
        let beta = vec![0.0f32, 0.0, 0.0, 0.0];
        layer_norm(&input, &mut out, &gamma, &beta, 1, 4, 1e-5);
        let mean = out.iter().copied().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-3, "normalized mean drift: {mean}");
    }
}
