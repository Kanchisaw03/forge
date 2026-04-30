use crate::exp_approx::exp_slice;

/// Numerically stable softmax over `rows x dim` layout.
///
/// Invariants:
/// - `input.len() == rows * dim`
/// - `output.len() == rows * dim`
pub fn softmax(input: &[f32], output: &mut [f32], rows: usize, dim: usize) {
    assert_eq!(input.len(), rows * dim);
    assert_eq!(output.len(), rows * dim);

    for row in 0..rows {
        let base = row * dim;
        let src = &input[base..base + dim];
        let dst = &mut output[base..base + dim];
        let max_val = src
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut shifted = vec![0.0f32; dim];
        for (i, v) in src.iter().copied().enumerate() {
            shifted[i] = v - max_val;
        }
        exp_slice(&shifted, dst);
        let sum = dst.iter().copied().sum::<f32>();
        let inv_sum = 1.0 / sum;
        for v in dst {
            *v *= inv_sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outputs_sum_to_one_per_row() {
        let input = vec![1.0f32, 2.0, 3.0, 100.0, 100.0, 101.0];
        let mut out = vec![0.0f32; input.len()];
        softmax(&input, &mut out, 2, 3);
        let row0 = out[0..3].iter().sum::<f32>();
        let row1 = out[3..6].iter().sum::<f32>();
        assert!((row0 - 1.0).abs() < 1e-6);
        assert!((row1 - 1.0).abs() < 1e-6);
    }
}
