/// ReLU activation.
pub fn relu(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    for (dst, src) in output.iter_mut().zip(input.iter().copied()) {
        *dst = src.max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_clamps_negative_values() {
        let input = [-1.0, 0.5, -0.2, 3.0];
        let mut out = [0.0; 4];
        relu(&input, &mut out);
        assert_eq!(out, [0.0, 0.5, 0.0, 3.0]);
    }
}
