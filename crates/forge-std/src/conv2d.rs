/// Naive NCHW conv2d (single batch).
///
/// Invariants:
/// - `input.len() == in_c * in_h * in_w`
/// - `weight.len() == out_c * in_c * k_h * k_w`
/// - `bias.len() == out_c`
/// - `output.len() == out_c * out_h * out_w`
pub fn conv2d(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    in_c: usize,
    in_h: usize,
    in_w: usize,
    out_c: usize,
    k_h: usize,
    k_w: usize,
    stride: usize,
    pad: usize,
) {
    let out_h = (in_h + 2 * pad - k_h) / stride + 1;
    let out_w = (in_w + 2 * pad - k_w) / stride + 1;
    assert_eq!(input.len(), in_c * in_h * in_w);
    assert_eq!(weight.len(), out_c * in_c * k_h * k_w);
    assert_eq!(bias.len(), out_c);
    assert_eq!(output.len(), out_c * out_h * out_w);

    for oc in 0..out_c {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc = bias[oc];
                for ic in 0..in_c {
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let ih = oh * stride + kh;
                            let iw = ow * stride + kw;
                            let ih_pad = ih as isize - pad as isize;
                            let iw_pad = iw as isize - pad as isize;
                            if ih_pad < 0
                                || iw_pad < 0
                                || ih_pad >= in_h as isize
                                || iw_pad >= in_w as isize
                            {
                                continue;
                            }
                            let in_idx = ic * in_h * in_w + ih_pad as usize * in_w + iw_pad as usize;
                            let w_idx = (((oc * in_c + ic) * k_h + kh) * k_w) + kw;
                            acc += input[in_idx] * weight[w_idx];
                        }
                    }
                }
                let out_idx = oc * out_h * out_w + oh * out_w + ow;
                output[out_idx] = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv2d_basic_case() {
        let input = vec![
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let weight = vec![
            1.0f32, 0.0,
            0.0, -1.0,
        ];
        let bias = vec![0.0f32];
        let mut out = vec![0.0f32; 4];
        conv2d(
            &input, &weight, &bias, &mut out,
            1, 3, 3,
            1, 2, 2,
            1, 0,
        );
        assert_eq!(out, vec![-4.0, -4.0, -4.0, -4.0]);
    }
}
