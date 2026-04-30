use forge_ir::{ConstValue, Inst, IrType};

/// AVX2 instruction selection helper.
pub struct Avx2Selector;

impl Avx2Selector {
    /// Returns a Rust expression implementing `inst`, or `None` when unsupported.
    pub fn emit_inst(inst: &Inst, ty: Option<&IrType>) -> Option<String> {
        let expr = match inst {
            Inst::FAdd { lhs, rhs, .. } if is_vec_f32(ty) => {
                format!("_mm256_add_ps(v{}, v{})", lhs.0, rhs.0)
            }
            Inst::FSub { lhs, rhs, .. } if is_vec_f32(ty) => {
                format!("_mm256_sub_ps(v{}, v{})", lhs.0, rhs.0)
            }
            Inst::FMul { lhs, rhs, .. } if is_vec_f32(ty) => {
                format!("_mm256_mul_ps(v{}, v{})", lhs.0, rhs.0)
            }
            Inst::FDiv { lhs, rhs, .. } if is_vec_f32(ty) => {
                format!("_mm256_div_ps(v{}, v{})", lhs.0, rhs.0)
            }
            Inst::FMA { a, b, c, .. } if is_vec_f32(ty) => {
                format!("_mm256_fmadd_ps(v{}, v{}, v{})", a.0, b.0, c.0)
            }
            Inst::IAdd { lhs, rhs, .. } if is_vec_i32(ty) => {
                format!("_mm256_add_epi32(v{}, v{})", lhs.0, rhs.0)
            }
            Inst::ISub { lhs, rhs, .. } if is_vec_i32(ty) => {
                format!("_mm256_sub_epi32(v{}, v{})", lhs.0, rhs.0)
            }
            Inst::IMul { lhs, rhs, .. } if is_vec_i32(ty) => {
                format!("_mm256_mullo_epi32(v{}, v{})", lhs.0, rhs.0)
            }
            Inst::And { lhs, rhs, .. } if is_vec_i32(ty) => {
                format!("_mm256_and_si256(v{}, v{})", lhs.0, rhs.0)
            }
            Inst::Or { lhs, rhs, .. } if is_vec_i32(ty) => {
                format!("_mm256_or_si256(v{}, v{})", lhs.0, rhs.0)
            }
            Inst::Xor { lhs, rhs, .. } if is_vec_i32(ty) => {
                format!("_mm256_xor_si256(v{}, v{})", lhs.0, rhs.0)
            }
            Inst::Splat { scalar, .. } if is_vec_f32(ty) => {
                format!("_mm256_set1_ps(v{} as f32)", scalar.0)
            }
            Inst::Blend { a, b, mask, .. } if is_vec_f32(ty) => {
                format!("_mm256_blendv_ps(v{}, v{}, v{})", a.0, b.0, mask.0)
            }
            Inst::Const { value, .. } => return Some(emit_const(value)),
            _ => return None,
        };
        Some(expr)
    }
}

fn emit_const(value: &ConstValue) -> String {
    match value {
        ConstValue::F32(v) => format!("{v}f32"),
        ConstValue::F64(v) => format!("{v}f64"),
        ConstValue::I32(v) => format!("{v}i32"),
        ConstValue::I64(v) => format!("{v}i64"),
        ConstValue::U32(v) => format!("{v}u32"),
        ConstValue::U64(v) => format!("{v}u64"),
        ConstValue::Bool(v) => format!("{v}"),
    }
}

fn is_vec_f32(ty: Option<&IrType>) -> bool {
    matches!(
        ty,
        Some(IrType::Vector { elem, lanes }) if **elem == IrType::F32 && *lanes == 8
    )
}

fn is_vec_i32(ty: Option<&IrType>) -> bool {
    matches!(
        ty,
        Some(IrType::Vector { elem, lanes }) if **elem == IrType::I32 && *lanes == 8
    )
}
