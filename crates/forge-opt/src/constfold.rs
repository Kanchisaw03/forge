use std::collections::HashMap;

use forge_ir::{ConstValue, ForgeFunction, Inst, IrType};

use crate::pass::FunctionPass;

/// Constant folding pass.
#[derive(Default)]
pub struct ConstantFolding;

impl FunctionPass for ConstantFolding {
    fn name(&self) -> &'static str {
        "ConstantFolding"
    }

    fn run(&mut self, func: &mut ForgeFunction) -> bool {
        let mut consts: HashMap<forge_ir::Value, ConstValue> = HashMap::new();
        for inst in &func.insts {
            if let Inst::Const { result, value } = inst {
                consts.insert(*result, value.clone());
            }
        }

        let mut changed = false;
        for inst_id in 0..func.insts.len() {
            let inst = func.insts[inst_id].clone();
            let replacement = match inst {
                Inst::FAdd { result, lhs, rhs } => fold_fbin(result, lhs, rhs, &consts, |a, b| a + b),
                Inst::FSub { result, lhs, rhs } => fold_fbin(result, lhs, rhs, &consts, |a, b| a - b),
                Inst::FMul { result, lhs, rhs } => fold_fbin(result, lhs, rhs, &consts, |a, b| a * b),
                Inst::FDiv { result, lhs, rhs } => fold_fbin(result, lhs, rhs, &consts, |a, b| a / b),
                Inst::IAdd { result, lhs, rhs } => fold_ubin(result, lhs, rhs, &consts, |a, b| a.wrapping_add(b)),
                Inst::ISub { result, lhs, rhs } => fold_ubin(result, lhs, rhs, &consts, |a, b| a.wrapping_sub(b)),
                Inst::IMul { result, lhs, rhs } => fold_ubin(result, lhs, rhs, &consts, |a, b| a.wrapping_mul(b)),
                Inst::ICmp { result, pred, lhs, rhs } => fold_icmp(result, pred, lhs, rhs, &consts),
                Inst::FCmp { result, pred, lhs, rhs } => fold_fcmp(result, pred, lhs, rhs, &consts),
                _ => None,
            };
            if let Some(new_inst) = replacement {
                if let Inst::Const { result, value } = &new_inst {
                    consts.insert(*result, value.clone());
                }
                func.insts[inst_id] = new_inst;
                changed = true;
            }
        }
        if changed {
            func.rebuild_uses();
        }
        changed
    }
}

fn fold_fbin(
    result: forge_ir::Value,
    lhs: forge_ir::Value,
    rhs: forge_ir::Value,
    consts: &HashMap<forge_ir::Value, ConstValue>,
    op: impl FnOnce(f64, f64) -> f64,
) -> Option<Inst> {
    let a = const_f64(consts, lhs)?;
    let b = const_f64(consts, rhs)?;
    let out = op(a, b);
    let value = if matches!(consts.get(&lhs), Some(ConstValue::F32(_)))
        || matches!(consts.get(&rhs), Some(ConstValue::F32(_)))
    {
        ConstValue::F32(out as f32)
    } else {
        ConstValue::F64(out)
    };
    Some(Inst::Const { result, value })
}

fn fold_ubin(
    result: forge_ir::Value,
    lhs: forge_ir::Value,
    rhs: forge_ir::Value,
    consts: &HashMap<forge_ir::Value, ConstValue>,
    op: impl FnOnce(u64, u64) -> u64,
) -> Option<Inst> {
    let a = const_u64(consts, lhs)?;
    let b = const_u64(consts, rhs)?;
    let out = op(a, b);
    let value = match (
        consts.get(&lhs).map(const_kind),
        consts.get(&rhs).map(const_kind),
    ) {
        (Some(IrType::I32), _) | (_, Some(IrType::I32)) => ConstValue::I32(out as i32),
        (Some(IrType::I64), _) | (_, Some(IrType::I64)) => ConstValue::I64(out as i64),
        (Some(IrType::U64), _) | (_, Some(IrType::U64)) => ConstValue::U64(out),
        _ => ConstValue::U32(out as u32),
    };
    Some(Inst::Const { result, value })
}

fn fold_icmp(
    result: forge_ir::Value,
    pred: forge_ir::IntPred,
    lhs: forge_ir::Value,
    rhs: forge_ir::Value,
    consts: &HashMap<forge_ir::Value, ConstValue>,
) -> Option<Inst> {
    let a = const_u64(consts, lhs)?;
    let b = const_u64(consts, rhs)?;
    let out = match pred {
        forge_ir::IntPred::Eq => a == b,
        forge_ir::IntPred::Ne => a != b,
        forge_ir::IntPred::Ult | forge_ir::IntPred::Slt => a < b,
        forge_ir::IntPred::Ule | forge_ir::IntPred::Sle => a <= b,
        forge_ir::IntPred::Ugt | forge_ir::IntPred::Sgt => a > b,
        forge_ir::IntPred::Uge | forge_ir::IntPred::Sge => a >= b,
    };
    Some(Inst::Const {
        result,
        value: ConstValue::Bool(out),
    })
}

fn fold_fcmp(
    result: forge_ir::Value,
    pred: forge_ir::FloatPred,
    lhs: forge_ir::Value,
    rhs: forge_ir::Value,
    consts: &HashMap<forge_ir::Value, ConstValue>,
) -> Option<Inst> {
    let a = const_f64(consts, lhs)?;
    let b = const_f64(consts, rhs)?;
    let out = match pred {
        forge_ir::FloatPred::Oeq => a == b,
        forge_ir::FloatPred::One => a != b,
        forge_ir::FloatPred::Olt => a < b,
        forge_ir::FloatPred::Ole => a <= b,
        forge_ir::FloatPred::Ogt => a > b,
        forge_ir::FloatPred::Oge => a >= b,
        forge_ir::FloatPred::Uno => a.is_nan() || b.is_nan(),
    };
    Some(Inst::Const {
        result,
        value: ConstValue::Bool(out),
    })
}

fn const_u64(map: &HashMap<forge_ir::Value, ConstValue>, v: forge_ir::Value) -> Option<u64> {
    match map.get(&v)? {
        ConstValue::I32(x) => Some(*x as i64 as u64),
        ConstValue::I64(x) => Some(*x as u64),
        ConstValue::U32(x) => Some(*x as u64),
        ConstValue::U64(x) => Some(*x),
        _ => None,
    }
}

fn const_f64(map: &HashMap<forge_ir::Value, ConstValue>, v: forge_ir::Value) -> Option<f64> {
    match map.get(&v)? {
        ConstValue::F32(x) => Some(*x as f64),
        ConstValue::F64(x) => Some(*x),
        _ => None,
    }
}

fn const_kind(c: &ConstValue) -> IrType {
    match c {
        ConstValue::F32(_) => IrType::F32,
        ConstValue::F64(_) => IrType::F64,
        ConstValue::I32(_) => IrType::I32,
        ConstValue::I64(_) => IrType::I64,
        ConstValue::U32(_) => IrType::U32,
        ConstValue::U64(_) => IrType::U64,
        ConstValue::Bool(_) => IrType::Bool,
    }
}

#[cfg(test)]
mod tests {
    use forge_ir::{ForgeFunction, Terminator};

    use super::*;

    #[test]
    fn folds_simple_float_mul() {
        let mut f = ForgeFunction::new("cf");
        let a = f.fresh_value(IrType::F32);
        let b = f.fresh_value(IrType::F32);
        let out = f.fresh_value(IrType::F32);
        f.append_inst(
            f.entry,
            Inst::Const {
                result: a,
                value: ConstValue::F32(3.0),
            },
        );
        f.append_inst(
            f.entry,
            Inst::Const {
                result: b,
                value: ConstValue::F32(2.0),
            },
        );
        f.append_inst(f.entry, Inst::FMul { result: out, lhs: a, rhs: b });
        f.set_terminator(f.entry, Terminator::Return(Some(out)));
        let mut pass = ConstantFolding;
        let changed = pass.run(&mut f);
        assert!(changed);
        assert!(matches!(
            f.insts[2],
            Inst::Const {
                value: ConstValue::F32(v),
                ..
            } if (v - 6.0).abs() < f32::EPSILON
        ));
    }
}
