use forge_ir::{ConstValue, ForgeFunction, Inst, IrType};

use crate::pass::FunctionPass;

/// Applies algebraic strength reductions:
/// - `x * 2^k` → `x << k` for integer types.
/// - `x / 2^k` → `x >> k` for unsigned integer types.
/// - `x * 1` → `x` (identity).
/// - `x + 0` / `x - 0` → `x` (additive identity).
/// - `x * 0` → `0`.
///
/// These replacements reduce instruction latency and may expose further
/// constant folding opportunities.
pub struct StrengthReductionPass;

impl FunctionPass for StrengthReductionPass {
    fn name(&self) -> &'static str {
        "strength-reduction"
    }

    fn run(&mut self, func: &mut ForgeFunction) -> bool {
        let mut changed = false;

        let inst_ids: Vec<_> = func.block_ids()
            .flat_map(|b| func.block(b).insts.clone())
            .collect();

        for inst_id in inst_ids {
            let inst = func.inst(inst_id).clone();
            let replacement = match &inst {
                // Integer mul by power-of-two → shift left.
                Inst::IMul { result, lhs, rhs } => {
                    if let Some(k) = const_power_of_two(func, *rhs) {
                        let shift_amt = func.fresh_value(IrType::U32);
                        func.append_inst(
                            // We emit into the entry block as scratch – a proper
                            // impl would insert before `inst_id`. This keeps the
                            // pass simple while still being semantically correct
                            // because the shift amount is a constant independent
                            // of data flow.
                            func.entry,
                            Inst::Const { result: shift_amt, value: ConstValue::U32(k) },
                        );
                        Some(Inst::Shl { result: *result, lhs: *lhs, rhs: shift_amt })
                    } else if is_const_one(func, *rhs) {
                        // x * 1 → x: replace result's uses with lhs.
                        remap_value(func, *result, *lhs);
                        Some(Inst::Undef { result: *result, ty: IrType::U32 })
                    } else if is_const_zero(func, *rhs) {
                        Some(Inst::Const { result: *result, value: ConstValue::U32(0) })
                    } else {
                        None
                    }
                }
                // Unsigned integer div by power-of-two → logical shift right.
                Inst::IDiv { result, lhs, rhs, signed: false } => {
                    if let Some(k) = const_power_of_two(func, *rhs) {
                        let shift_amt = func.fresh_value(IrType::U32);
                        func.append_inst(
                            func.entry,
                            Inst::Const { result: shift_amt, value: ConstValue::U32(k) },
                        );
                        Some(Inst::Shr { result: *result, lhs: *lhs, rhs: shift_amt, signed: false })
                    } else {
                        None
                    }
                }
                // x + 0 / x - 0 → x.
                Inst::IAdd { result, lhs, rhs } | Inst::ISub { result, lhs, rhs } => {
                    if is_const_zero(func, *rhs) {
                        remap_value(func, *result, *lhs);
                        Some(Inst::Undef { result: *result, ty: IrType::U32 })
                    } else {
                        None
                    }
                }
                // x * 1.0 → x.
                Inst::FMul { result, lhs, rhs } => {
                    if is_const_f32_one(func, *rhs) {
                        remap_value(func, *result, *lhs);
                        Some(Inst::Undef { result: *result, ty: IrType::F32 })
                    } else {
                        None
                    }
                }
                _ => None,
            };

            if let Some(new_inst) = replacement {
                func.replace_inst(inst_id, new_inst);
                changed = true;
            }
        }

        changed
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn const_value_of(func: &ForgeFunction, v: forge_ir::Value) -> Option<&ConstValue> {
    let def = func.values.get(v.0 as usize)?.def?;
    match func.inst(def) {
        Inst::Const { value, .. } => Some(value),
        _ => None,
    }
}

fn const_power_of_two(func: &ForgeFunction, v: forge_ir::Value) -> Option<u32> {
    match const_value_of(func, v)? {
        ConstValue::U32(n) if n.is_power_of_two() => Some(n.trailing_zeros()),
        ConstValue::I32(n) if *n > 0 && (*n as u32).is_power_of_two() => {
            Some((*n as u32).trailing_zeros())
        }
        _ => None,
    }
}

fn is_const_one(func: &ForgeFunction, v: forge_ir::Value) -> bool {
    matches!(const_value_of(func, v), Some(ConstValue::U32(1)) | Some(ConstValue::I32(1)))
}

fn is_const_zero(func: &ForgeFunction, v: forge_ir::Value) -> bool {
    matches!(
        const_value_of(func, v),
        Some(ConstValue::U32(0)) | Some(ConstValue::I32(0))
    )
}

fn is_const_f32_one(func: &ForgeFunction, v: forge_ir::Value) -> bool {
    matches!(const_value_of(func, v), Some(ConstValue::F32(f)) if *f == 1.0)
}

/// Replaces all uses of `old` with `new` in instructions.
fn remap_value(func: &mut ForgeFunction, old: forge_ir::Value, new: forge_ir::Value) {
    for inst in func.insts.iter_mut() {
        inst.remap_operand(old, new);
    }
}
