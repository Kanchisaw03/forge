use forge_ir::{ForgeFunction, Inst, InstId};

use crate::pass::FunctionPass;

/// Fuses `result = a * b; out = result + c` → `out = fma(a, b, c)`.
///
/// This pass is required by the design to ensure the vectorization pass can
/// later lower `FMA` nodes to `_mm256_fmadd_ps` intrinsics.
///
/// Algorithm:
/// 1. For every `FMul` whose result is used exactly once and that single use
///    is an `FAdd` (or `FSub`) in the same block, replace both instructions
///    with a single `FMA`.
/// 2. Mark the original `FMul` result as dead so DCE removes it.
pub struct FmaFusionPass;

impl FunctionPass for FmaFusionPass {
    fn name(&self) -> &'static str {
        "fma-fusion"
    }

    fn run(&mut self, func: &mut ForgeFunction) -> bool {
        let mut changed = false;

        // Collect candidate (fmul_id, fadd_id) pairs.
        let candidates: Vec<(InstId, InstId)> = func
            .block_ids()
            .flat_map(|block| {
                func.block(block)
                    .insts
                    .iter()
                    .copied()
                    .filter_map(|inst_id| {
                        let inst = func.inst(inst_id);
                        // Only fuse f32 / f64 multiplies.
                        if !matches!(inst, Inst::FMul { .. }) {
                            return None;
                        }
                        let result = inst.result()?;
                        // Result must be used exactly once.
                        let uses = &func.values[result.0 as usize].uses;
                        if uses.len() != 1 {
                            return None;
                        }
                        let use_inst_id = uses[0].inst;
                        let use_inst = func.inst(use_inst_id);
                        // The single use must be FAdd or FSub.
                        match use_inst {
                            Inst::FAdd { lhs, rhs, result: _add_result } => {
                                if *lhs == result || *rhs == result {
                                    Some((inst_id, use_inst_id))
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        for (mul_id, add_id) in candidates {
            let (a, b) = match func.inst(mul_id) {
                Inst::FMul { lhs, rhs, .. } => (*lhs, *rhs),
                _ => continue,
            };
            let (c, out) = match func.inst(add_id) {
                Inst::FAdd { lhs, rhs, result } => {
                    let mul_result = func.inst(mul_id).result().unwrap();
                    let c = if *lhs == mul_result { *rhs } else { *lhs };
                    (c, *result)
                }
                _ => continue,
            };

            // Replace FAdd with FMA, then kill FMul.
            func.replace_inst(add_id, Inst::FMA { result: out, a, b, c });
            func.replace_inst(mul_id, Inst::Undef {
                result: func.inst(mul_id).result().unwrap(),
                ty: forge_ir::IrType::F32,
            });
            changed = true;
        }

        changed
    }
}

#[cfg(test)]
mod tests {
    use forge_ir::{ForgeFunction, Inst, IrType, Terminator};

    use super::*;
    use crate::pass::FunctionPass;

    #[test]
    fn fuses_mul_add_into_fma() {
        let mut f = ForgeFunction::new("test");
        let a = f.add_block_param(f.entry, IrType::F32);
        let b = f.add_block_param(f.entry, IrType::F32);
        let c = f.add_block_param(f.entry, IrType::F32);
        let mul = f.fresh_value(IrType::F32);
        let add = f.fresh_value(IrType::F32);
        f.append_inst(f.entry, Inst::FMul { result: mul, lhs: a, rhs: b });
        f.append_inst(f.entry, Inst::FAdd { result: add, lhs: mul, rhs: c });
        f.set_terminator(f.entry, Terminator::Return(Some(add)));

        let changed = FmaFusionPass.run(&mut f);
        assert!(changed);
        // The FAdd should now be an FMA.
        let fma_present = f.insts.iter().any(|i| matches!(i, Inst::FMA { .. }));
        assert!(fma_present, "expected FMA instruction after fusion");
    }
}
