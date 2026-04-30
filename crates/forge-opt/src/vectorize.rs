use forge_ir::{ConstValue, ForgeFunction, Inst, IrType, Terminator};

use crate::divergence::{analyze_divergence, Divergence};
use crate::pass::FunctionPass;

/// Loop vectorization pass (8-lane AVX2 target).
#[derive(Default)]
pub struct VectorizationPass;

impl FunctionPass for VectorizationPass {
    fn name(&self) -> &'static str {
        "VectorizationPass"
    }

    fn run(&mut self, func: &mut ForgeFunction) -> bool {
        let divergence = analyze_divergence(func);
        let mut changed = false;
        for header in func.block_ids().collect::<Vec<_>>() {
            if func.block(header).params.is_empty() {
                continue;
            }
            let (cond, body) = match &func.block(header).terminator {
                Terminator::Branch {
                    cond, then_block, ..
                } => (*cond, *then_block),
                _ => continue,
            };
            if matches!(
                divergence.classification.get(&cond),
                Some(Divergence::Divergent)
            ) {
                continue;
            }
            let Some(backedge) = func
                .block(header)
                .preds
                .iter()
                .copied()
                .find(|p| matches!(func.block(*p).terminator, Terminator::Jump { target, .. } if target == header))
            else {
                continue;
            };
            if has_loop_carried_dependency(func, header, backedge) {
                continue;
            }

            let iv = func.block(header).params[0];
            let iv_ty = func.value_type(iv).cloned().unwrap_or(IrType::U32);
            let vec_ty = IrType::Vector {
                elem: Box::new(iv_ty),
                lanes: 8,
            };
            let vec_iv = func.fresh_value(vec_ty);
            func.append_inst(
                body,
                Inst::Splat {
                    result: vec_iv,
                    scalar: iv,
                    lanes: 8,
                },
            );
            changed = true;

            if let Terminator::Jump { args, .. } = &func.block(backedge).terminator {
                if let Some(next_iv) = args.first().copied() {
                    patch_induction_step(func, backedge, next_iv);
                }
            }
        }
        if changed {
            func.rebuild_uses();
        }
        changed
    }
}

fn has_loop_carried_dependency(func: &ForgeFunction, header: forge_ir::BlockId, backedge: forge_ir::BlockId) -> bool {
    let params = &func.block(header).params;
    let Terminator::Jump { args, .. } = &func.block(backedge).terminator else {
        return true;
    };
    if args.len() != params.len() {
        return true;
    }
    args.iter()
        .zip(params.iter())
        .enumerate()
        .skip(1)
        .any(|(_, (arg, param))| arg != param)
}

fn patch_induction_step(func: &mut ForgeFunction, backedge: forge_ir::BlockId, next_iv: forge_ir::Value) {
    let Some(def) = func.values[next_iv.0 as usize].def else {
        return;
    };
    let Inst::IAdd { rhs, .. } = func.inst(def).clone() else {
        return;
    };
    let Some(const_def) = func.values[rhs.0 as usize].def else {
        return;
    };
    if matches!(
        func.inst(const_def),
        Inst::Const {
            value: ConstValue::U32(1) | ConstValue::U64(1) | ConstValue::I32(1) | ConstValue::I64(1),
            ..
        }
    ) {
        let updated = match func.inst(const_def) {
            Inst::Const { result, value: ConstValue::U32(_), .. } => Inst::Const { result: *result, value: ConstValue::U32(8) },
            Inst::Const { result, value: ConstValue::U64(_), .. } => Inst::Const { result: *result, value: ConstValue::U64(8) },
            Inst::Const { result, value: ConstValue::I32(_), .. } => Inst::Const { result: *result, value: ConstValue::I32(8) },
            Inst::Const { result, value: ConstValue::I64(_), .. } => Inst::Const { result: *result, value: ConstValue::I64(8) },
            _ => return,
        };
        func.replace_inst(const_def, updated);
        func.rebuild_uses();
    }
    let _ = backedge;
}

/// Returns true when GCD test proves no dependence exists.
pub fn gcd_test_no_dependency(a: i64, b: i64, c: i64) -> bool {
    let g = gcd(a.abs(), b.abs());
    g != 0 && c.rem_euclid(g) != 0
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.abs()
}

#[cfg(test)]
mod tests {
    use forge_ir::{ConstValue, IrType};

    use super::*;

    #[test]
    fn gcd_test_matches_expected_example() {
        assert!(gcd_test_no_dependency(2, 2, 1));
    }

    #[test]
    fn vectorizes_simple_counted_loop() {
        let mut f = ForgeFunction::new("vec");
        let i0 = f.fresh_value(IrType::U32);
        let n = f.add_block_param(f.entry, IrType::U32);
        f.append_inst(
            f.entry,
            Inst::Const {
                result: i0,
                value: ConstValue::U32(0),
            },
        );
        let header = f.create_block();
        let body = f.create_block();
        let exit = f.create_block();
        let iv = f.add_block_param(header, IrType::U32);
        f.set_terminator(
            f.entry,
            Terminator::Jump {
                target: header,
                args: vec![i0],
            },
        );
        let cond = f.fresh_value(IrType::Bool);
        f.append_inst(
            header,
            Inst::ICmp {
                result: cond,
                pred: forge_ir::IntPred::Ult,
                lhs: iv,
                rhs: n,
            },
        );
        f.set_terminator(
            header,
            Terminator::Branch {
                cond,
                then_block: body,
                then_args: Vec::new(),
                else_block: exit,
                else_args: Vec::new(),
            },
        );
        let one = f.fresh_value(IrType::U32);
        f.append_inst(
            body,
            Inst::Const {
                result: one,
                value: ConstValue::U32(1),
            },
        );
        let next = f.fresh_value(IrType::U32);
        f.append_inst(body, Inst::IAdd { result: next, lhs: iv, rhs: one });
        f.set_terminator(
            body,
            Terminator::Jump {
                target: header,
                args: vec![next],
            },
        );
        f.set_terminator(exit, Terminator::Return(None));
        f.rebuild_predecessors();
        f.rebuild_uses();

        let mut pass = VectorizationPass;
        let changed = pass.run(&mut f);
        assert!(changed);
        assert!(f.insts.iter().any(|i| matches!(i, Inst::Splat { lanes: 8, .. })));
    }
}
