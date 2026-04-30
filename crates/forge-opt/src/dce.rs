use forge_ir::{ForgeFunction, Inst, InstId};

use crate::pass::FunctionPass;

/// Dead-code elimination over SSA instructions.
#[derive(Default)]
pub struct DeadCodeElimination;

impl FunctionPass for DeadCodeElimination {
    fn name(&self) -> &'static str {
        "DeadCodeElimination"
    }

    fn run(&mut self, func: &mut ForgeFunction) -> bool {
        func.rebuild_uses();
        let mut remove = Vec::new();
        for block in func.block_ids() {
            for inst_id in &func.block(block).insts {
                let inst = func.inst(*inst_id);
                let Some(result) = inst.result() else {
                    continue;
                };
                let has_uses = func.values[result.0 as usize]
                    .uses
                    .iter()
                    .any(|u| u.inst != InstId(u32::MAX));
                if !has_uses && !has_side_effect(inst) {
                    remove.push(*inst_id);
                }
            }
        }
        if remove.is_empty() {
            return false;
        }
        func.remove_insts(&remove);
        true
    }
}

fn has_side_effect(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Store { .. }
            | Inst::AtomicAdd { .. }
            | Inst::AtomicCAS { .. }
            | Inst::SyncBlock
            | Inst::SyncWarp
            | Inst::MemFence { .. }
            | Inst::Call { .. }
    )
}

#[cfg(test)]
mod tests {
    use forge_ir::{ConstValue, IrType, Terminator};

    use super::*;

    #[test]
    fn removes_dead_add_instruction() {
        let mut f = ForgeFunction::new("dce");
        let a = f.fresh_value(IrType::F32);
        let b = f.fresh_value(IrType::F32);
        let dead = f.fresh_value(IrType::F32);
        f.append_inst(
            f.entry,
            Inst::Const {
                result: a,
                value: ConstValue::F32(1.0),
            },
        );
        f.append_inst(
            f.entry,
            Inst::Const {
                result: b,
                value: ConstValue::F32(2.0),
            },
        );
        f.append_inst(f.entry, Inst::FAdd { result: dead, lhs: a, rhs: b });
        f.set_terminator(f.entry, Terminator::Return(None));
        let mut pass = DeadCodeElimination;
        let changed = pass.run(&mut f);
        assert!(changed);
        assert!(f.insts.len() < 3, "dead instruction should be removed");
    }
}
