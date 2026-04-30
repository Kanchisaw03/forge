use std::collections::{HashMap, HashSet, VecDeque};

use forge_ir::{ForgeFunction, Inst, InstId, Value};

use crate::pass::FunctionPass;

/// Per-value divergence classification.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Divergence {
    Uniform,
    Divergent,
}

/// Divergence analysis result.
#[derive(Clone, Debug, Default)]
pub struct DivergenceInfo {
    /// Classification for each SSA value.
    pub classification: HashMap<Value, Divergence>,
    /// Blocks whose branch condition is divergent.
    pub divergent_branches: HashSet<forge_ir::BlockId>,
}

/// Forward dataflow divergence analysis pass.
#[derive(Default)]
pub struct DivergenceAnalysis {
    /// Last computed divergence state.
    pub info: DivergenceInfo,
}

impl FunctionPass for DivergenceAnalysis {
    fn name(&self) -> &'static str {
        "DivergenceAnalysis"
    }

    fn run(&mut self, func: &mut ForgeFunction) -> bool {
        self.info = analyze_divergence(func);
        false
    }
}

/// Computes divergence information for one function.
pub fn analyze_divergence(func: &ForgeFunction) -> DivergenceInfo {
    let mut classification = HashMap::new();
    for value in &func.values {
        classification.insert(value.id, Divergence::Uniform);
    }
    let mut users: HashMap<Value, Vec<InstId>> = HashMap::new();
    for (idx, inst) in func.insts.iter().enumerate() {
        let inst_id = InstId(idx as u32);
        for op in inst.operands() {
            users.entry(op).or_default().push(inst_id);
        }
    }
    let mut queue = VecDeque::new();
    for (inst_idx, inst) in func.insts.iter().enumerate() {
        if let Some(result) = inst.result() {
            if is_seed_divergent(inst) {
                classification.insert(result, Divergence::Divergent);
                queue.extend(users.get(&result).cloned().unwrap_or_default());
            }
        }
        if matches!(inst, Inst::ThreadIdxX { .. } | Inst::ThreadIdxY { .. } | Inst::ThreadIdxZ { .. } | Inst::WarpLaneId { .. }) {
            queue.push_back(InstId(inst_idx as u32));
        }
    }

    while let Some(inst_id) = queue.pop_front() {
        let inst = &func.insts[inst_id.0 as usize];
        let Some(result) = inst.result() else {
            continue;
        };
        let new_state = if is_seed_divergent(inst) {
            Divergence::Divergent
        } else if inst
            .operands()
            .iter()
            .any(|v| matches!(classification.get(v), Some(Divergence::Divergent)))
        {
            Divergence::Divergent
        } else {
            Divergence::Uniform
        };
        let prev = classification.get(&result).copied().unwrap_or(Divergence::Uniform);
        if prev != new_state {
            classification.insert(result, new_state);
            queue.extend(users.get(&result).cloned().unwrap_or_default());
        }
    }

    let mut divergent_branches = HashSet::new();
    for block in func.block_ids() {
        if let forge_ir::Terminator::Branch { cond, .. } = &func.block(block).terminator {
            if matches!(classification.get(cond), Some(Divergence::Divergent)) {
                divergent_branches.insert(block);
            }
        }
    }

    DivergenceInfo {
        classification,
        divergent_branches,
    }
}

fn is_seed_divergent(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::ThreadIdxX { .. }
            | Inst::ThreadIdxY { .. }
            | Inst::ThreadIdxZ { .. }
            | Inst::WarpLaneId { .. }
    )
}

#[cfg(test)]
mod tests {
    use forge_ir::{ConstValue, IrType, Terminator};

    use super::*;

    #[test]
    fn marks_thread_ids_divergent_and_params_uniform() {
        let mut f = ForgeFunction::new("div");
        let n = f.add_block_param(f.entry, IrType::U32);
        let tid = f.fresh_value(IrType::U32);
        f.append_inst(f.entry, Inst::ThreadIdxX { result: tid });
        let cond = f.fresh_value(IrType::Bool);
        f.append_inst(
            f.entry,
            Inst::ICmp {
                result: cond,
                pred: forge_ir::IntPred::Ult,
                lhs: tid,
                rhs: n,
            },
        );
        let c0 = f.fresh_value(IrType::U32);
        f.append_inst(
            f.entry,
            Inst::Const {
                result: c0,
                value: ConstValue::U32(0),
            },
        );
        f.set_terminator(f.entry, Terminator::Return(Some(c0)));
        f.rebuild_uses();
        let info = analyze_divergence(&f);
        assert_eq!(info.classification.get(&n), Some(&Divergence::Uniform));
        assert_eq!(info.classification.get(&tid), Some(&Divergence::Divergent));
        assert_eq!(info.classification.get(&cond), Some(&Divergence::Divergent));
    }
}
