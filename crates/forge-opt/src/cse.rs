use std::collections::HashMap;

use forge_ir::{ForgeFunction, Inst, InstId, Terminator, Value};

use crate::domtree::compute_domtree;
use crate::pass::FunctionPass;

/// Common subexpression elimination pass.
#[derive(Default)]
pub struct CommonSubexpressionElimination;

impl FunctionPass for CommonSubexpressionElimination {
    fn name(&self) -> &'static str {
        "CommonSubexpressionElimination"
    }

    fn run(&mut self, func: &mut ForgeFunction) -> bool {
        let dom = compute_domtree(func);
        let mut inst_block = HashMap::new();
        for block in func.block_ids() {
            for inst_id in &func.block(block).insts {
                inst_block.insert(*inst_id, block);
            }
        }

        let mut available: HashMap<CseKey, (InstId, Value)> = HashMap::new();
        let mut rewrites: Vec<(Value, Value)> = Vec::new();
        let mut remove = Vec::new();

        for block in func.block_ids() {
            for inst_id in func.block(block).insts.clone() {
                let inst = func.inst(inst_id).clone();
                let Some(result) = inst.result() else {
                    continue;
                };
                let Some(key) = cse_key(&inst) else {
                    continue;
                };
                if let Some((prior_inst, prior_value)) = available.get(&key).copied() {
                    let prior_block = inst_block[&prior_inst];
                    if dom.dominates(prior_block, block) {
                        rewrites.push((result, prior_value));
                        remove.push(inst_id);
                        continue;
                    }
                }
                available.insert(key, (inst_id, result));
            }
        }

        if rewrites.is_empty() {
            return false;
        }

        for (old, new) in rewrites {
            rewrite_value(func, old, new);
        }
        func.remove_insts(&remove);
        true
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
enum CseKey {
    IAdd(Value, Value),
    ISub(Value, Value),
    IMul(Value, Value),
    FAdd(Value, Value),
    FSub(Value, Value),
    FMul(Value, Value),
    And(Value, Value),
    Or(Value, Value),
    Xor(Value, Value),
    Shl(Value, Value),
    Shr(Value, Value, bool),
    ICmp(forge_ir::IntPred, Value, Value),
    FCmp(forge_ir::FloatPred, Value, Value),
    GetElementPtr(Value, Vec<Value>),
}

fn cse_key(inst: &Inst) -> Option<CseKey> {
    match inst {
        Inst::IAdd { lhs, rhs, .. } => Some(CseKey::IAdd(*lhs, *rhs)),
        Inst::ISub { lhs, rhs, .. } => Some(CseKey::ISub(*lhs, *rhs)),
        Inst::IMul { lhs, rhs, .. } => Some(CseKey::IMul(*lhs, *rhs)),
        Inst::FAdd { lhs, rhs, .. } => Some(CseKey::FAdd(*lhs, *rhs)),
        Inst::FSub { lhs, rhs, .. } => Some(CseKey::FSub(*lhs, *rhs)),
        Inst::FMul { lhs, rhs, .. } => Some(CseKey::FMul(*lhs, *rhs)),
        Inst::And { lhs, rhs, .. } => Some(CseKey::And(*lhs, *rhs)),
        Inst::Or { lhs, rhs, .. } => Some(CseKey::Or(*lhs, *rhs)),
        Inst::Xor { lhs, rhs, .. } => Some(CseKey::Xor(*lhs, *rhs)),
        Inst::Shl { lhs, rhs, .. } => Some(CseKey::Shl(*lhs, *rhs)),
        Inst::Shr { lhs, rhs, signed, .. } => Some(CseKey::Shr(*lhs, *rhs, *signed)),
        Inst::ICmp { pred, lhs, rhs, .. } => Some(CseKey::ICmp(*pred, *lhs, *rhs)),
        Inst::FCmp { pred, lhs, rhs, .. } => Some(CseKey::FCmp(*pred, *lhs, *rhs)),
        Inst::GetElementPtr { base, indices, .. } => {
            Some(CseKey::GetElementPtr(*base, indices.clone()))
        }
        _ => None,
    }
}

fn rewrite_value(func: &mut ForgeFunction, old: Value, new: Value) {
    for inst in &mut func.insts {
        rewrite_inst_operand(inst, old, new);
    }
    for block in &mut func.blocks {
        match &mut block.terminator {
            Terminator::Return(v) => {
                if let Some(val) = v {
                    if *val == old {
                        *val = new;
                    }
                }
            }
            Terminator::Jump { args, .. } => {
                for arg in args {
                    if *arg == old {
                        *arg = new;
                    }
                }
            }
            Terminator::Branch {
                cond,
                then_args,
                else_args,
                ..
            } => {
                if *cond == old {
                    *cond = new;
                }
                for arg in then_args {
                    if *arg == old {
                        *arg = new;
                    }
                }
                for arg in else_args {
                    if *arg == old {
                        *arg = new;
                    }
                }
            }
            Terminator::Unreachable => {}
        }
    }
    func.rebuild_uses();
}

fn rewrite_inst_operand(inst: &mut Inst, old: Value, new: Value) {
    macro_rules! rw {
        ($v:expr) => {
            if *$v == old {
                *$v = new;
            }
        };
    }
    match inst {
        Inst::IAdd { lhs, rhs, .. }
        | Inst::ISub { lhs, rhs, .. }
        | Inst::IMul { lhs, rhs, .. }
        | Inst::IDiv { lhs, rhs, .. }
        | Inst::IRem { lhs, rhs, .. }
        | Inst::FAdd { lhs, rhs, .. }
        | Inst::FSub { lhs, rhs, .. }
        | Inst::FMul { lhs, rhs, .. }
        | Inst::FDiv { lhs, rhs, .. }
        | Inst::And { lhs, rhs, .. }
        | Inst::Or { lhs, rhs, .. }
        | Inst::Xor { lhs, rhs, .. }
        | Inst::Shl { lhs, rhs, .. }
        | Inst::Shr { lhs, rhs, .. }
        | Inst::ICmp { lhs, rhs, .. }
        | Inst::FCmp { lhs, rhs, .. } => {
            rw!(lhs);
            rw!(rhs);
        }
        Inst::INeg { operand, .. }
        | Inst::FNeg { operand, .. }
        | Inst::Not { operand, .. }
        | Inst::ZExt { operand, .. }
        | Inst::SExt { operand, .. }
        | Inst::Trunc { operand, .. }
        | Inst::FPToI { operand, .. }
        | Inst::IToFP { operand, .. }
        | Inst::FPExt { operand, .. }
        | Inst::FPTrunc { operand, .. }
        | Inst::Bitcast { operand, .. } => rw!(operand),
        Inst::FMA { a, b, c, .. } => {
            rw!(a);
            rw!(b);
            rw!(c);
        }
        Inst::Load { ptr, .. } | Inst::Prefetch { ptr, .. } => rw!(ptr),
        Inst::Store { ptr, value, .. } => {
            rw!(ptr);
            rw!(value);
        }
        Inst::GetElementPtr { base, indices, .. } => {
            rw!(base);
            for idx in indices {
                rw!(idx);
            }
        }
        Inst::Alloca { count, .. } => {
            if let Some(c) = count {
                rw!(c);
            }
        }
        Inst::AtomicAdd { ptr, value, .. } => {
            rw!(ptr);
            rw!(value);
        }
        Inst::AtomicCAS { ptr, cmp, new: n, .. } => {
            rw!(ptr);
            rw!(cmp);
            rw!(n);
        }
        Inst::Splat { scalar, .. } => rw!(scalar),
        Inst::Extract { vec, .. } => rw!(vec),
        Inst::Insert { vec, scalar, .. } => {
            rw!(vec);
            rw!(scalar);
        }
        Inst::Shuffle { a, b, .. } => {
            rw!(a);
            rw!(b);
        }
        Inst::Blend { a, b, mask, .. } => {
            rw!(a);
            rw!(b);
            rw!(mask);
        }
        Inst::Call { args, .. } => {
            for arg in args {
                rw!(arg);
            }
        }
        Inst::Gather { base, indices, .. } => {
            rw!(base);
            rw!(indices);
        }
        Inst::Scatter { base, indices, value, .. } => {
            rw!(base);
            rw!(indices);
            rw!(value);
        }
        Inst::FAbs { operand, .. } | Inst::FSqrt { operand, .. } => rw!(operand),
        Inst::FMin { lhs, rhs, .. } | Inst::FMax { lhs, rhs, .. } => {
            rw!(lhs);
            rw!(rhs);
        }
        Inst::ThreadIdxX { .. }
        | Inst::ThreadIdxY { .. }
        | Inst::ThreadIdxZ { .. }
        | Inst::BlockIdxX { .. }
        | Inst::BlockIdxY { .. }
        | Inst::BlockIdxZ { .. }
        | Inst::BlockDimX { .. }
        | Inst::BlockDimY { .. }
        | Inst::BlockDimZ { .. }
        | Inst::GridDimX { .. }
        | Inst::GridDimY { .. }
        | Inst::GridDimZ { .. }
        | Inst::WarpLaneId { .. }
        | Inst::SyncBlock
        | Inst::SyncWarp
        | Inst::MemFence { .. }
        | Inst::Const { .. }
        | Inst::Undef { .. } => {}
    }
}
