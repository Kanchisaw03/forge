use std::collections::HashSet;

use forge_ir::{BlockId, ForgeFunction, Terminator};

use crate::pass::FunctionPass;

/// Simplifies the control-flow graph by:
/// 1. Merging a block that has exactly one predecessor into its predecessor
///    when the predecessor's only successor is that block.
/// 2. Removing unreachable blocks (blocks with no predecessors except entry).
///
/// This is critical after SSA construction which can produce chains of
/// single-instruction blocks with redundant Jump terminators.
pub struct SimplifyCfgPass;

impl FunctionPass for SimplifyCfgPass {
    fn name(&self) -> &'static str {
        "simplify-cfg"
    }

    fn run(&mut self, func: &mut ForgeFunction) -> bool {
        let mut changed = false;

        // Pass 1: remove unreachable blocks.
        let reachable = compute_reachable(func);
        let all_block_ids: Vec<BlockId> = func.block_ids().collect();
        for bid in &all_block_ids {
            if *bid == func.entry {
                continue;
            }
            if !reachable.contains(bid) {
                // Clear instructions from unreachable block so DCE can finish.
                func.block_mut(*bid).insts.clear();
                func.block_mut(*bid).terminator = Terminator::Unreachable;
                changed = true;
            }
        }

        // Pass 2: merge single-successor/single-predecessor chains.
        // Collect pairs where: pred has exactly one successor (Jump to succ),
        // and succ has exactly one predecessor.
        let pairs: Vec<(BlockId, BlockId)> = func
            .block_ids()
            .filter_map(|pred| {
                if let Terminator::Jump { target, ref args } = func.block(pred).terminator {
                    if args.is_empty() && func.block(target).preds.len() == 1 && target != func.entry {
                        Some((pred, target))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        for (pred, succ) in pairs {
            // Move all instructions from succ into pred.
            let succ_insts = func.block(succ).insts.clone();
            let succ_term = func.block(succ).terminator.clone();
            for inst_id in succ_insts {
                func.block_mut(pred).insts.push(inst_id);
            }
            func.set_terminator(pred, succ_term);
            // Clear merged block.
            func.block_mut(succ).insts.clear();
            func.block_mut(succ).terminator = Terminator::Unreachable;
            changed = true;
        }

        if changed {
            func.rebuild_predecessors();
            func.rebuild_uses();
        }
        changed
    }
}

fn compute_reachable(func: &ForgeFunction) -> HashSet<BlockId> {
    let mut visited = HashSet::new();
    let mut stack = vec![func.entry];
    while let Some(bid) = stack.pop() {
        if !visited.insert(bid) {
            continue;
        }
        for succ in func.block(bid).terminator.successors() {
            stack.push(succ);
        }
    }
    visited
}
