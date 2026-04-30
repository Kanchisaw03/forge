use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use forge_ir::{ForgeFunction, Inst, InstId};

/// Port class model for scheduler heuristics.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PortClass {
    P0,
    P1,
    Load,
    Store,
    Other,
}

/// List scheduler for one basic block.
pub struct ListScheduler;

impl ListScheduler {
    /// Returns scheduled instruction order for one block.
    pub fn schedule_block(func: &ForgeFunction, block: forge_ir::BlockId) -> Vec<InstId> {
        let insts = func.block(block).insts.clone();
        if insts.len() <= 1 {
            return insts;
        }
        let deps = build_deps(func, &insts);
        let mut indegree: HashMap<InstId, usize> = HashMap::new();
        let mut succs: HashMap<InstId, Vec<InstId>> = HashMap::new();
        for (to, froms) in &deps {
            indegree.insert(*to, froms.len());
            for from in froms {
                succs.entry(*from).or_default().push(*to);
            }
        }
        for id in &insts {
            indegree.entry(*id).or_insert(0);
        }

        let critical = critical_path_lengths(&insts, &succs);
        let mut ready: BinaryHeap<(usize, Reverse<u32>, InstId)> = BinaryHeap::new();
        for id in &insts {
            if indegree[id] == 0 {
                ready.push((critical[id], Reverse(port_weight(func.inst(*id))), *id));
            }
        }

        let mut out = Vec::with_capacity(insts.len());
        let mut scheduled = HashSet::new();
        while let Some((_cp, _port, id)) = ready.pop() {
            if !scheduled.insert(id) {
                continue;
            }
            out.push(id);
            if let Some(nexts) = succs.get(&id) {
                for n in nexts {
                    let deg = indegree.get_mut(n).expect("node exists");
                    *deg -= 1;
                    if *deg == 0 {
                        ready.push((critical[n], Reverse(port_weight(func.inst(*n))), *n));
                    }
                }
            }
        }
        if out.len() == insts.len() {
            out
        } else {
            insts
        }
    }
}

fn build_deps(func: &ForgeFunction, insts: &[InstId]) -> HashMap<InstId, Vec<InstId>> {
    let mut def_inst = HashMap::new();
    for id in insts {
        if let Some(v) = func.inst(*id).result() {
            def_inst.insert(v, *id);
        }
    }
    let mut deps = HashMap::new();
    for id in insts {
        let mut froms = Vec::new();
        for op in func.inst(*id).operands() {
            if let Some(def) = def_inst.get(&op).copied() {
                froms.push(def);
            }
        }
        deps.insert(*id, froms);
    }
    deps
}

fn critical_path_lengths(
    insts: &[InstId],
    succs: &HashMap<InstId, Vec<InstId>>,
) -> HashMap<InstId, usize> {
    let mut memo = HashMap::new();
    fn dfs(
        id: InstId,
        succs: &HashMap<InstId, Vec<InstId>>,
        memo: &mut HashMap<InstId, usize>,
    ) -> usize {
        if let Some(v) = memo.get(&id).copied() {
            return v;
        }
        let max_child = succs
            .get(&id)
            .map(|s| s.iter().map(|n| dfs(*n, succs, memo)).max().unwrap_or(0))
            .unwrap_or(0);
        let len = 1 + max_child;
        memo.insert(id, len);
        len
    }
    for id in insts {
        dfs(*id, succs, &mut memo);
    }
    memo
}

fn port_weight(inst: &Inst) -> u32 {
    match classify_port(inst) {
        PortClass::P0 => 0,
        PortClass::P1 => 1,
        PortClass::Load => 2,
        PortClass::Store => 3,
        PortClass::Other => 4,
    }
}

fn classify_port(inst: &Inst) -> PortClass {
    match inst {
        Inst::FMul { .. } | Inst::FDiv { .. } | Inst::FMA { .. } | Inst::Shl { .. } | Inst::Blend { .. } => PortClass::P0,
        Inst::FAdd { .. } | Inst::FSub { .. } | Inst::IAdd { .. } | Inst::ISub { .. } | Inst::IMul { .. } => PortClass::P1,
        Inst::Load { .. } => PortClass::Load,
        Inst::Store { .. } => PortClass::Store,
        _ => PortClass::Other,
    }
}
