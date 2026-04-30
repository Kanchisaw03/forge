use std::collections::HashMap;

use forge_ir::{BlockId, ForgeFunction};

/// Immediate-dominator information for one function.
#[derive(Clone, Debug, Default)]
pub struct DomTree {
    /// `idom[b] = immediate dominator of b`, with entry mapped to itself.
    pub idom: HashMap<BlockId, BlockId>,
}

impl DomTree {
    /// Returns true when `a` dominates `b`.
    pub fn dominates(&self, a: BlockId, mut b: BlockId) -> bool {
        if a == b {
            return true;
        }
        while let Some(idom) = self.idom.get(&b).copied() {
            if idom == b {
                break;
            }
            if idom == a {
                return true;
            }
            b = idom;
        }
        false
    }
}

/// Computes dominators via Lengauer-Tarjan.
pub fn compute_domtree(func: &ForgeFunction) -> DomTree {
    let n_blocks = func.blocks.len();
    if n_blocks == 0 {
        return DomTree::default();
    }

    let mut parent = vec![None::<usize>; n_blocks];
    let mut semi = vec![0usize; n_blocks];
    let mut label = vec![0usize; n_blocks];
    let mut ancestor = vec![None::<usize>; n_blocks];
    let mut bucket = vec![Vec::<usize>::new(); n_blocks];
    let mut vertex = Vec::<usize>::new();
    let mut dfs_index = vec![None::<usize>; n_blocks];
    let mut pred = vec![Vec::<usize>::new(); n_blocks];

    fn dfs(
        func: &ForgeFunction,
        v: usize,
        parent_v: Option<usize>,
        parent: &mut [Option<usize>],
        semi: &mut [usize],
        label: &mut [usize],
        vertex: &mut Vec<usize>,
        dfs_index: &mut [Option<usize>],
    ) {
        let idx = vertex.len();
        dfs_index[v] = Some(idx);
        semi[v] = idx;
        label[v] = v;
        vertex.push(v);
        parent[v] = parent_v;
        for succ in func.blocks[v].terminator.successors() {
            let s = succ.0 as usize;
            if dfs_index[s].is_none() {
                dfs(
                    func,
                    s,
                    Some(v),
                    parent,
                    semi,
                    label,
                    vertex,
                    dfs_index,
                );
            }
        }
    }

    dfs(
        func,
        func.entry.0 as usize,
        None,
        &mut parent,
        &mut semi,
        &mut label,
        &mut vertex,
        &mut dfs_index,
    );

    for b in &vertex {
        let block = BlockId(*b as u32);
        for p in &func.block(block).preds {
            pred[*b].push(p.0 as usize);
        }
    }

    fn compress(v: usize, ancestor: &mut [Option<usize>], label: &mut [usize], semi: &[usize]) {
        if let Some(a) = ancestor[v] {
            if ancestor[a].is_some() {
                compress(a, ancestor, label, semi);
                if semi[label[a]] < semi[label[v]] {
                    label[v] = label[a];
                }
                ancestor[v] = ancestor[a];
            }
        }
    }

    fn eval(v: usize, ancestor: &mut [Option<usize>], label: &mut [usize], semi: &[usize]) -> usize {
        if ancestor[v].is_none() {
            return label[v];
        }
        compress(v, ancestor, label, semi);
        label[v]
    }

    fn link(v: usize, w: usize, ancestor: &mut [Option<usize>]) {
        ancestor[w] = Some(v);
    }

    let mut idom = vec![None::<usize>; n_blocks];
    for i in (1..vertex.len()).rev() {
        let w = vertex[i];
        let mut s = semi[w];
        for v in &pred[w] {
            if dfs_index[*v].is_none() {
                continue;
            }
            let u = eval(*v, &mut ancestor, &mut label, &semi);
            s = s.min(semi[u]);
        }
        semi[w] = s;
        let semi_vertex = vertex[s];
        bucket[semi_vertex].push(w);
        let p = parent[w].expect("non-entry has parent");
        link(p, w, &mut ancestor);
        let moved = std::mem::take(&mut bucket[p]);
        for v in moved {
            let u = eval(v, &mut ancestor, &mut label, &semi);
            if semi[u] < semi[v] {
                idom[v] = Some(u);
            } else {
                idom[v] = Some(p);
            }
        }
    }

    for i in 1..vertex.len() {
        let w = vertex[i];
        let id = idom[w].expect("must have idom");
        if id != vertex[semi[w]] {
            idom[w] = idom[id];
        }
    }

    let mut out = DomTree::default();
    let entry = func.entry.0 as usize;
    out.idom.insert(func.entry, func.entry);
    for &v in &vertex {
        if v == entry {
            continue;
        }
        if let Some(id) = idom[v] {
            out.idom.insert(BlockId(v as u32), BlockId(id as u32));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use forge_ir::{Inst, Terminator};

    use super::*;

    #[test]
    fn computes_expected_dominators_for_diamond_cfg() {
        let mut f = ForgeFunction::new("dom");
        let then_b = f.create_block();
        let else_b = f.create_block();
        let join_b = f.create_block();
        let c = f.fresh_value(forge_ir::IrType::Bool);
        f.append_inst(f.entry, Inst::Undef { result: c, ty: forge_ir::IrType::Bool });
        f.set_terminator(
            f.entry,
            Terminator::Branch {
                cond: c,
                then_block: then_b,
                then_args: Vec::new(),
                else_block: else_b,
                else_args: Vec::new(),
            },
        );
        f.set_terminator(
            then_b,
            Terminator::Jump {
                target: join_b,
                args: Vec::new(),
            },
        );
        f.set_terminator(
            else_b,
            Terminator::Jump {
                target: join_b,
                args: Vec::new(),
            },
        );
        f.set_terminator(join_b, Terminator::Return(None));
        f.rebuild_predecessors();

        let dom = compute_domtree(&f);
        assert!(dom.dominates(f.entry, then_b));
        assert!(dom.dominates(f.entry, else_b));
        assert!(dom.dominates(f.entry, join_b));
        assert!(!dom.dominates(then_b, else_b));
    }
}
