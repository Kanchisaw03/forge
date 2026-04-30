use std::collections::HashMap;

use crate::inst::{Inst, Terminator};
use crate::types::{BlockId, InstId, IrType, Value};

/// A single recorded use of an SSA value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Use {
    /// Instruction where the use occurs.
    pub inst: InstId,
    /// Block containing the use.
    pub block: BlockId,
}

/// Metadata for one SSA value.
#[derive(Clone, PartialEq, Debug)]
pub struct ValueData {
    /// Value handle.
    pub id: Value,
    /// Static IR type.
    pub ty: IrType,
    /// Defining instruction, or `None` for block parameters.
    pub def: Option<InstId>,
    /// Recorded use sites.
    pub uses: Vec<Use>,
}

/// A single SSA basic block.
#[derive(Clone, PartialEq, Debug)]
pub struct BasicBlock {
    /// Block identifier.
    pub id: BlockId,
    /// SSA block parameters (Cranelift-style phis).
    pub params: Vec<Value>,
    /// Instruction IDs in program order.
    pub insts: Vec<InstId>,
    /// Required terminator.
    pub terminator: Terminator,
    /// Predecessor blocks.
    pub preds: Vec<BlockId>,
}

/// Function-level IR container.
#[derive(Clone, PartialEq, Debug)]
pub struct ForgeFunction {
    /// Symbolic function name.
    pub name: String,
    /// Entry block for control-flow traversal.
    pub entry: BlockId,
    /// Value table.
    pub values: Vec<ValueData>,
    /// Instruction table.
    pub insts: Vec<Inst>,
    /// Block table.
    pub blocks: Vec<BasicBlock>,
    /// Fast lookup from value to type.
    pub value_types: HashMap<Value, IrType>,
}

impl ForgeFunction {
    /// Constructs an empty function with one entry block.
    pub fn new(name: impl Into<String>) -> Self {
        let entry = BlockId(0);
        let blocks = vec![BasicBlock {
            id: entry,
            params: Vec::new(),
            insts: Vec::new(),
            terminator: Terminator::Unreachable,
            preds: Vec::new(),
        }];
        Self {
            name: name.into(),
            entry,
            values: Vec::new(),
            insts: Vec::new(),
            blocks,
            value_types: HashMap::new(),
        }
    }

    /// Creates a new basic block and returns its ID.
    pub fn create_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(BasicBlock {
            id,
            params: Vec::new(),
            insts: Vec::new(),
            terminator: Terminator::Unreachable,
            preds: Vec::new(),
        });
        id
    }

    /// Adds a block parameter and returns the value handle.
    pub fn add_block_param(&mut self, block: BlockId, ty: IrType) -> Value {
        let v = self.fresh_value(ty);
        self.values[v.0 as usize].def = None;
        self.block_mut(block).params.push(v);
        v
    }

    /// Allocates a new SSA value of type `ty`.
    pub fn fresh_value(&mut self, ty: IrType) -> Value {
        let id = Value(self.values.len() as u32);
        self.values.push(ValueData {
            id,
            ty: ty.clone(),
            def: None,
            uses: Vec::new(),
        });
        self.value_types.insert(id, ty);
        id
    }

    /// Returns the type of a value.
    pub fn value_type(&self, value: Value) -> Option<&IrType> {
        self.value_types.get(&value)
    }

    /// Appends one instruction to `block`.
    ///
    /// Performance rationale: instructions are stored densely in vectors and
    /// referenced by integer IDs to keep pass traversal cache-friendly.
    pub fn append_inst(&mut self, block: BlockId, inst: Inst) -> InstId {
        let id = InstId(self.insts.len() as u32);
        for operand in inst.operands() {
            if let Some(v) = self.values.get_mut(operand.0 as usize) {
                v.uses.push(Use { inst: id, block });
            }
        }
        if let Some(result) = inst.result() {
            if let Some(v) = self.values.get_mut(result.0 as usize) {
                v.def = Some(id);
            }
        }
        self.insts.push(inst);
        self.block_mut(block).insts.push(id);
        id
    }

    /// Replaces one existing instruction while preserving def-use metadata.
    pub fn replace_inst(&mut self, inst_id: InstId, new_inst: Inst) {
        self.insts[inst_id.0 as usize] = new_inst;
        self.rebuild_uses();
    }

    /// Sets block terminator and updates predecessor metadata.
    pub fn set_terminator(&mut self, block: BlockId, term: Terminator) {
        self.block_mut(block).terminator = term;
        self.rebuild_predecessors();
    }

    /// Returns immutable block reference.
    pub fn block(&self, id: BlockId) -> &BasicBlock {
        &self.blocks[id.0 as usize]
    }

    /// Returns mutable block reference.
    pub fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock {
        &mut self.blocks[id.0 as usize]
    }

    /// Returns immutable instruction reference.
    pub fn inst(&self, id: InstId) -> &Inst {
        &self.insts[id.0 as usize]
    }

    /// Returns mutable instruction reference.
    pub fn inst_mut(&mut self, id: InstId) -> &mut Inst {
        &mut self.insts[id.0 as usize]
    }

    /// Removes instructions by ID and compacts the function.
    pub fn remove_insts(&mut self, removed: &[InstId]) {
        if removed.is_empty() {
            return;
        }
        let mut kill = vec![false; self.insts.len()];
        for id in removed {
            kill[id.0 as usize] = true;
        }
        let mut remap = vec![InstId(0); self.insts.len()];
        let mut new_insts = Vec::with_capacity(self.insts.len() - removed.len());
        for (old_idx, inst) in self.insts.iter().cloned().enumerate() {
            if kill[old_idx] {
                continue;
            }
            let new_id = InstId(new_insts.len() as u32);
            remap[old_idx] = new_id;
            new_insts.push(inst);
        }
        self.insts = new_insts;
        for block in &mut self.blocks {
            block.insts.retain(|id| !kill[id.0 as usize]);
            for id in &mut block.insts {
                *id = remap[id.0 as usize];
            }
        }
        for value in &mut self.values {
            if let Some(def) = value.def {
                if kill[def.0 as usize] {
                    value.def = None;
                } else {
                    value.def = Some(remap[def.0 as usize]);
                }
            }
        }
        self.rebuild_uses();
    }

    /// Recomputes predecessor lists from terminators.
    pub fn rebuild_predecessors(&mut self) {
        for b in &mut self.blocks {
            b.preds.clear();
        }
        for block in 0..self.blocks.len() {
            let bid = BlockId(block as u32);
            for succ in self.blocks[block].terminator.successors() {
                self.blocks[succ.0 as usize].preds.push(bid);
            }
        }
    }

    /// Recomputes all use lists from instructions and terminators.
    pub fn rebuild_uses(&mut self) {
        for value in &mut self.values {
            value.uses.clear();
        }
        for (block_idx, block) in self.blocks.iter().enumerate() {
            let bid = BlockId(block_idx as u32);
            for inst_id in &block.insts {
                let inst = &self.insts[inst_id.0 as usize];
                for operand in inst.operands() {
                    if let Some(v) = self.values.get_mut(operand.0 as usize) {
                        v.uses.push(Use {
                            inst: *inst_id,
                            block: bid,
                        });
                    }
                }
            }
            for operand in block.terminator.operands() {
                if let Some(v) = self.values.get_mut(operand.0 as usize) {
                    v.uses.push(Use {
                        inst: InstId(u32::MAX),
                        block: bid,
                    });
                }
            }
        }
    }

    /// Appends one argument to an incoming edge into `target`.
    pub fn append_edge_arg(&mut self, pred: BlockId, target: BlockId, arg: Value) {
        let term = self.block_mut(pred).terminator.clone();
        let updated = match term {
            Terminator::Jump {
                target: edge_target,
                mut args,
            } if edge_target == target => {
                args.push(arg);
                Terminator::Jump {
                    target: edge_target,
                    args,
                }
            }
            Terminator::Branch {
                cond,
                then_block,
                mut then_args,
                else_block,
                mut else_args,
            } => {
                if then_block == target {
                    then_args.push(arg);
                }
                if else_block == target {
                    else_args.push(arg);
                }
                Terminator::Branch {
                    cond,
                    then_block,
                    then_args,
                    else_block,
                    else_args,
                }
            }
            other => other,
        };
        self.set_terminator(pred, updated);
    }

    /// Returns block order as created.
    pub fn block_ids(&self) -> impl Iterator<Item = BlockId> + '_ {
        (0..self.blocks.len()).map(|i| BlockId(i as u32))
    }
}
