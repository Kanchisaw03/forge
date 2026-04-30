use crate::types::{BlockId, ConstValue, FuncRef, IrType, Value};

/// Integer comparison predicates.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum IntPred {
    Eq,
    Ne,
    Slt,
    Sle,
    Sgt,
    Sge,
    Ult,
    Ule,
    Ugt,
    Uge,
}

/// Floating-point comparison predicates.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FloatPred {
    Oeq,
    One,
    Olt,
    Ole,
    Ogt,
    Oge,
    Uno,
}

/// Memory ordering model for atomic and fence instructions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MemOrdering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

/// Instruction set for Forge SSA IR.
#[derive(Clone, PartialEq, Debug)]
pub enum Inst {
    IAdd { result: Value, lhs: Value, rhs: Value },
    ISub { result: Value, lhs: Value, rhs: Value },
    IMul { result: Value, lhs: Value, rhs: Value },
    IDiv {
        result: Value,
        lhs: Value,
        rhs: Value,
        signed: bool,
    },
    IRem {
        result: Value,
        lhs: Value,
        rhs: Value,
        signed: bool,
    },
    INeg { result: Value, operand: Value },
    FAdd { result: Value, lhs: Value, rhs: Value },
    FSub { result: Value, lhs: Value, rhs: Value },
    FMul { result: Value, lhs: Value, rhs: Value },
    FDiv { result: Value, lhs: Value, rhs: Value },
    FNeg { result: Value, operand: Value },
    FMA {
        result: Value,
        a: Value,
        b: Value,
        c: Value,
    },
    And { result: Value, lhs: Value, rhs: Value },
    Or { result: Value, lhs: Value, rhs: Value },
    Xor { result: Value, lhs: Value, rhs: Value },
    Not { result: Value, operand: Value },
    Shl { result: Value, lhs: Value, rhs: Value },
    Shr {
        result: Value,
        lhs: Value,
        rhs: Value,
        signed: bool,
    },
    ICmp {
        result: Value,
        pred: IntPred,
        lhs: Value,
        rhs: Value,
    },
    FCmp {
        result: Value,
        pred: FloatPred,
        lhs: Value,
        rhs: Value,
    },
    Load {
        result: Value,
        ty: IrType,
        ptr: Value,
        align: u32,
    },
    Store { ptr: Value, value: Value, align: u32 },
    GetElementPtr {
        result: Value,
        base: Value,
        indices: Vec<Value>,
    },
    Alloca {
        result: Value,
        ty: IrType,
        count: Option<Value>,
    },
    AtomicAdd {
        result: Value,
        ptr: Value,
        value: Value,
        ordering: MemOrdering,
    },
    AtomicCAS {
        result: Value,
        ptr: Value,
        cmp: Value,
        new: Value,
        ordering: MemOrdering,
    },
    Prefetch { ptr: Value, locality: u8 },
    ZExt {
        result: Value,
        operand: Value,
        ty: IrType,
    },
    SExt {
        result: Value,
        operand: Value,
        ty: IrType,
    },
    Trunc {
        result: Value,
        operand: Value,
        ty: IrType,
    },
    FPToI {
        result: Value,
        operand: Value,
        ty: IrType,
        signed: bool,
    },
    IToFP {
        result: Value,
        operand: Value,
        ty: IrType,
        signed: bool,
    },
    FPExt {
        result: Value,
        operand: Value,
        ty: IrType,
    },
    FPTrunc {
        result: Value,
        operand: Value,
        ty: IrType,
    },
    Bitcast {
        result: Value,
        operand: Value,
        ty: IrType,
    },
    Splat {
        result: Value,
        scalar: Value,
        lanes: u8,
    },
    Extract { result: Value, vec: Value, idx: u8 },
    Insert {
        result: Value,
        vec: Value,
        scalar: Value,
        idx: u8,
    },
    Shuffle {
        result: Value,
        a: Value,
        b: Value,
        mask: Vec<u8>,
    },
    Blend {
        result: Value,
        a: Value,
        b: Value,
        mask: Value,
    },
    ThreadIdxX { result: Value },
    ThreadIdxY { result: Value },
    ThreadIdxZ { result: Value },
    BlockIdxX { result: Value },
    BlockIdxY { result: Value },
    /// Block Z index (third grid dimension).
    BlockIdxZ { result: Value },
    BlockDimX { result: Value },
    BlockDimY { result: Value },
    /// Block Z dimension size.
    BlockDimZ { result: Value },
    GridDimX { result: Value },
    /// Grid Y dimension size.
    GridDimY { result: Value },
    /// Grid Z dimension size.
    GridDimZ { result: Value },
    WarpLaneId { result: Value },
    SyncBlock,
    SyncWarp,
    MemFence { ordering: MemOrdering },
    /// Gather: load from base[indices[i]] for each lane.
    Gather {
        result: Value,
        base: Value,
        indices: Value,
        ty: IrType,
        scale: u8,
    },
    /// Scatter: store value lanes to base[indices[i]].
    Scatter {
        base: Value,
        indices: Value,
        value: Value,
        scale: u8,
    },
    /// lane-wise absolute value.
    FAbs { result: Value, operand: Value },
    /// lane-wise sqrt.
    FSqrt { result: Value, operand: Value },
    /// lane-wise min.
    FMin { result: Value, lhs: Value, rhs: Value },
    /// lane-wise max.
    FMax { result: Value, lhs: Value, rhs: Value },
    Const { result: Value, value: ConstValue },
    Undef { result: Value, ty: IrType },
    Call {
        result: Option<Value>,
        func: FuncRef,
        args: Vec<Value>,
    },
}

impl Inst {
    /// Returns the SSA value defined by this instruction, if any.
    pub const fn result(&self) -> Option<Value> {
        match self {
            Inst::IAdd { result, .. }
            | Inst::ISub { result, .. }
            | Inst::IMul { result, .. }
            | Inst::IDiv { result, .. }
            | Inst::IRem { result, .. }
            | Inst::INeg { result, .. }
            | Inst::FAdd { result, .. }
            | Inst::FSub { result, .. }
            | Inst::FMul { result, .. }
            | Inst::FDiv { result, .. }
            | Inst::FNeg { result, .. }
            | Inst::FMA { result, .. }
            | Inst::And { result, .. }
            | Inst::Or { result, .. }
            | Inst::Xor { result, .. }
            | Inst::Not { result, .. }
            | Inst::Shl { result, .. }
            | Inst::Shr { result, .. }
            | Inst::ICmp { result, .. }
            | Inst::FCmp { result, .. }
            | Inst::Load { result, .. }
            | Inst::GetElementPtr { result, .. }
            | Inst::Alloca { result, .. }
            | Inst::AtomicAdd { result, .. }
            | Inst::AtomicCAS { result, .. }
            | Inst::ZExt { result, .. }
            | Inst::SExt { result, .. }
            | Inst::Trunc { result, .. }
            | Inst::FPToI { result, .. }
            | Inst::IToFP { result, .. }
            | Inst::FPExt { result, .. }
            | Inst::FPTrunc { result, .. }
            | Inst::Bitcast { result, .. }
            | Inst::Splat { result, .. }
            | Inst::Extract { result, .. }
            | Inst::Insert { result, .. }
            | Inst::Shuffle { result, .. }
            | Inst::Blend { result, .. }
            | Inst::ThreadIdxX { result }
            | Inst::ThreadIdxY { result }
            | Inst::ThreadIdxZ { result }
            | Inst::BlockIdxX { result }
            | Inst::BlockIdxY { result }
            | Inst::BlockIdxZ { result }
            | Inst::BlockDimX { result }
            | Inst::BlockDimY { result }
            | Inst::BlockDimZ { result }
            | Inst::GridDimX { result }
            | Inst::GridDimY { result }
            | Inst::GridDimZ { result }
            | Inst::WarpLaneId { result }
            | Inst::Gather { result, .. }
            | Inst::FAbs { result, .. }
            | Inst::FSqrt { result, .. }
            | Inst::FMin { result, .. }
            | Inst::FMax { result, .. }
            | Inst::Const { result, .. }
            | Inst::Undef { result, .. } => Some(*result),
            Inst::Call { result, .. } => *result,
            Inst::Store { .. }
            | Inst::Prefetch { .. }
            | Inst::Scatter { .. }
            | Inst::SyncBlock
            | Inst::SyncWarp
            | Inst::MemFence { .. } => None,
        }
    }

    /// Returns all operand values consumed by this instruction.
    pub fn operands(&self) -> Vec<Value> {
        match self {
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
            | Inst::FCmp { lhs, rhs, .. } => vec![*lhs, *rhs],
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
            | Inst::Bitcast { operand, .. } => vec![*operand],
            Inst::FMA { a, b, c, .. } => vec![*a, *b, *c],
            Inst::Load { ptr, .. } => vec![*ptr],
            Inst::Store { ptr, value, .. } => vec![*ptr, *value],
            Inst::GetElementPtr { base, indices, .. } => {
                let mut ops = Vec::with_capacity(indices.len() + 1);
                ops.push(*base);
                ops.extend(indices.iter().copied());
                ops
            }
            Inst::Alloca { count, .. } => count.iter().copied().collect(),
            Inst::AtomicAdd { ptr, value, .. } => vec![*ptr, *value],
            Inst::AtomicCAS { ptr, cmp, new, .. } => vec![*ptr, *cmp, *new],
            Inst::Prefetch { ptr, .. } => vec![*ptr],
            Inst::Splat { scalar, .. } => vec![*scalar],
            Inst::Extract { vec, .. } => vec![*vec],
            Inst::Insert { vec, scalar, .. } => vec![*vec, *scalar],
            Inst::Shuffle { a, b, .. } => vec![*a, *b],
            Inst::Blend { a, b, mask, .. } => vec![*a, *b, *mask],
            Inst::Call { args, .. } => args.clone(),
            Inst::Gather { base, indices, .. } => vec![*base, *indices],
            Inst::Scatter { base, indices, value, .. } => vec![*base, *indices, *value],
            Inst::FAbs { operand, .. } | Inst::FSqrt { operand, .. } => vec![*operand],
            Inst::FMin { lhs, rhs, .. } | Inst::FMax { lhs, rhs, .. } => vec![*lhs, *rhs],
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
            | Inst::Undef { .. } => Vec::new(),
        }
    }

    /// Replaces all occurrences of `old` as an operand with `new`.
    pub fn remap_operand(&mut self, old: Value, new: Value) {
        macro_rules! remap {
            ($v:expr) => { if *$v == old { *$v = new; } };
        }
        match self {
            Inst::IAdd { lhs, rhs, .. } | Inst::ISub { lhs, rhs, .. }
            | Inst::IMul { lhs, rhs, .. } | Inst::And { lhs, rhs, .. }
            | Inst::Or { lhs, rhs, .. } | Inst::Xor { lhs, rhs, .. }
            | Inst::Shl { lhs, rhs, .. } | Inst::FAdd { lhs, rhs, .. }
            | Inst::FSub { lhs, rhs, .. } | Inst::FMul { lhs, rhs, .. }
            | Inst::FDiv { lhs, rhs, .. } | Inst::FMin { lhs, rhs, .. }
            | Inst::FMax { lhs, rhs, .. } => { remap!(lhs); remap!(rhs); }
            Inst::IDiv { lhs, rhs, .. } | Inst::IRem { lhs, rhs, .. }
            | Inst::Shr { lhs, rhs, .. } | Inst::ICmp { lhs, rhs, .. }
            | Inst::FCmp { lhs, rhs, .. } => { remap!(lhs); remap!(rhs); }
            Inst::FMA { a, b, c, .. } => { remap!(a); remap!(b); remap!(c); }
            Inst::INeg { operand, .. } | Inst::FNeg { operand, .. }
            | Inst::Not { operand, .. } | Inst::ZExt { operand, .. }
            | Inst::SExt { operand, .. } | Inst::Trunc { operand, .. }
            | Inst::FPToI { operand, .. } | Inst::IToFP { operand, .. }
            | Inst::FPExt { operand, .. } | Inst::FPTrunc { operand, .. }
            | Inst::Bitcast { operand, .. } | Inst::FAbs { operand, .. }
            | Inst::FSqrt { operand, .. } => { remap!(operand); }
            Inst::Load { ptr, .. } => { remap!(ptr); }
            Inst::Store { ptr, value, .. } => { remap!(ptr); remap!(value); }
            Inst::AtomicAdd { ptr, value, .. } => { remap!(ptr); remap!(value); }
            Inst::AtomicCAS { ptr, cmp, new, .. } => { remap!(ptr); remap!(cmp); remap!(new); }
            Inst::Prefetch { ptr, .. } => { remap!(ptr); }
            Inst::Splat { scalar, .. } => { remap!(scalar); }
            Inst::Extract { vec, .. } => { remap!(vec); }
            Inst::Insert { vec, scalar, .. } => { remap!(vec); remap!(scalar); }
            Inst::Shuffle { a, b, .. } => { remap!(a); remap!(b); }
            Inst::Blend { a, b, mask, .. } => { remap!(a); remap!(b); remap!(mask); }
            Inst::Gather { base, indices, .. } => { remap!(base); remap!(indices); }
            Inst::Scatter { base, indices, value, .. } => { remap!(base); remap!(indices); remap!(value); }
            Inst::GetElementPtr { base, indices, .. } => {
                remap!(base);
                for idx in indices.iter_mut() { if *idx == old { *idx = new; } }
            }
            Inst::Call { args, .. } => {
                for a in args.iter_mut() { if *a == old { *a = new; } }
            }
            Inst::Alloca { count, .. } => {
                if let Some(c) = count.as_mut() { if *c == old { *c = new; } }
            }
            // Instructions with no mutable operands.
            Inst::ThreadIdxX { .. } | Inst::ThreadIdxY { .. } | Inst::ThreadIdxZ { .. }
            | Inst::BlockIdxX { .. } | Inst::BlockIdxY { .. } | Inst::BlockIdxZ { .. }
            | Inst::BlockDimX { .. } | Inst::BlockDimY { .. } | Inst::BlockDimZ { .. }
            | Inst::GridDimX { .. } | Inst::GridDimY { .. } | Inst::GridDimZ { .. }
            | Inst::WarpLaneId { .. } | Inst::SyncBlock | Inst::SyncWarp
            | Inst::MemFence { .. } | Inst::Const { .. } | Inst::Undef { .. } => {}
        }
    }
}

/// Control-flow terminator for a basic block.
#[derive(Clone, PartialEq, Debug)]
pub enum Terminator {
    /// Return from function with an optional value.
    Return(Option<Value>),
    /// Unconditional branch with block arguments.
    Jump { target: BlockId, args: Vec<Value> },
    /// Conditional branch with block arguments for each edge.
    Branch {
        cond: Value,
        then_block: BlockId,
        then_args: Vec<Value>,
        else_block: BlockId,
        else_args: Vec<Value>,
    },
    /// Unreachable control-flow sink.
    Unreachable,
}

impl Terminator {
    /// Returns all operand values referenced by this terminator.
    pub fn operands(&self) -> Vec<Value> {
        match self {
            Terminator::Return(Some(v)) => vec![*v],
            Terminator::Return(None) | Terminator::Unreachable => Vec::new(),
            Terminator::Jump { args, .. } => args.clone(),
            Terminator::Branch {
                cond,
                then_args,
                else_args,
                ..
            } => {
                let mut out = Vec::with_capacity(1 + then_args.len() + else_args.len());
                out.push(*cond);
                out.extend(then_args.iter().copied());
                out.extend(else_args.iter().copied());
                out
            }
        }
    }

    /// Returns the list of successor blocks.
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Terminator::Jump { target, .. } => vec![*target],
            Terminator::Branch {
                then_block,
                else_block,
                ..
            } => vec![*then_block, *else_block],
            Terminator::Return(_) | Terminator::Unreachable => Vec::new(),
        }
    }
}
