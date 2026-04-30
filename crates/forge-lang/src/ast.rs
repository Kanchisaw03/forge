use std::collections::HashMap;

use crate::span::SourceSpan;

/// Compact symbol identifier used for interned identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol(pub u32);

/// String interning table for identifiers and path segments.
///
/// Invariants:
/// - Each distinct string maps to exactly one `Symbol`.
/// - `resolve(sym)` returns the same text that was interned for `sym`.
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    map: HashMap<String, Symbol>,
    rev: Vec<String>,
}

impl SymbolTable {
    /// Creates an empty symbol table.
    ///
    /// Errors:
    /// - This function does not fail.
    pub fn new() -> Self {
        Self::default()
    }

    /// Interns `text` and returns its symbol.
    ///
    /// Performance rationale: the compiler compares symbols as `u32`, avoiding
    /// repeated string hashing in hot parser and resolver paths.
    ///
    /// Errors:
    /// - Panics if the table would exceed `u32::MAX` entries.
    pub fn intern(&mut self, text: &str) -> Symbol {
        if let Some(sym) = self.map.get(text).copied() {
            return sym;
        }

        let idx = self.rev.len();
        let sym = Symbol(u32::try_from(idx).expect("symbol table exceeded u32 capacity"));
        let owned = text.to_owned();
        self.map.insert(owned.clone(), sym);
        self.rev.push(owned);
        sym
    }

    /// Resolves `symbol` to its original interned string.
    ///
    /// Errors:
    /// - Returns `None` when the symbol is out of bounds for this table.
    pub fn resolve(&self, symbol: Symbol) -> Option<&str> {
        self.rev.get(symbol.0 as usize).map(String::as_str)
    }
}

/// Parsed module root.
#[derive(Debug, Clone)]
pub struct Module<'arena> {
    /// Top-level kernel definitions.
    pub kernels: Vec<KernelDef<'arena>>,
    /// Helper function definitions available for inlining.
    pub functions: Vec<FnDecl<'arena>>,
}

/// Helper function declaration (non-kernel, inlineable by the optimizer).
#[derive(Debug, Clone)]
pub struct FnDecl<'arena> {
    /// Function name.
    pub name: Symbol,
    /// Parameters.
    pub params: Vec<Param<'arena>>,
    /// Optional return type.
    pub ret_ty: Option<&'arena Type<'arena>>,
    /// Function body.
    pub body: Block<'arena>,
    /// Full declaration span.
    pub span: SourceSpan,
}

/// Forge kernel definition.
#[derive(Debug, Clone)]
pub struct KernelDef<'arena> {
    /// Leading attributes (`@kernel`, etc.).
    pub attrs: Vec<Attr>,
    /// Kernel function name.
    pub name: Symbol,
    /// Kernel parameters.
    pub params: Vec<Param<'arena>>,
    /// Optional return type.
    pub ret_ty: Option<&'arena Type<'arena>>,
    /// Kernel body block.
    pub body: Block<'arena>,
    /// Full declaration span.
    pub span: SourceSpan,
}

/// Kernel parameter.
#[derive(Debug, Clone)]
pub struct Param<'arena> {
    /// Parameter name.
    pub name: Symbol,
    /// Parameter type.
    pub ty: &'arena Type<'arena>,
    /// Optional parameter attributes.
    pub attrs: Vec<Attr>,
    /// Parameter span.
    pub span: SourceSpan,
}

/// Block scope.
#[derive(Debug, Clone)]
pub struct Block<'arena> {
    /// Statements belonging to this block.
    pub stmts: &'arena [Stmt<'arena>],
    /// Block span.
    pub span: SourceSpan,
}

/// Statement nodes.
#[derive(Debug, Clone)]
pub enum Stmt<'arena> {
    /// `let` binding.
    Let {
        name: Symbol,
        ty: Option<&'arena Type<'arena>>,
        init: Option<&'arena Expr<'arena>>,
        attrs: Vec<Attr>,
        span: SourceSpan,
    },
    /// Assignment / compound assignment.
    Assign {
        target: &'arena Expr<'arena>,
        op: Option<BinOp>,
        value: &'arena Expr<'arena>,
        span: SourceSpan,
    },
    /// Standalone expression statement.
    Expr(&'arena Expr<'arena>, SourceSpan),
    /// `if` statement.
    If {
        cond: &'arena Expr<'arena>,
        then: Block<'arena>,
        else_: Option<Block<'arena>>,
        span: SourceSpan,
    },
    /// `for` statement.
    For {
        var: Symbol,
        iter: &'arena Expr<'arena>,
        body: Block<'arena>,
        span: SourceSpan,
    },
    /// `while` statement.
    While {
        cond: &'arena Expr<'arena>,
        body: Block<'arena>,
        span: SourceSpan,
    },
    /// `loop` statement.
    Loop {
        body: Block<'arena>,
        span: SourceSpan,
    },
    /// `return` statement.
    Return {
        value: Option<&'arena Expr<'arena>>,
        span: SourceSpan,
    },
    /// `break` statement.
    Break {
        span: SourceSpan,
    },
    /// `continue` statement.
    Continue {
        span: SourceSpan,
    },
}

impl<'arena> Stmt<'arena> {
    /// Returns the statement span.
    ///
    /// Errors:
    /// - This function does not fail.
    pub const fn span(&self) -> SourceSpan {
        match self {
            Stmt::Let { span, .. }
            | Stmt::Assign { span, .. }
            | Stmt::If { span, .. }
            | Stmt::For { span, .. }
            | Stmt::While { span, .. }
            | Stmt::Loop { span, .. }
            | Stmt::Return { span, .. }
            | Stmt::Break { span }
            | Stmt::Continue { span }
            | Stmt::Expr(_, span) => *span,
        }
    }
}

/// Type syntax.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type<'arena> {
    Named(Symbol),
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
    Bool,
    Usize,
    Slice {
        elem: &'arena Type<'arena>,
        mutable: bool,
    },
    Ptr {
        elem: &'arena Type<'arena>,
        mutable: bool,
    },
    Vector {
        elem: &'arena Type<'arena>,
        lanes: u8,
    },
    Void,
}

/// Supported attributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attr {
    Kernel,
    Shared,
    Global,
    Register,
    Unknown(Symbol),
}

/// Literal values.
#[derive(Debug, Clone, PartialEq)]
pub enum Lit {
    Int(u64),
    Float(f64),
    Bool(bool),
    String(Symbol),
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    LogicalAnd,
    LogicalOr,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    LogicalNot,
    Neg,
    Ref,
    RefMut,
    Deref,
    BitNot,
}

/// Built-in intrinsic operations recognized by the frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicKind {
    ThreadX,
    ThreadY,
    ThreadZ,
    BlockX,
    BlockY,
    BlockZ,
    WarpLaneId,
    ForgeSyncBlock,
    ForgeSyncWarp,
    ForgeAtomicAdd,
    ForgeExp,
    ForgeSqrt,
    ForgeFma,
}

/// Expression nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr<'arena> {
    Lit {
        lit: Lit,
        span: SourceSpan,
    },
    Ident {
        symbol: Symbol,
        span: SourceSpan,
    },
    Path {
        segments: &'arena [Symbol],
        span: SourceSpan,
    },
    Intrinsic {
        kind: IntrinsicKind,
        span: SourceSpan,
    },
    BinOp {
        op: BinOp,
        lhs: &'arena Expr<'arena>,
        rhs: &'arena Expr<'arena>,
        span: SourceSpan,
    },
    UnOp {
        op: UnOp,
        expr: &'arena Expr<'arena>,
        span: SourceSpan,
    },
    Call {
        func: &'arena Expr<'arena>,
        args: &'arena [&'arena Expr<'arena>],
        span: SourceSpan,
    },
    Index {
        base: &'arena Expr<'arena>,
        index: &'arena Expr<'arena>,
        span: SourceSpan,
    },
    Field {
        base: &'arena Expr<'arena>,
        field: Symbol,
        span: SourceSpan,
    },
    Cast {
        expr: &'arena Expr<'arena>,
        ty: &'arena Type<'arena>,
        span: SourceSpan,
    },
    Range {
        lo: Option<&'arena Expr<'arena>>,
        hi: Option<&'arena Expr<'arena>>,
        inclusive: bool,
        span: SourceSpan,
    },
}

impl<'arena> Expr<'arena> {
    /// Returns the expression span.
    ///
    /// Errors:
    /// - Panics if called on leaf nodes where span must be supplied by caller.
    pub const fn span(&self) -> SourceSpan {
        match self {
            Expr::Lit { span, .. } | Expr::Ident { span, .. } => *span,
            Expr::Path { span, .. }
            | Expr::Intrinsic { span, .. }
            | Expr::BinOp { span, .. }
            | Expr::UnOp { span, .. }
            | Expr::Call { span, .. }
            | Expr::Index { span, .. }
            | Expr::Field { span, .. }
            | Expr::Cast { span, .. }
            | Expr::Range { span, .. } => *span,
        }
    }
}
