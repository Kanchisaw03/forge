use std::fmt;

/// SSA value handle used as an index into a function's value table.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Value(pub u32);

impl Value {
    /// Returns the raw numeric identifier.
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Basic block handle used as an index into a function's block table.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BlockId(pub u32);

impl BlockId {
    /// Returns the raw numeric identifier.
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Instruction handle used as an index into a function's instruction table.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct InstId(pub u32);

impl InstId {
    /// Returns the raw numeric identifier.
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Constant payload for `Inst::Const`.
#[derive(Clone, PartialEq, Debug)]
pub enum ConstValue {
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
    /// 32-bit signed integer.
    I32(i32),
    /// 64-bit signed integer.
    I64(i64),
    /// 32-bit unsigned integer.
    U32(u32),
    /// 64-bit unsigned integer.
    U64(u64),
    /// Boolean constant.
    Bool(bool),
}

impl ConstValue {
    /// Returns the IR type represented by this constant.
    pub const fn ir_type(&self) -> IrType {
        match self {
            ConstValue::F32(_) => IrType::F32,
            ConstValue::F64(_) => IrType::F64,
            ConstValue::I32(_) => IrType::I32,
            ConstValue::I64(_) => IrType::I64,
            ConstValue::U32(_) => IrType::U32,
            ConstValue::U64(_) => IrType::U64,
            ConstValue::Bool(_) => IrType::Bool,
        }
    }
}

/// Symbolic function reference used by `Inst::Call`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FuncRef {
    /// Stable function name.
    pub name: String,
}

impl FuncRef {
    /// Creates a function reference from `name`.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// Complete Forge IR type set.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum IrType {
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 32-bit unsigned integer.
    U32,
    /// 64-bit unsigned integer.
    U64,
    /// Boolean.
    Bool,
    /// Pointer to an element type.
    Ptr { elem: Box<IrType>, mutable: bool },
    /// Fixed-width SIMD vector.
    Vector { elem: Box<IrType>, lanes: u8 },
    /// No value.
    Void,
}

impl IrType {
    /// Returns true when the type is a floating-point scalar or vector.
    pub fn is_float(&self) -> bool {
        match self {
            IrType::F32 | IrType::F64 => true,
            IrType::Vector { elem, .. } => matches!(elem.as_ref(), IrType::F32 | IrType::F64),
            _ => false,
        }
    }

    /// Returns true when the type is an integer scalar or vector.
    pub fn is_integer(&self) -> bool {
        match self {
            IrType::I32 | IrType::I64 | IrType::U32 | IrType::U64 | IrType::Bool => true,
            IrType::Vector { elem, .. } => {
                matches!(
                    elem.as_ref(),
                    IrType::I32 | IrType::I64 | IrType::U32 | IrType::U64 | IrType::Bool
                )
            }
            _ => false,
        }
    }

    /// Returns the pointee type for pointers.
    pub fn pointee(&self) -> Option<&IrType> {
        match self {
            IrType::Ptr { elem, .. } => Some(elem.as_ref()),
            _ => None,
        }
    }
}

impl fmt::Display for IrType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrType::F32 => write!(f, "f32"),
            IrType::F64 => write!(f, "f64"),
            IrType::I32 => write!(f, "i32"),
            IrType::I64 => write!(f, "i64"),
            IrType::U32 => write!(f, "u32"),
            IrType::U64 => write!(f, "u64"),
            IrType::Bool => write!(f, "bool"),
            IrType::Ptr { elem, mutable } => {
                if *mutable {
                    write!(f, "*mut {elem}")
                } else {
                    write!(f, "*const {elem}")
                }
            }
            IrType::Vector { elem, lanes } => write!(f, "{elem}x{lanes}"),
            IrType::Void => write!(f, "void"),
        }
    }
}
