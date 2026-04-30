//! Forge SSA IR and lowering pipeline.

pub mod func;
pub mod inst;
pub mod lower;
pub mod module;
pub mod types;
pub mod verify;

pub use func::{BasicBlock, ForgeFunction, Use, ValueData};
pub use inst::{FloatPred, Inst, IntPred, MemOrdering, Terminator};
pub use lower::{lower_module, LowerError, SsaBuilder};
pub use module::ForgeModule;
pub use types::{BlockId, ConstValue, FuncRef, InstId, IrType, Value};
pub use verify::{verify_function, VerifyError};
