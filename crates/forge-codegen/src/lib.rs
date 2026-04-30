//! Forge IR to AVX2 Rust code generation.

pub mod avx2;
pub mod emit;
pub mod lower;
pub mod regalloc;
pub mod schedule;

pub use avx2::Avx2Selector;
pub use emit::{emit_function, EmitError};
pub use lower::{compile_generated_source, lower_function_to_rust, lower_module_to_rust, CodegenError};
pub use regalloc::{LinearScanAllocator, LiveInterval, Location, PhysReg};
pub use schedule::{ListScheduler, PortClass};
