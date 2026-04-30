//! Forge optimization passes.

pub mod constfold;
pub mod cse;
pub mod divergence;
pub mod dce;
pub mod domtree;
pub mod fma_fusion;
pub mod memlayout;
pub mod pass;
pub mod simplify_cfg;
pub mod strength_reduction;
pub mod vectorize;

pub use constfold::ConstantFolding;
pub use cse::CommonSubexpressionElimination;
pub use divergence::{analyze_divergence, Divergence, DivergenceAnalysis, DivergenceInfo};
pub use dce::DeadCodeElimination;
pub use domtree::{compute_domtree, DomTree};
pub use fma_fusion::FmaFusionPass;
pub use memlayout::MemoryLayoutOptimization;
pub use pass::{FunctionPass, PassManager};
pub use simplify_cfg::SimplifyCfgPass;
pub use strength_reduction::StrengthReductionPass;
pub use vectorize::{gcd_test_no_dependency, VectorizationPass};

use forge_ir::ForgeFunction;

/// Runs the full Forge optimization pipeline matching the design specification.
///
/// Pass order:
/// 1. `SimplifyCfgPass`          — clean up empty blocks from SSA construction.
/// 2. `ConstantFolding`           — fold constants before other passes.
/// 3. `StrengthReductionPass`     — lower mul/div by power-of-two to shifts.
/// 4. `DeadCodeElimination`       — remove newly-dead values.
/// 5. `CommonSubexpressionElimination` — deduplicate equal expressions.
/// 6. `FmaFusionPass`             — fuse FMul+FAdd → FMA before vectorize.
/// 7. `DivergenceAnalysis`        — classify uniform vs. divergent values.
/// 8. `VectorizationPass`         — widen scalar loops to f32x8 (AVX2 width).
/// 9. `MemoryLayoutOptimization`  — promote strided accesses to coalesced.
/// 10. `DeadCodeElimination`      — final cleanup after vectorisation.
pub fn run_default_pipeline(func: &mut ForgeFunction) {
    let mut pm = PassManager::new();
    pm.add_pass(SimplifyCfgPass);
    pm.add_pass(ConstantFolding);
    pm.add_pass(StrengthReductionPass);
    pm.add_pass(DeadCodeElimination);
    pm.add_pass(CommonSubexpressionElimination);
    pm.add_pass(FmaFusionPass);
    pm.add_pass(DivergenceAnalysis::default());
    pm.add_pass(VectorizationPass);
    pm.add_pass(MemoryLayoutOptimization);
    pm.add_pass(DeadCodeElimination);
    pm.run(func);
}
