use forge_ir::ForgeFunction;

/// Function-level optimization pass contract.
pub trait FunctionPass {
    /// Human-readable pass name.
    fn name(&self) -> &'static str;
    /// Runs the pass and returns `true` when it modified IR.
    fn run(&mut self, func: &mut ForgeFunction) -> bool;
}

/// Repeating function pass pipeline.
pub struct PassManager {
    passes: Vec<Box<dyn FunctionPass>>,
}

impl PassManager {
    /// Creates an empty pass manager.
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Appends one pass to the pipeline.
    pub fn add_pass<P: FunctionPass + 'static>(&mut self, pass: P) {
        self.passes.push(Box::new(pass));
    }

    /// Runs all passes to a fixed point with a bounded iteration count.
    pub fn run(&mut self, func: &mut ForgeFunction) {
        for _ in 0..20 {
            let mut changed = false;
            for pass in &mut self.passes {
                changed |= pass.run(func);
            }
            if !changed {
                break;
            }
        }
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}
