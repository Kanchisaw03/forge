use crate::func::ForgeFunction;

/// Top-level IR container.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct ForgeModule {
    /// All compiled functions in this module.
    pub functions: Vec<ForgeFunction>,
}

impl ForgeModule {
    /// Creates an empty IR module.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one function to the module.
    pub fn push_function(&mut self, func: ForgeFunction) {
        self.functions.push(func);
    }

    /// Finds a function by name.
    pub fn function(&self, name: &str) -> Option<&ForgeFunction> {
        self.functions.iter().find(|f| f.name == name)
    }

    /// Finds a mutable function by name.
    pub fn function_mut(&mut self, name: &str) -> Option<&mut ForgeFunction> {
        self.functions.iter_mut().find(|f| f.name == name)
    }
}
