#![cfg(feature = "python")]

use std::path::Path;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::compile::compile_file;

/// Python-exposed kernel wrapper.
#[pyclass]
pub struct CompiledKernel {
    generated_path: String,
}

#[pymethods]
impl CompiledKernel {
    /// Returns the generated source path.
    pub fn path(&self) -> String {
        self.generated_path.clone()
    }
}

/// Compiles a Forge kernel file and returns a Python wrapper.
#[pyfunction]
pub fn compile(path: &str) -> PyResult<CompiledKernel> {
    let out = compile_file(Path::new(path), None)
        .map_err(|e| PyRuntimeError::new_err(format!("compile failed: {e}")))?;
    Ok(CompiledKernel {
        generated_path: out.display().to_string(),
    })
}

/// Python module entrypoint.
#[pymodule]
pub fn forge(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_class::<CompiledKernel>()?;
    Ok(())
}
