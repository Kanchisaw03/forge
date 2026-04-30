use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use forge_ir::{ForgeFunction, ForgeModule};
use forge_opt::run_default_pipeline;

use crate::emit::{emit_function, EmitError};

/// Code generation errors.
#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    /// Emitter failure.
    #[error("emit failure: {0}")]
    Emit(#[from] EmitError),
    /// External rustc invocation failure.
    #[error("rustc compile failure: {0}")]
    Rustc(String),
    /// IO failure while creating temporary source.
    #[error("io failure: {0}")]
    Io(#[from] std::io::Error),
}

/// Compiles one IR function into Rust source, after optimization.
pub fn lower_function_to_rust(
    func: &mut ForgeFunction,
    symbol_name: &str,
) -> Result<String, CodegenError> {
    run_default_pipeline(func);
    Ok(emit_function(func, symbol_name)?)
}

/// Compiles all functions in a module to Rust source files.
pub fn lower_module_to_rust(module: &mut ForgeModule) -> Result<Vec<(String, String)>, CodegenError> {
    let mut out = Vec::with_capacity(module.functions.len());
    for func in &mut module.functions {
        let symbol = format!("forge_kernel_{}", func.name);
        let source = lower_function_to_rust(func, &symbol)?;
        out.push((symbol, source));
    }
    Ok(out)
}

/// Compiles generated Rust source with rustc for validation.
pub fn compile_generated_source(source: &str) -> Result<(), CodegenError> {
    let mut path = std::env::temp_dir();
    path.push(unique_file_name("forge_codegen_tmp", "rs"));
    fs::write(&path, source)?;
    let mut out = PathBuf::from(&path);
    out.set_extension("rlib");
    let output = Command::new("rustc")
        .arg("--crate-type=lib")
        .arg("--edition=2021")
        .arg(&path)
        .arg("-o")
        .arg(&out)
        .output()?;
    let _ = fs::remove_file(&path);
    let _ = fs::remove_file(&out);
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    Err(CodegenError::Rustc(stderr))
}

fn unique_file_name(prefix: &str, ext: &str) -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{prefix}_{ts}.{ext}")
}

#[cfg(test)]
mod tests {
    use forge_ir::{ForgeFunction, Inst, IrType, Terminator};

    use super::*;

    #[test]
    fn generated_code_compiles_for_simple_kernel() {
        let mut f = ForgeFunction::new("simple");
        let a = f.add_block_param(
            f.entry,
            IrType::Vector {
                elem: Box::new(IrType::F32),
                lanes: 8,
            },
        );
        let b = f.add_block_param(
            f.entry,
            IrType::Vector {
                elem: Box::new(IrType::F32),
                lanes: 8,
            },
        );
        let out = f.fresh_value(IrType::Vector {
            elem: Box::new(IrType::F32),
            lanes: 8,
        });
        f.append_inst(f.entry, Inst::FAdd { result: out, lhs: a, rhs: b });
        f.set_terminator(f.entry, Terminator::Return(Some(out)));
        let src = lower_function_to_rust(&mut f, "forge_kernel_simple").expect("codegen must succeed");
        compile_generated_source(&src).expect("generated source should compile");
    }
}
