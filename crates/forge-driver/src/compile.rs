use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use bumpalo::Bump;
use forge_codegen::{lower_module_to_rust, CodegenError};
use forge_ir::lower_module;
use forge_lang::{resolve_module, Lexer, ParseSession};
use forge_opt::run_default_pipeline;

/// Dump stages for `forge dump`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DumpStage {
    Lex,
    Ast,
    Ir,
    IrOpt,
    Codegen,
}

impl DumpStage {
    /// Parses CLI stage name.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "lex" => Some(Self::Lex),
            "ast" => Some(Self::Ast),
            "ir" => Some(Self::Ir),
            "ir-opt" => Some(Self::IrOpt),
            "codegen" => Some(Self::Codegen),
            _ => None,
        }
    }
}

/// Compile pipeline output.
pub struct CompileArtifact {
    /// Generated Rust sources keyed by symbol.
    pub generated: Vec<(String, String)>,
}

/// Runs full compile pipeline and returns generated Rust source units.
pub fn compile_source(source: &str) -> Result<CompileArtifact> {
    let arena = Bump::new();
    let (tokens, lex_errors) = Lexer::new(source).lex_all();
    if !lex_errors.is_empty() {
        return Err(anyhow!("lexing failed:\n{lex_errors:#?}"));
    }
    let tokens = arena.alloc_slice_fill_iter(tokens);
    let mut parser = ParseSession::new(&arena, tokens);
    let module = parser.parse_module();
    if !parser.errors.is_empty() {
        return Err(anyhow!("parsing failed:\n{:#?}", parser.errors));
    }
    let (_scopes, res_errors) = resolve_module(&module, &parser.symbols);
    if !res_errors.is_empty() {
        return Err(anyhow!("name resolution failed:\n{res_errors:#?}"));
    }
    let mut ir_module = lower_module(&module, &parser.symbols)
        .map_err(|errs| anyhow!("lowering failed:\n{errs:#?}"))?;
    for func in &mut ir_module.functions {
        run_default_pipeline(func);
    }
    let generated = lower_module_to_rust(&mut ir_module).map_err(map_codegen_error)?;
    Ok(CompileArtifact { generated })
}

/// Compiles input file and writes generated output file.
pub fn compile_file(input: &Path, output: Option<&Path>) -> Result<PathBuf> {
    let source = fs::read_to_string(input)
        .with_context(|| format!("failed to read `{}`", input.display()))?;
    let artifact = compile_source(&source)?;
    let mut content = String::new();
    for (_name, src) in artifact.generated {
        content.push_str(&src);
        content.push('\n');
    }
    let out = output
        .map(PathBuf::from)
        .unwrap_or_else(|| input.with_extension("generated.rs"));
    fs::write(&out, content)
        .with_context(|| format!("failed to write `{}`", out.display()))?;
    Ok(out)
}

/// Runs syntax+semantic checks only.
pub fn check_file(input: &Path) -> Result<()> {
    let source = fs::read_to_string(input)
        .with_context(|| format!("failed to read `{}`", input.display()))?;
    let arena = Bump::new();
    let (tokens, lex_errors) = Lexer::new(&source).lex_all();
    if !lex_errors.is_empty() {
        return Err(anyhow!("lexing failed:\n{lex_errors:#?}"));
    }
    let tokens = arena.alloc_slice_fill_iter(tokens);
    let mut parser = ParseSession::new(&arena, tokens);
    let module = parser.parse_module();
    if !parser.errors.is_empty() {
        return Err(anyhow!("parsing failed:\n{:#?}", parser.errors));
    }
    let (_scope, res_errors) = resolve_module(&module, &parser.symbols);
    if !res_errors.is_empty() {
        return Err(anyhow!("resolution failed:\n{res_errors:#?}"));
    }
    Ok(())
}

/// Dumps one intermediate stage as text.
pub fn dump_stage(input: &Path, stage: DumpStage) -> Result<String> {
    let source = fs::read_to_string(input)
        .with_context(|| format!("failed to read `{}`", input.display()))?;
    let arena = Bump::new();
    let (tokens, lex_errors) = Lexer::new(&source).lex_all();
    if !lex_errors.is_empty() {
        return Err(anyhow!("lexing failed:\n{lex_errors:#?}"));
    }
    if matches!(stage, DumpStage::Lex) {
        return Ok(format!("{tokens:#?}"));
    }
    let tokens = arena.alloc_slice_fill_iter(tokens);
    let mut parser = ParseSession::new(&arena, tokens);
    let module = parser.parse_module();
    if !parser.errors.is_empty() {
        return Err(anyhow!("parsing failed:\n{:#?}", parser.errors));
    }
    if matches!(stage, DumpStage::Ast) {
        return Ok(format!("{module:#?}"));
    }
    let (_scope, res_errors) = resolve_module(&module, &parser.symbols);
    if !res_errors.is_empty() {
        return Err(anyhow!("resolution failed:\n{res_errors:#?}"));
    }
    let mut ir_module = lower_module(&module, &parser.symbols)
        .map_err(|errs| anyhow!("lowering failed:\n{errs:#?}"))?;
    if matches!(stage, DumpStage::Ir) {
        return Ok(format!("{ir_module:#?}"));
    }
    for func in &mut ir_module.functions {
        run_default_pipeline(func);
    }
    if matches!(stage, DumpStage::IrOpt) {
        return Ok(format!("{ir_module:#?}"));
    }
    let generated = lower_module_to_rust(&mut ir_module).map_err(map_codegen_error)?;
    Ok(generated
        .into_iter()
        .map(|(_, src)| src)
        .collect::<Vec<_>>()
        .join("\n\n"))
}

fn map_codegen_error(err: CodegenError) -> anyhow::Error {
    anyhow!("code generation failed: {err}")
}
