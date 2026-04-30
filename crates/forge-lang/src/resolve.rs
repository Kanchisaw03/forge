use std::collections::HashMap;

use thiserror::Error;

use crate::ast::{Block, Expr, KernelDef, Module, Stmt, Symbol, SymbolTable};
use crate::span::SourceSpan;

/// Scope tree root and nodes built during resolution.
#[derive(Debug, Clone, Default)]
pub struct ScopeTree {
    /// Scope nodes in insertion order.
    pub scopes: Vec<ScopeNode>,
    /// Root scope id.
    pub root: usize,
}

/// Single lexical scope.
#[derive(Debug, Clone, Default)]
pub struct ScopeNode {
    /// Parent scope id, or `None` for the root.
    pub parent: Option<usize>,
    /// Child scope ids.
    pub children: Vec<usize>,
    /// Symbol bindings introduced in this scope.
    pub bindings: HashMap<Symbol, SourceSpan>,
}

/// Name resolution diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ResolutionErrorKind {
    /// Encountered an identifier that has no visible binding.
    #[error("undefined identifier `{name}`")]
    UndefinedIdentifier { name: String },

    /// Encountered a redefinition in the same or nested active scope.
    #[error("redefinition of `{name}` is not allowed (shadowing disabled)")]
    Redefinition { name: String },
}

/// Resolution error with source span.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{kind} at {span}")]
pub struct ResolutionError {
    /// Resolution problem kind.
    pub kind: ResolutionErrorKind,
    /// Precise source span for the diagnostic.
    pub span: SourceSpan,
}

/// Resolves names for all kernels and returns the built scope tree plus errors.
///
/// Invariants:
/// - `module` and `symbols` must come from the same parse session.
///
/// Errors:
/// - Returns all resolvable diagnostics; does not short-circuit at first failure.
pub fn resolve_module(
    module: &Module<'_>,
    symbols: &SymbolTable,
) -> (ScopeTree, Vec<ResolutionError>) {
    let mut resolver = Resolver::new(symbols);
    resolver.resolve_module(module);
    (resolver.scope_tree, resolver.errors)
}

struct Resolver<'sym> {
    symbols: &'sym SymbolTable,
    scope_tree: ScopeTree,
    active_scopes: Vec<usize>,
    errors: Vec<ResolutionError>,
}

impl<'sym> Resolver<'sym> {
    fn new(symbols: &'sym SymbolTable) -> Self {
        let root = ScopeNode::default();
        Self {
            symbols,
            scope_tree: ScopeTree {
                scopes: vec![root],
                root: 0,
            },
            active_scopes: vec![0],
            errors: Vec::new(),
        }
    }

    fn resolve_module(&mut self, module: &Module<'_>) {
        for func in &module.functions {
            let scope = self.push_scope();
            for param in &func.params {
                self.define_symbol(param.name, param.span);
            }
            self.resolve_block(&func.body);
            self.pop_scope(scope);
        }
        for kernel in &module.kernels {
            self.resolve_kernel(kernel);
        }
    }

    fn resolve_kernel(&mut self, kernel: &KernelDef<'_>) {
        let kernel_scope = self.push_scope();

        for param in &kernel.params {
            self.define_symbol(param.name, param.span);
        }

        self.resolve_block(&kernel.body);

        self.pop_scope(kernel_scope);
    }

    fn resolve_block(&mut self, block: &Block<'_>) {
        let block_scope = self.push_scope();

        for stmt in block.stmts {
            self.resolve_stmt(stmt);
        }

        self.pop_scope(block_scope);
    }

    fn resolve_stmt(&mut self, stmt: &Stmt<'_>) {
        match stmt {
            Stmt::Let {
                name,
                init,
                attrs: _,
                span,
                ..
            } => {
                if let Some(init) = init {
                    self.resolve_expr(init);
                }
                self.define_symbol(*name, *span);
            }
            Stmt::Assign { target, value, .. } => {
                self.resolve_expr(target);
                self.resolve_expr(value);
            }
            Stmt::Expr(expr, _) => self.resolve_expr(expr),
            Stmt::If {
                cond,
                then,
                else_,
                ..
            } => {
                self.resolve_expr(cond);
                self.resolve_block(then);
                if let Some(else_block) = else_ {
                    self.resolve_block(else_block);
                }
            }
            Stmt::For {
                var,
                iter,
                body,
                span,
            } => {
                self.resolve_expr(iter);
                let loop_scope = self.push_scope();
                self.define_symbol(*var, *span);
                self.resolve_block(body);
                self.pop_scope(loop_scope);
            }
            Stmt::While { cond, body, .. } => {
                self.resolve_expr(cond);
                self.resolve_block(body);
            }
            Stmt::Loop { body, .. } => self.resolve_block(body),
            Stmt::Return { value, .. } => {
                if let Some(value) = value {
                    self.resolve_expr(value);
                }
            }
            Stmt::Break { .. } | Stmt::Continue { .. } => {}
        }
    }

    fn resolve_expr(&mut self, expr: &Expr<'_>) {
        match expr {
            Expr::Lit { .. } => {}
            Expr::Ident { symbol, span } => {
                if !self.is_defined(*symbol) {
                    self.errors.push(ResolutionError {
                        kind: ResolutionErrorKind::UndefinedIdentifier {
                            name: self.symbol_name(*symbol),
                        },
                        span: *span,
                    });
                }
            }
            Expr::Path { segments, span } => {
                if !is_builtin_path(segments, self.symbols) {
                    let first = segments.first().copied();
                    if let Some(sym) = first {
                        if !self.is_defined(sym) {
                            self.errors.push(ResolutionError {
                                kind: ResolutionErrorKind::UndefinedIdentifier {
                                    name: self.symbol_name(sym),
                                },
                                span: *span,
                            });
                        }
                    }
                }
            }
            Expr::Intrinsic { .. } => {}
            Expr::BinOp { lhs, rhs, .. } => {
                self.resolve_expr(lhs);
                self.resolve_expr(rhs);
            }
            Expr::UnOp { expr, .. } => self.resolve_expr(expr),
            Expr::Call { func, args, .. } => {
                self.resolve_expr(func);
                for arg in *args {
                    self.resolve_expr(arg);
                }
            }
            Expr::Index { base, index, .. } => {
                self.resolve_expr(base);
                self.resolve_expr(index);
            }
            Expr::Field { base, .. } => self.resolve_expr(base),
            Expr::Cast { expr, .. } => self.resolve_expr(expr),
            Expr::Range { lo, hi, .. } => {
                if let Some(lo) = lo {
                    self.resolve_expr(lo);
                }
                if let Some(hi) = hi {
                    self.resolve_expr(hi);
                }
            }
        }
    }

    fn define_symbol(&mut self, symbol: Symbol, span: SourceSpan) {
        if self.exists_in_active_chain(symbol) {
            self.errors.push(ResolutionError {
                kind: ResolutionErrorKind::Redefinition {
                    name: self.symbol_name(symbol),
                },
                span,
            });
            return;
        }

        let current = *self
            .active_scopes
            .last()
            .expect("active scope stack must be non-empty");
        self.scope_tree.scopes[current].bindings.insert(symbol, span);
    }

    fn exists_in_active_chain(&self, symbol: Symbol) -> bool {
        let mut scope = self.active_scopes.last().copied();
        while let Some(id) = scope {
            if self.scope_tree.scopes[id].bindings.contains_key(&symbol) {
                return true;
            }
            scope = self.scope_tree.scopes[id].parent;
        }
        false
    }

    fn is_defined(&self, symbol: Symbol) -> bool {
        let mut scope = self.active_scopes.last().copied();
        while let Some(id) = scope {
            if self.scope_tree.scopes[id].bindings.contains_key(&symbol) {
                return true;
            }
            scope = self.scope_tree.scopes[id].parent;
        }
        false
    }

    fn push_scope(&mut self) -> usize {
        let parent = self.active_scopes.last().copied();
        let id = self.scope_tree.scopes.len();
        self.scope_tree.scopes.push(ScopeNode {
            parent,
            children: Vec::new(),
            bindings: HashMap::new(),
        });
        if let Some(parent) = parent {
            self.scope_tree.scopes[parent].children.push(id);
        }
        self.active_scopes.push(id);
        id
    }

    fn pop_scope(&mut self, expected: usize) {
        let popped = self
            .active_scopes
            .pop()
            .expect("active scope stack must be non-empty");
        debug_assert_eq!(popped, expected);
    }

    fn symbol_name(&self, symbol: Symbol) -> String {
        self.symbols
            .resolve(symbol)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("<sym:{}>", symbol.0))
    }
}

fn is_builtin_path(segments: &[Symbol], symbols: &SymbolTable) -> bool {
    if segments.len() != 2 {
        return false;
    }

    let a = symbols.resolve(segments[0]);
    let b = symbols.resolve(segments[1]);

    matches!(
        (a, b),
        (Some("thread"), Some("x" | "y" | "z"))
            | (Some("block"), Some("x" | "y" | "z"))
            | (Some("warp"), Some("lane_id"))
            | (
                Some("forge"),
                Some("sync_block" | "sync_warp" | "atomic_add" | "exp" | "sqrt" | "fma")
            )
    )
}

#[cfg(test)]
mod tests {
    use bumpalo::Bump;

    use crate::lexer::Lexer;
    use crate::parser::ParseSession;

    use super::*;

    #[test]
    fn reports_undefined_identifier_with_correct_span() {
        let arena = Bump::new();
        let src = r#"
@kernel
fn k(a: u32) {
    let x = y;
}
"#;

        let (tokens, lex_errors) = Lexer::new(src).lex_all();
        assert!(lex_errors.is_empty(), "lexer errors: {lex_errors:#?}");

        let tokens = arena.alloc_slice_fill_iter(tokens);
        let mut parser = ParseSession::new(&arena, tokens);
        let module = parser.parse_module();
        assert!(parser.errors.is_empty(), "parse errors: {:#?}", parser.errors);

        let (_scopes, errors) = resolve_module(&module, &parser.symbols);
        assert_eq!(errors.len(), 1, "unexpected resolution errors: {errors:#?}");

        let err = &errors[0];
        assert!(matches!(
            err.kind,
            ResolutionErrorKind::UndefinedIdentifier { .. }
        ));
        assert_eq!(err.span.slice(src), "y");
    }

    #[test]
    fn reports_redefinition_when_shadowing_is_attempted() {
        let arena = Bump::new();
        let src = r#"
@kernel
fn k(a: u32) {
    let x = a;
    if x > 0 {
        let x = 1;
    }
}
"#;

        let (tokens, lex_errors) = Lexer::new(src).lex_all();
        assert!(lex_errors.is_empty(), "lexer errors: {lex_errors:#?}");

        let tokens = arena.alloc_slice_fill_iter(tokens);
        let mut parser = ParseSession::new(&arena, tokens);
        let module = parser.parse_module();
        assert!(parser.errors.is_empty(), "parse errors: {:#?}", parser.errors);

        let (_scopes, errors) = resolve_module(&module, &parser.symbols);
        assert_eq!(errors.len(), 1, "unexpected errors: {errors:#?}");
        assert!(matches!(
            errors[0].kind,
            ResolutionErrorKind::Redefinition { .. }
        ));
    }
}
