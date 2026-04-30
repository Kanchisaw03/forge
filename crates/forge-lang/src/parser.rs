use bumpalo::Bump;

use crate::ast::{
    Attr, BinOp, Block, Expr, FnDecl, IntrinsicKind, KernelDef, Lit, Module, Param, Stmt, Symbol,
    SymbolTable, Type, UnOp,
};
use crate::error::ParseError;
use crate::span::SourceSpan;
use crate::token::{NumericBase, Token, TokenKind};

/// Arena-backed parser session.
///
/// Invariants:
/// - `tokens` belong to the same source unit.
/// - `arena` outlives all AST references produced by this session.
pub struct ParseSession<'arena> {
    /// Allocation arena for AST nodes and slices.
    pub arena: &'arena Bump,
    /// Token stream to parse.
    pub tokens: &'arena [Token],
    /// Current token index.
    pub pos: usize,
    /// Collected parse errors.
    pub errors: Vec<ParseError>,
    /// Symbol interner used by parser and resolver.
    pub symbols: SymbolTable,
}

impl<'arena> ParseSession<'arena> {
    /// Creates a parser session.
    ///
    /// Errors:
    /// - This function does not fail.
    pub fn new(arena: &'arena Bump, tokens: &'arena [Token]) -> Self {
        Self {
            arena,
            tokens,
            pos: 0,
            errors: Vec::new(),
            symbols: SymbolTable::new(),
        }
    }

    /// Parses a full module, recovering from statement and declaration errors.
    ///
    /// Errors:
    /// - Individual parse issues are accumulated in `self.errors`.
    pub fn parse_module(&mut self) -> Module<'arena> {
        let mut kernels = Vec::new();
        let mut functions = Vec::new();

        while !self.at_eof() {
            let attrs = self.parse_attrs();

            if self.at(|k| matches!(k, TokenKind::Fn)) {
                let is_kernel = attrs.iter().any(|a| matches!(a, Attr::Kernel));
                if is_kernel {
                    if let Some(kernel) = self.parse_kernel(attrs) {
                        kernels.push(kernel);
                    }
                } else if let Some(decl) = self.parse_fn_decl() {
                    functions.push(decl);
                }
                continue;
            }

            if self.at_eof() {
                break;
            }

            self.push_error_unexpected("top-level `fn` or `@kernel fn`", self.current_span());
            self.synchronize_declaration();
        }

        Module { kernels, functions }
    }

    fn parse_kernel(&mut self, attrs: Vec<Attr>) -> Option<KernelDef<'arena>> {
        let start = self.current_span();
        self.bump(); // fn

        let (name, _) = self.expect_ident_symbol("kernel name")?;
        self.expect_simple(TokenKind::LParen, "`(` after kernel name")?;

        let mut params = Vec::new();
        if !self.at(|k| matches!(k, TokenKind::RParen)) {
            loop {
                let param_attrs = self.parse_attrs();
                let (param_name, param_span) = self.expect_ident_symbol("parameter name")?;
                self.expect_simple(TokenKind::Colon, "`:` after parameter name")?;
                let param_ty = self.parse_type()?;
                let span = combine_span(param_span, self.prev_span());
                params.push(Param {
                    name: param_name,
                    ty: param_ty,
                    attrs: param_attrs,
                    span,
                });

                if self.eat(|k| matches!(k, TokenKind::Comma)).is_some() {
                    if self.at(|k| matches!(k, TokenKind::RParen)) {
                        break;
                    }
                    continue;
                }
                break;
            }
        }

        self.expect_simple(TokenKind::RParen, "`)` after parameter list")?;

        let ret_ty = if self.eat(|k| matches!(k, TokenKind::Arrow)).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = self.parse_block()?;
        let span = combine_span(start, body.span);

        Some(KernelDef {
            attrs,
            name,
            params,
            ret_ty,
            body,
            span,
        })
    }

    /// Parses a plain helper function declaration (no `@kernel` attribute).
    fn parse_fn_decl(&mut self) -> Option<FnDecl<'arena>> {
        let start = self.current_span();
        self.bump(); // fn
        let (name, _) = self.expect_ident_symbol("function name")?;
        self.expect_simple(TokenKind::LParen, "`(` after function name")?;
        let mut params = Vec::new();
        if !self.at(|k| matches!(k, TokenKind::RParen)) {
            loop {
                let param_attrs = self.parse_attrs();
                let (param_name, param_span) = self.expect_ident_symbol("parameter name")?;
                self.expect_simple(TokenKind::Colon, "`:` after parameter name")?;
                let param_ty = self.parse_type()?;
                let span = combine_span(param_span, self.prev_span());
                params.push(Param { name: param_name, ty: param_ty, attrs: param_attrs, span });
                if self.eat(|k| matches!(k, TokenKind::Comma)).is_some() {
                    if self.at(|k| matches!(k, TokenKind::RParen)) { break; }
                    continue;
                }
                break;
            }
        }
        self.expect_simple(TokenKind::RParen, "`)` after parameter list")?;
        let ret_ty = if self.eat(|k| matches!(k, TokenKind::Arrow)).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };
        let body = self.parse_block()?;
        let span = combine_span(start, body.span);
        Some(FnDecl { name, params, ret_ty, body, span })
    }

    fn parse_attrs(&mut self) -> Vec<Attr> {
        let mut attrs = Vec::new();

        while self.eat(|k| matches!(k, TokenKind::At)).is_some() {
            let Some((sym, span)) = self.expect_ident_symbol("attribute name after `@`") else {
                self.synchronize_statement();
                break;
            };

            let attr = match self.symbols.resolve(sym) {
                Some("kernel") => Attr::Kernel,
                Some("shared") => Attr::Shared,
                Some("global") => Attr::Global,
                Some("register") => Attr::Register,
                _ => Attr::Unknown(sym),
            };

            attrs.push(attr);

            // Preserve span in diagnostics when unknown attrs are used in invalid places.
            if matches!(attr, Attr::Unknown(_)) && self.at(|k| matches!(k, TokenKind::Semicolon)) {
                self.errors.push(ParseError::Message {
                    message: "attribute must precede a declaration or let statement".to_owned(),
                    span,
                });
            }
        }

        attrs
    }

    fn parse_block(&mut self) -> Option<Block<'arena>> {
        let start = self.expect_simple(TokenKind::LBrace, "`{` to start block")?;

        let mut stmts = Vec::new();
        while !self.at(|k| matches!(k, TokenKind::RBrace | TokenKind::Eof))
            && !self.at_declaration_sync_boundary()
        {
            let before = self.pos;
            match self.parse_stmt() {
                Some(stmt) => stmts.push(stmt),
                None => {
                    self.synchronize_statement();
                    // Recovery invariant: always advance at least one token when
                    // parse_stmt failed and synchronize did not consume anything.
                    if self.pos == before && !self.at_eof() {
                        let _ = self.bump();
                    }
                }
            }
        }

        let end = if self.at(|k| matches!(k, TokenKind::RBrace)) {
            self.expect_simple(TokenKind::RBrace, "`}` to end block")?
        } else {
            // Recovery rationale: when we encounter a declaration boundary or EOF
            // without a closing brace, synthesize the block end so outer parsing
            // can continue with the next kernel.
            self.errors.push(ParseError::Message {
                message: "missing `}` to end block; inserted synthetic brace for recovery"
                    .to_owned(),
                span: self.current_span(),
            });
            self.prev_span()
        };
        let stmts_slice = self.arena.alloc_slice_fill_iter(stmts);

        Some(Block {
            stmts: stmts_slice,
            span: combine_span(start, end),
        })
    }

    fn parse_stmt(&mut self) -> Option<Stmt<'arena>> {
        let attrs = self.parse_attrs();

        if self.at(|k| matches!(k, TokenKind::Let)) {
            return self.parse_let_stmt(attrs);
        }

        if !attrs.is_empty() {
            self.errors.push(ParseError::Message {
                message: "attributes are only valid on `let` statements and declarations"
                    .to_owned(),
                span: self.current_span(),
            });
        }

        if self.at(|k| matches!(k, TokenKind::If)) {
            return self.parse_if_stmt();
        }
        if self.at(|k| matches!(k, TokenKind::For)) {
            return self.parse_for_stmt();
        }
        if self.at(|k| matches!(k, TokenKind::While)) {
            return self.parse_while_stmt();
        }
        if self.at(|k| matches!(k, TokenKind::Loop)) {
            return self.parse_loop_stmt();
        }
        if self.at(|k| matches!(k, TokenKind::Return)) {
            return self.parse_return_stmt();
        }
        if self.at(|k| matches!(k, TokenKind::Break)) {
            let start = self.bump()?.span;
            let end = self.expect_simple(TokenKind::Semicolon, "`;` after break")?;
            return Some(Stmt::Break {
                span: combine_span(start, end),
            });
        }
        if self.at(|k| matches!(k, TokenKind::Continue)) {
            let start = self.bump()?.span;
            let end = self.expect_simple(TokenKind::Semicolon, "`;` after continue")?;
            return Some(Stmt::Continue {
                span: combine_span(start, end),
            });
        }

        self.parse_expr_or_assign_stmt()
    }

    fn parse_let_stmt(&mut self, attrs: Vec<Attr>) -> Option<Stmt<'arena>> {
        let start = self.bump()?.span; // let

        if self.eat(|k| matches!(k, TokenKind::Mut)).is_some() {
            // Mutation is represented in type/assignment semantics; the AST keeps
            // the same let node form and relies on later passes for mutability checks.
        }

        let (name, _) = self.expect_ident_symbol("binding name")?;

        let ty = if self.eat(|k| matches!(k, TokenKind::Colon)).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };

        let init = if self.eat(|k| matches!(k, TokenKind::Eq)).is_some() {
            Some(self.parse_expr(0)?)
        } else {
            None
        };

        let end = self.expect_simple(TokenKind::Semicolon, "`;` after let statement")?;

        Some(Stmt::Let {
            name,
            ty,
            init,
            attrs,
            span: combine_span(start, end),
        })
    }

    fn parse_if_stmt(&mut self) -> Option<Stmt<'arena>> {
        let start = self.bump()?.span; // if
        let cond = self.parse_expr(0)?;
        let then = self.parse_block()?;

        let else_ = if self.eat(|k| matches!(k, TokenKind::Else)).is_some() {
            if self.at(|k| matches!(k, TokenKind::If)) {
                let nested = self.parse_if_stmt()?;
                let nested_span = nested.span();
                let nested_slice = self.arena.alloc_slice_fill_iter([nested]);
                Some(Block {
                    stmts: nested_slice,
                    span: nested_span,
                })
            } else {
                Some(self.parse_block()?)
            }
        } else {
            None
        };

        let end = else_.as_ref().map_or(then.span, |b| b.span);
        Some(Stmt::If {
            cond,
            then,
            else_,
            span: combine_span(start, end),
        })
    }

    fn parse_for_stmt(&mut self) -> Option<Stmt<'arena>> {
        let start = self.bump()?.span; // for
        let (var, _) = self.expect_ident_symbol("loop variable")?;
        self.expect_simple(TokenKind::In, "`in` in for statement")?;
        let iter = self.parse_expr(0)?;
        let body = self.parse_block()?;
        let span = combine_span(start, body.span);

        Some(Stmt::For {
            var,
            iter,
            body,
            span,
        })
    }

    fn parse_while_stmt(&mut self) -> Option<Stmt<'arena>> {
        let start = self.bump()?.span; // while
        let cond = self.parse_expr(0)?;
        let body = self.parse_block()?;
        let span = combine_span(start, body.span);

        Some(Stmt::While { cond, body, span })
    }

    fn parse_loop_stmt(&mut self) -> Option<Stmt<'arena>> {
        let start = self.bump()?.span; // loop
        let body = self.parse_block()?;
        let span = combine_span(start, body.span);
        Some(Stmt::Loop { body, span })
    }

    fn parse_return_stmt(&mut self) -> Option<Stmt<'arena>> {
        let start = self.bump()?.span; // return
        let value = if self.at(|k| matches!(k, TokenKind::Semicolon)) {
            None
        } else {
            Some(self.parse_expr(0)?)
        };
        let end = self.expect_simple(TokenKind::Semicolon, "`;` after return")?;
        Some(Stmt::Return {
            value,
            span: combine_span(start, end),
        })
    }

    fn parse_expr_or_assign_stmt(&mut self) -> Option<Stmt<'arena>> {
        let lhs = self.parse_expr(0)?;

        if let Some((op, _op_span)) = self.try_parse_assignment_operator() {
            let value = self.parse_expr(0)?;
            let end = self.expect_simple(TokenKind::Semicolon, "`;` after assignment")?;
            let span = combine_span(lhs.span(), end);
            return Some(Stmt::Assign {
                target: lhs,
                op,
                value,
                span,
            });
        }

        let end = self.expect_simple(TokenKind::Semicolon, "`;` after expression")?;
        let span = combine_span(lhs.span(), end);
        Some(Stmt::Expr(lhs, span))
    }

    fn try_parse_assignment_operator(&mut self) -> Option<(Option<BinOp>, SourceSpan)> {
        let token = self.current()?.clone();

        let mapped = match token.node {
            TokenKind::Eq => Some(None),
            TokenKind::PlusEq => Some(Some(BinOp::Add)),
            TokenKind::MinusEq => Some(Some(BinOp::Sub)),
            TokenKind::StarEq => Some(Some(BinOp::Mul)),
            TokenKind::SlashEq => Some(Some(BinOp::Div)),
            TokenKind::PercentEq => Some(Some(BinOp::Rem)),
            TokenKind::AmpEq => Some(Some(BinOp::BitAnd)),
            TokenKind::PipeEq => Some(Some(BinOp::BitOr)),
            TokenKind::CaretEq => Some(Some(BinOp::BitXor)),
            TokenKind::ShlEq => Some(Some(BinOp::Shl)),
            TokenKind::ShrEq => Some(Some(BinOp::Shr)),
            _ => None,
        }?;

        self.bump();
        Some((mapped, token.span))
    }

    fn parse_expr(&mut self, min_bp: u8) -> Option<&'arena Expr<'arena>> {
        let mut lhs = self.parse_prefix()?;

        loop {
            // Highest-precedence suffix operators.
            if self.at(|k| matches!(k, TokenKind::LParen)) {
                lhs = self.parse_call_suffix(lhs)?;
                continue;
            }
            if self.at(|k| matches!(k, TokenKind::LBracket)) {
                lhs = self.parse_index_suffix(lhs)?;
                continue;
            }
            if self.at(|k| matches!(k, TokenKind::Dot)) {
                lhs = self.parse_field_suffix(lhs)?;
                continue;
            }

            let Some((lbp, rbp, op_kind)) = self.infix_binding_power() else {
                break;
            };

            if lbp < min_bp {
                break;
            }

            match op_kind {
                InfixKind::Binary(op) => {
                    let op_span = self.bump()?.span;
                    let rhs = self.parse_expr(rbp)?;

                    if is_comparison(op)
                        && matches!(lhs, Expr::BinOp { op: prev, .. } if is_comparison(*prev))
                    {
                        self.errors.push(ParseError::Message {
                            message:
                                "comparison operators are non-associative; use parentheses"
                                    .to_owned(),
                            span: op_span,
                        });
                    }

                    let span = combine_span(lhs.span(), rhs.span());
                    lhs = self.alloc_expr(Expr::BinOp {
                        op,
                        lhs,
                        rhs,
                        span,
                    });
                }
                InfixKind::Range { inclusive } => {
                    self.bump();
                    let rhs = self.parse_expr(rbp)?;
                    let span = combine_span(lhs.span(), rhs.span());
                    lhs = self.alloc_expr(Expr::Range {
                        lo: Some(lhs),
                        hi: Some(rhs),
                        inclusive,
                        span,
                    });
                }
                InfixKind::Cast => {
                    self.bump();
                    let ty = self.parse_type()?;
                    let span = combine_span(lhs.span(), self.prev_span());
                    lhs = self.alloc_expr(Expr::Cast {
                        expr: lhs,
                        ty,
                        span,
                    });
                }
            }
        }

        Some(lhs)
    }

    fn parse_prefix(&mut self) -> Option<&'arena Expr<'arena>> {
        let token = self.current()?.clone();

        match token.node {
            TokenKind::IntLiteral { text, base, .. } => {
                self.bump();
                let value = parse_int_literal_value(&text, base).unwrap_or_else(|| {
                    self.errors.push(ParseError::Message {
                        message: format!("failed to parse integer literal `{text}`"),
                        span: token.span,
                    });
                    0
                });
                Some(self.alloc_expr(Expr::Lit {
                    lit: Lit::Int(value),
                    span: token.span,
                }))
            }
            TokenKind::FloatLiteral { text, .. } => {
                self.bump();
                let value = parse_float_literal_value(&text).unwrap_or_else(|| {
                    self.errors.push(ParseError::Message {
                        message: format!("failed to parse float literal `{text}`"),
                        span: token.span,
                    });
                    0.0
                });
                Some(self.alloc_expr(Expr::Lit {
                    lit: Lit::Float(value),
                    span: token.span,
                }))
            }
            TokenKind::True => {
                self.bump();
                Some(self.alloc_expr(Expr::Lit {
                    lit: Lit::Bool(true),
                    span: token.span,
                }))
            }
            TokenKind::False => {
                self.bump();
                Some(self.alloc_expr(Expr::Lit {
                    lit: Lit::Bool(false),
                    span: token.span,
                }))
            }
            TokenKind::StringLiteral { value, .. } => {
                self.bump();
                let sym = self.symbols.intern(&value);
                Some(self.alloc_expr(Expr::Lit {
                    lit: Lit::String(sym),
                    span: token.span,
                }))
            }
            TokenKind::Ident(_) => self.parse_ident_or_path_expr(),
            TokenKind::LParen => {
                self.bump();
                let expr = self.parse_expr(0)?;
                self.expect_simple(TokenKind::RParen, "`)` to close expression")?;
                Some(expr)
            }
            TokenKind::Bang => {
                let start = self.bump()?.span;
                let expr = self.parse_expr(PREFIX_BP)?;
                let span = combine_span(start, expr.span());
                Some(self.alloc_expr(Expr::UnOp {
                    op: UnOp::LogicalNot,
                    expr,
                    span,
                }))
            }
            TokenKind::Minus => {
                let start = self.bump()?.span;
                let expr = self.parse_expr(PREFIX_BP)?;
                let span = combine_span(start, expr.span());
                Some(self.alloc_expr(Expr::UnOp {
                    op: UnOp::Neg,
                    expr,
                    span,
                }))
            }
            TokenKind::Amp => {
                let start = self.bump()?.span;
                let op = if self.eat(|k| matches!(k, TokenKind::Mut)).is_some() {
                    UnOp::RefMut
                } else {
                    UnOp::Ref
                };
                let expr = self.parse_expr(PREFIX_BP)?;
                let span = combine_span(start, expr.span());
                Some(self.alloc_expr(Expr::UnOp { op, expr, span }))
            }
            TokenKind::Star => {
                let start = self.bump()?.span;
                let expr = self.parse_expr(PREFIX_BP)?;
                let span = combine_span(start, expr.span());
                Some(self.alloc_expr(Expr::UnOp {
                    op: UnOp::Deref,
                    expr,
                    span,
                }))
            }
            TokenKind::Tilde => {
                let start = self.bump()?.span;
                let expr = self.parse_expr(PREFIX_BP)?;
                let span = combine_span(start, expr.span());
                Some(self.alloc_expr(Expr::UnOp {
                    op: UnOp::BitNot,
                    expr,
                    span,
                }))
            }
            TokenKind::DotDot | TokenKind::DotDotEq => {
                let inclusive = matches!(token.node, TokenKind::DotDotEq);
                let start = self.bump()?.span;
                let hi = self.parse_expr(RANGE_RBP)?;
                let span = combine_span(start, hi.span());
                Some(self.alloc_expr(Expr::Range {
                    lo: None,
                    hi: Some(hi),
                    inclusive,
                    span,
                }))
            }
            _ => {
                self.push_error_unexpected("expression", token.span);
                None
            }
        }
    }

    fn parse_ident_or_path_expr(&mut self) -> Option<&'arena Expr<'arena>> {
        let (first_sym, first_span, first_text) = self.expect_ident_symbol_with_text("identifier")?;
        let mut segments_sym = vec![first_sym];
        let mut segments_text = vec![first_text];
        let mut end_span = first_span;

        while self.eat(|k| matches!(k, TokenKind::ColonColon)).is_some() {
            let (seg_sym, seg_span, seg_text) =
                self.expect_ident_symbol_with_text("path segment after `::`")?;
            segments_sym.push(seg_sym);
            segments_text.push(seg_text);
            end_span = seg_span;
        }

        let span = combine_span(first_span, end_span);

        if let Some(kind) = intrinsic_from_path(&segments_text) {
            return Some(self.alloc_expr(Expr::Intrinsic { kind, span }));
        }

        if segments_sym.len() == 1 {
            return Some(self.alloc_expr(Expr::Ident {
                symbol: segments_sym[0],
                span,
            }));
        }

        let slice = self.arena.alloc_slice_copy(&segments_sym);
        Some(self.alloc_expr(Expr::Path {
            segments: slice,
            span,
        }))
    }

    fn parse_call_suffix(&mut self, func: &'arena Expr<'arena>) -> Option<&'arena Expr<'arena>> {
        let start = func.span();
        self.expect_simple(TokenKind::LParen, "`(` for call")?;

        let mut args = Vec::new();
        if !self.at(|k| matches!(k, TokenKind::RParen)) {
            loop {
                let arg = self.parse_expr(0)?;
                args.push(arg);

                if self.eat(|k| matches!(k, TokenKind::Comma)).is_some() {
                    if self.at(|k| matches!(k, TokenKind::RParen)) {
                        break;
                    }
                    continue;
                }
                break;
            }
        }

        let end = self.expect_simple(TokenKind::RParen, "`)` after call arguments")?;
        let arg_slice = self.arena.alloc_slice_copy(&args);
        let span = combine_span(start, end);

        Some(self.alloc_expr(Expr::Call {
            func,
            args: arg_slice,
            span,
        }))
    }

    fn parse_index_suffix(
        &mut self,
        base: &'arena Expr<'arena>,
    ) -> Option<&'arena Expr<'arena>> {
        let start = base.span();
        self.expect_simple(TokenKind::LBracket, "`[` for indexing")?;
        let index = self.parse_expr(0)?;
        let end = self.expect_simple(TokenKind::RBracket, "`]` after index")?;
        let span = combine_span(start, end);

        Some(self.alloc_expr(Expr::Index { base, index, span }))
    }

    fn parse_field_suffix(
        &mut self,
        base: &'arena Expr<'arena>,
    ) -> Option<&'arena Expr<'arena>> {
        let start = base.span();
        self.expect_simple(TokenKind::Dot, "`.` for field access")?;
        let (field, end) = self.expect_ident_symbol("field name after `.`")?;
        let span = combine_span(start, end);

        Some(self.alloc_expr(Expr::Field { base, field, span }))
    }

    fn infix_binding_power(&self) -> Option<(u8, u8, InfixKind)> {
        let kind = &self.current()?.node;

        let mapped = match kind {
            TokenKind::DotDot => (RANGE_LBP, RANGE_RBP, InfixKind::Range { inclusive: false }),
            TokenKind::DotDotEq => (RANGE_LBP, RANGE_RBP, InfixKind::Range { inclusive: true }),
            TokenKind::OrOr => (
                LOGICAL_OR_LBP,
                LOGICAL_OR_RBP,
                InfixKind::Binary(BinOp::LogicalOr),
            ),
            TokenKind::AndAnd => (
                LOGICAL_AND_LBP,
                LOGICAL_AND_RBP,
                InfixKind::Binary(BinOp::LogicalAnd),
            ),
            TokenKind::EqEq => (CMP_LBP, CMP_RBP, InfixKind::Binary(BinOp::Eq)),
            TokenKind::NotEq => (CMP_LBP, CMP_RBP, InfixKind::Binary(BinOp::Ne)),
            TokenKind::Less => (CMP_LBP, CMP_RBP, InfixKind::Binary(BinOp::Lt)),
            TokenKind::Greater => (CMP_LBP, CMP_RBP, InfixKind::Binary(BinOp::Gt)),
            TokenKind::LessEq => (CMP_LBP, CMP_RBP, InfixKind::Binary(BinOp::Le)),
            TokenKind::GreaterEq => (CMP_LBP, CMP_RBP, InfixKind::Binary(BinOp::Ge)),
            TokenKind::Pipe => (BIT_OR_LBP, BIT_OR_RBP, InfixKind::Binary(BinOp::BitOr)),
            TokenKind::Caret => (
                BIT_XOR_LBP,
                BIT_XOR_RBP,
                InfixKind::Binary(BinOp::BitXor),
            ),
            TokenKind::Amp => (BIT_AND_LBP, BIT_AND_RBP, InfixKind::Binary(BinOp::BitAnd)),
            TokenKind::Shl => (SHIFT_LBP, SHIFT_RBP, InfixKind::Binary(BinOp::Shl)),
            TokenKind::Shr => (SHIFT_LBP, SHIFT_RBP, InfixKind::Binary(BinOp::Shr)),
            TokenKind::Plus => (ADD_LBP, ADD_RBP, InfixKind::Binary(BinOp::Add)),
            TokenKind::Minus => (ADD_LBP, ADD_RBP, InfixKind::Binary(BinOp::Sub)),
            TokenKind::Star => (MUL_LBP, MUL_RBP, InfixKind::Binary(BinOp::Mul)),
            TokenKind::Slash => (MUL_LBP, MUL_RBP, InfixKind::Binary(BinOp::Div)),
            TokenKind::Percent => (MUL_LBP, MUL_RBP, InfixKind::Binary(BinOp::Rem)),
            TokenKind::As => (CAST_LBP, CAST_RBP, InfixKind::Cast),
            _ => return None,
        };

        Some(mapped)
    }

    fn parse_type(&mut self) -> Option<&'arena Type<'arena>> {
        let token = self.current()?.clone();

        let ty = match token.node {
            TokenKind::F32 => {
                self.bump();
                Type::F32
            }
            TokenKind::F64 => {
                self.bump();
                Type::F64
            }
            TokenKind::I32 => {
                self.bump();
                Type::I32
            }
            TokenKind::I64 => {
                self.bump();
                Type::I64
            }
            TokenKind::U32 => {
                self.bump();
                Type::U32
            }
            TokenKind::U64 => {
                self.bump();
                Type::U64
            }
            TokenKind::Bool => {
                self.bump();
                Type::Bool
            }
            TokenKind::Usize => {
                self.bump();
                Type::Usize
            }
            TokenKind::Amp => {
                self.bump();
                let mutable = self.eat(|k| matches!(k, TokenKind::Mut)).is_some();
                self.expect_simple(TokenKind::LBracket, "`[` after reference type")?;
                let elem = self.parse_type()?;
                self.expect_simple(TokenKind::RBracket, "`]` after slice type")?;
                Type::Slice { elem, mutable }
            }
            TokenKind::Star => {
                self.bump();
                let mutable = if self.eat(|k| matches!(k, TokenKind::Mut)).is_some() {
                    true
                } else {
                    self.eat(|k| matches!(k, TokenKind::Const)).is_none()
                };
                let elem = self.parse_type()?;
                Type::Ptr { elem, mutable }
            }
            TokenKind::Ident(ref text) => {
                if let Some((base, lanes)) = parse_vector_type(text) {
                    self.bump();
                    Type::Vector {
                        elem: self.alloc_type(base.as_type()),
                        lanes,
                    }
                } else {
                    let sym = self.symbols.intern(text);
                    self.bump();
                    Type::Named(sym)
                }
            }
            _ => {
                self.push_error_unexpected("type", token.span);
                return None;
            }
        };

        Some(self.alloc_type(ty))
    }

    fn expect_ident_symbol(&mut self, expected: &str) -> Option<(Symbol, SourceSpan)> {
        let (sym, span, _) = self.expect_ident_symbol_with_text(expected)?;
        Some((sym, span))
    }

    fn expect_ident_symbol_with_text(
        &mut self,
        expected: &str,
    ) -> Option<(Symbol, SourceSpan, String)> {
        let token = self.current()?.clone();
        if let TokenKind::Ident(text) = token.node {
            self.bump();
            let sym = self.symbols.intern(&text);
            Some((sym, token.span, text))
        } else {
            self.errors.push(ParseError::UnexpectedToken {
                expected: expected.to_owned(),
                found: format!("{:?}", token.node),
                span: token.span,
            });
            None
        }
    }

    fn expect_simple(&mut self, expected: TokenKind, msg: &str) -> Option<SourceSpan> {
        let token = self.current()?.clone();
        if same_variant(&token.node, &expected) {
            self.bump();
            return Some(token.span);
        }

        self.errors.push(ParseError::UnexpectedToken {
            expected: msg.to_owned(),
            found: format!("{:?}", token.node),
            span: token.span,
        });
        None
    }

    fn push_error_unexpected(&mut self, expected: &str, span: SourceSpan) {
        let found = self
            .current()
            .map(|t| format!("{:?}", t.node))
            .unwrap_or_else(|| "<eof>".to_owned());
        self.errors.push(ParseError::UnexpectedToken {
            expected: expected.to_owned(),
            found,
            span,
        });
    }

    fn synchronize_statement(&mut self) {
        while !self.at_eof() {
            if self.eat(|k| matches!(k, TokenKind::Semicolon)).is_some() {
                return;
            }

            if self.at(|k| {
                matches!(
                    k,
                    TokenKind::RBrace
                        | TokenKind::Fn
                        | TokenKind::At
                        | TokenKind::If
                        | TokenKind::For
                        | TokenKind::While
                        | TokenKind::Loop
                        | TokenKind::Return
                        | TokenKind::Let
                )
            }) {
                return;
            }

            self.bump();
        }
    }

    fn synchronize_declaration(&mut self) {
        while !self.at_eof() {
            if self.at(|k| matches!(k, TokenKind::Fn | TokenKind::At | TokenKind::Eof)) {
                return;
            }
            self.bump();
        }
    }

    fn at_declaration_sync_boundary(&self) -> bool {
        if self.at(|k| matches!(k, TokenKind::Fn)) {
            return true;
        }

        if self.at(|k| matches!(k, TokenKind::At))
            && matches!(
                self.tokens.get(self.pos + 1).map(|t| &t.node),
                Some(TokenKind::Ident(name)) if name == "kernel"
            )
        {
            return true;
        }

        false
    }

    fn alloc_expr(&self, expr: Expr<'arena>) -> &'arena Expr<'arena> {
        self.arena.alloc(expr)
    }

    fn alloc_type(&self, ty: Type<'arena>) -> &'arena Type<'arena> {
        self.arena.alloc(ty)
    }

    fn current(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn prev_span(&self) -> SourceSpan {
        if self.pos == 0 {
            self.current_span()
        } else {
            self.tokens[self.pos - 1].span
        }
    }

    fn current_span(&self) -> SourceSpan {
        self.current().map_or(
            SourceSpan::new(0, 0, 1, 1),
            |t| SourceSpan::new(t.span.start, t.span.end, t.span.line, t.span.col),
        )
    }

    fn at(&self, pred: impl FnOnce(&TokenKind) -> bool) -> bool {
        self.current().is_some_and(|t| pred(&t.node))
    }

    fn at_eof(&self) -> bool {
        self.at(|k| matches!(k, TokenKind::Eof)) || self.pos >= self.tokens.len()
    }

    fn eat(&mut self, pred: impl FnOnce(&TokenKind) -> bool) -> Option<&Token> {
        if self.at(pred) {
            return self.bump();
        }
        None
    }

    fn bump(&mut self) -> Option<&Token> {
        let tok = self.tokens.get(self.pos)?;
        self.pos += 1;
        Some(tok)
    }
}

const RANGE_LBP: u8 = 20;
const RANGE_RBP: u8 = 21;
const LOGICAL_OR_LBP: u8 = 30;
const LOGICAL_OR_RBP: u8 = 31;
const LOGICAL_AND_LBP: u8 = 40;
const LOGICAL_AND_RBP: u8 = 41;
const CMP_LBP: u8 = 50;
const CMP_RBP: u8 = 51;
const BIT_OR_LBP: u8 = 60;
const BIT_OR_RBP: u8 = 61;
const BIT_XOR_LBP: u8 = 70;
const BIT_XOR_RBP: u8 = 71;
const BIT_AND_LBP: u8 = 80;
const BIT_AND_RBP: u8 = 81;
const SHIFT_LBP: u8 = 90;
const SHIFT_RBP: u8 = 91;
const ADD_LBP: u8 = 100;
const ADD_RBP: u8 = 101;
const MUL_LBP: u8 = 110;
const MUL_RBP: u8 = 111;
const CAST_LBP: u8 = 120;
const CAST_RBP: u8 = 121;
const PREFIX_BP: u8 = 130;

enum InfixKind {
    Binary(BinOp),
    Range { inclusive: bool },
    Cast,
}

fn same_variant(a: &TokenKind, b: &TokenKind) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

fn combine_span(start: SourceSpan, end: SourceSpan) -> SourceSpan {
    SourceSpan::new(start.start, end.end, start.line, start.col)
}

fn is_comparison(op: BinOp) -> bool {
    matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge)
}

fn parse_int_literal_value(text: &str, base: NumericBase) -> Option<u64> {
    let stripped = strip_int_suffix(text);
    let clean = stripped.replace('_', "");

    match base {
        NumericBase::Decimal => clean.parse::<u64>().ok(),
        NumericBase::Hex => {
            let digits = clean.strip_prefix("0x").or_else(|| clean.strip_prefix("0X"))?;
            u64::from_str_radix(digits, 16).ok()
        }
        NumericBase::Binary => {
            let digits = clean.strip_prefix("0b").or_else(|| clean.strip_prefix("0B"))?;
            u64::from_str_radix(digits, 2).ok()
        }
        NumericBase::Octal => {
            let digits = clean.strip_prefix("0o").or_else(|| clean.strip_prefix("0O"))?;
            u64::from_str_radix(digits, 8).ok()
        }
    }
}

fn parse_float_literal_value(text: &str) -> Option<f64> {
    let stripped = strip_float_suffix(text);
    let clean = stripped.replace('_', "");
    clean.parse::<f64>().ok()
}

fn strip_int_suffix(text: &str) -> &str {
    for suffix in ["usize", "u32", "i32", "u64", "i64"] {
        if let Some(prefix) = text.strip_suffix(suffix) {
            return prefix;
        }
    }
    text
}

fn strip_float_suffix(text: &str) -> &str {
    if let Some(prefix) = text.strip_suffix("f32") {
        return prefix;
    }
    if let Some(prefix) = text.strip_suffix("f64") {
        return prefix;
    }
    text
}

fn parse_vector_type(text: &str) -> Option<(VectorBase, u8)> {
    let (base, lanes) = text.split_once('x')?;
    let lanes = lanes.parse::<u8>().ok()?;
    let base = match base {
        "f32" => VectorBase::F32,
        "f64" => VectorBase::F64,
        "i32" => VectorBase::I32,
        "i64" => VectorBase::I64,
        "u32" => VectorBase::U32,
        "u64" => VectorBase::U64,
        _ => return None,
    };
    Some((base, lanes))
}

#[derive(Clone, Copy)]
enum VectorBase {
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
}

impl VectorBase {
    fn as_type<'arena>(self) -> Type<'arena> {
        match self {
            VectorBase::F32 => Type::F32,
            VectorBase::F64 => Type::F64,
            VectorBase::I32 => Type::I32,
            VectorBase::I64 => Type::I64,
            VectorBase::U32 => Type::U32,
            VectorBase::U64 => Type::U64,
        }
    }
}

fn intrinsic_from_path(parts: &[String]) -> Option<IntrinsicKind> {
    if parts.len() != 2 {
        return None;
    }

    let kind = match (parts[0].as_str(), parts[1].as_str()) {
        ("thread", "x") => IntrinsicKind::ThreadX,
        ("thread", "y") => IntrinsicKind::ThreadY,
        ("thread", "z") => IntrinsicKind::ThreadZ,
        ("block", "x") => IntrinsicKind::BlockX,
        ("block", "y") => IntrinsicKind::BlockY,
        ("block", "z") => IntrinsicKind::BlockZ,
        ("warp", "lane_id") => IntrinsicKind::WarpLaneId,
        ("forge", "sync_block") => IntrinsicKind::ForgeSyncBlock,
        ("forge", "sync_warp") => IntrinsicKind::ForgeSyncWarp,
        ("forge", "atomic_add") => IntrinsicKind::ForgeAtomicAdd,
        ("forge", "exp") => IntrinsicKind::ForgeExp,
        ("forge", "sqrt") => IntrinsicKind::ForgeSqrt,
        ("forge", "fma") => IntrinsicKind::ForgeFma,
        _ => return None,
    };

    Some(kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    fn parse<'a>(arena: &'a Bump, src: &str) -> (Module<'a>, ParseSession<'a>) {
        let (tokens, lex_errors) = Lexer::new(src).lex_all();
        assert!(lex_errors.is_empty(), "lexer errors: {lex_errors:#?}");
        let tokens = arena.alloc_slice_fill_iter(tokens);
        let mut session = ParseSession::new(arena, tokens);
        let module = session.parse_module();
        (module, session)
    }

    #[test]
    fn parses_matmul_kernel_shape() {
        let arena = Bump::new();
        let src = r#"
@kernel
fn matmul(a: &[f32], b: &[f32], out: &mut [f32], N: u32) {
    let row = thread::x();
    let col = thread::y();

    if row < N && col < N {
        let mut sum: f32 = 0.0;
        for k in 0u32..N {
            sum += a[row * N + k] * b[k * N + col];
        }
        out[row * N + col] = sum;
    }
}
"#;

        let (module, session) = parse(&arena, src);
        assert!(session.errors.is_empty(), "parse errors: {:#?}", session.errors);
        assert_eq!(module.kernels.len(), 1);
        let k = &module.kernels[0];
        assert_eq!(k.params.len(), 4);
        assert!(matches!(k.attrs.as_slice(), [Attr::Kernel]));
        assert!(!k.body.stmts.is_empty());
    }

    #[test]
    fn parses_softmax_and_attaches_shared_attr() {
        let arena = Bump::new();
        let src = r#"
@kernel
fn softmax(input: &[f32], output: &mut [f32], N: u32) {
    let idx = thread::x();
    if idx >= N { return; }

    @shared let max_val: f32 = input[0];
    forge::sync_block();

    let val = input[idx];
    let exp_val = forge::exp(val - max_val);
    output[idx] = exp_val;
}
"#;

        let (module, session) = parse(&arena, src);
        assert!(session.errors.is_empty(), "parse errors: {:#?}", session.errors);
        assert_eq!(module.kernels.len(), 1);
        let kernel = &module.kernels[0];

        let has_shared_let = kernel.body.stmts.iter().any(|stmt| {
            matches!(
                stmt,
                Stmt::Let {
                    attrs,
                    name: _,
                    ty: _,
                    init: _,
                    span: _
                } if attrs.iter().any(|a| matches!(a, Attr::Shared))
            )
        });
        assert!(has_shared_let);
    }

    #[test]
    fn parses_operator_precedence_correctly() {
        let arena = Bump::new();
        let src = r#"
@kernel
fn p() {
    let x = 1 + 2 * 3;
}
"#;

        let (module, session) = parse(&arena, src);
        assert!(session.errors.is_empty(), "parse errors: {:#?}", session.errors);

        let kernel = &module.kernels[0];
        let Stmt::Let { init: Some(expr), .. } = &kernel.body.stmts[0] else {
            panic!("expected let statement");
        };

        let Expr::BinOp {
            op: BinOp::Add,
            lhs: _,
            rhs,
            ..
        } = *expr
        else {
            panic!("expected top-level add expression");
        };

        assert!(matches!(
            rhs,
            Expr::BinOp {
                op: BinOp::Mul,
                ..
            }
        ));
    }

    #[test]
    fn recovers_after_missing_brace_and_parses_next_kernel() {
        let arena = Bump::new();
        let src = r#"
@kernel
fn broken(a: u32) {
    let x = a;

@kernel
fn next() {
    let y = 1;
}
"#;

        let (module, session) = parse(&arena, src);
        assert!(
            !session.errors.is_empty(),
            "expected parse errors for missing brace"
        );
        assert_eq!(module.kernels.len(), 2, "parser should recover to next kernel");
    }
}
