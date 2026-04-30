use std::collections::{HashMap, HashSet};

use thiserror::Error;

use forge_lang::ast::{
    BinOp, Block, Expr, FnDecl, IntrinsicKind, KernelDef, Lit, Module, Stmt, Symbol, SymbolTable,
    Type, UnOp,
};

use crate::func::ForgeFunction;
use crate::inst::{FloatPred, Inst, IntPred, MemOrdering, Terminator};
use crate::module::ForgeModule;
use crate::types::{BlockId, ConstValue, FuncRef, IrType, Value};

/// IR lowering diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LowerError {
    /// A variable was referenced before definition.
    #[error("undefined variable `{name}` while lowering `{function}`")]
    UndefinedVariable { function: String, name: String },

    /// A feature is not yet supported in the current lowering mode.
    #[error("unsupported construct in `{function}`: {detail}")]
    UnsupportedConstruct { function: String, detail: String },

    /// Type information could not be established for one operation.
    #[error("type inference failure in `{function}`: {detail}")]
    TypeFailure { function: String, detail: String },
}

/// Stateful SSA construction helper.
#[derive(Debug, Default, Clone)]
pub struct SsaBuilder {
    /// For each `(block, symbol)`, last visible SSA definition.
    pub current_def: HashMap<(BlockId, Symbol), Value>,
    /// Completed blocks with finalized predecessors.
    pub sealed_blocks: HashSet<BlockId>,
    /// Block parameters created before a block is sealed.
    pub incomplete_params: HashMap<(BlockId, Symbol), Value>,
    predecessors: HashMap<BlockId, Vec<BlockId>>,
}

impl SsaBuilder {
    /// Records one predecessor relation.
    pub fn add_predecessor(&mut self, block: BlockId, pred: BlockId) {
        self.predecessors.entry(block).or_default().push(pred);
    }

    /// Marks a block sealed and backfills pending block params.
    pub fn seal_block(
        &mut self,
        block: BlockId,
        func: &mut ForgeFunction,
        var_types: &HashMap<Symbol, IrType>,
    ) {
        self.sealed_blocks.insert(block);
        let pending: Vec<(Symbol, Value)> = self
            .incomplete_params
            .iter()
            .filter_map(|((b, sym), val)| (*b == block).then_some((*sym, *val)))
            .collect();
        for (symbol, param) in pending {
            let preds = self.predecessors.get(&block).cloned().unwrap_or_default();
            for pred in preds {
                let arg = self.read_variable(symbol, pred, func, var_types);
                func.append_edge_arg(pred, block, arg);
            }
            self.current_def.insert((block, symbol), param);
            self.incomplete_params.remove(&(block, symbol));
        }
    }

    /// Writes one variable definition in a block.
    pub fn write_variable(&mut self, var: Symbol, block: BlockId, value: Value) {
        self.current_def.insert((block, var), value);
    }

    /// Reads one variable in SSA form.
    pub fn read_variable(
        &mut self,
        var: Symbol,
        block: BlockId,
        func: &mut ForgeFunction,
        var_types: &HashMap<Symbol, IrType>,
    ) -> Value {
        if let Some(&val) = self.current_def.get(&(block, var)) {
            return val;
        }
        self.read_variable_recursive(var, block, func, var_types)
    }

    fn read_variable_recursive(
        &mut self,
        var: Symbol,
        block: BlockId,
        func: &mut ForgeFunction,
        var_types: &HashMap<Symbol, IrType>,
    ) -> Value {
        if !self.sealed_blocks.contains(&block) {
            let ty = var_types.get(&var).cloned().unwrap_or(IrType::U32);
            let param = func.add_block_param(block, ty);
            self.incomplete_params.insert((block, var), param);
            self.current_def.insert((block, var), param);
            return param;
        }

        let preds = self.predecessors.get(&block).cloned().unwrap_or_default();
        if preds.is_empty() {
            let ty = var_types.get(&var).cloned().unwrap_or(IrType::U32);
            let undef = func.fresh_value(ty.clone());
            func.append_inst(block, Inst::Undef { result: undef, ty });
            self.current_def.insert((block, var), undef);
            return undef;
        }
        if preds.len() == 1 {
            let val = self.read_variable(var, preds[0], func, var_types);
            self.current_def.insert((block, var), val);
            return val;
        }

        let ty = var_types.get(&var).cloned().unwrap_or(IrType::U32);
        let param = func.add_block_param(block, ty);
        self.current_def.insert((block, var), param);
        for pred in preds {
            let arg = self.read_variable(var, pred, func, var_types);
            func.append_edge_arg(pred, block, arg);
        }
        param
    }
}

/// Lowers a parsed module into Forge IR.
pub fn lower_module(
    module: &Module<'_>,
    symbols: &SymbolTable,
) -> Result<ForgeModule, Vec<LowerError>> {
    let mut out = ForgeModule::new();
    let mut errors = Vec::new();

    // Lower helper functions first so kernels can call them by name.
    for func in &module.functions {
        match lower_fn_decl(func, symbols) {
            Ok(f) => out.push_function(f),
            Err(mut err) => errors.append(&mut err),
        }
    }

    // Lower kernels.
    for kernel in &module.kernels {
        match lower_kernel(kernel, symbols) {
            Ok(func) => out.push_function(func),
            Err(mut err) => errors.append(&mut err),
        }
    }
    if errors.is_empty() {
        Ok(out)
    } else {
        Err(errors)
    }
}

fn lower_kernel(
    kernel: &KernelDef<'_>,
    symbols: &SymbolTable,
) -> Result<ForgeFunction, Vec<LowerError>> {
    let mut lowerer = FunctionLowerer::new(kernel_name(kernel, symbols), symbols);
    lowerer.lower_params(kernel);
    lowerer.lower_block(&kernel.body);
    if matches!(
        lowerer.func.block(lowerer.current_block).terminator,
        Terminator::Unreachable
    ) {
        lowerer
            .func
            .set_terminator(lowerer.current_block, Terminator::Return(None));
    }
    lowerer.ssa.seal_block(
        lowerer.current_block,
        &mut lowerer.func,
        &lowerer.var_types,
    );
    if lowerer.errors.is_empty() {
        Ok(lowerer.func)
    } else {
        Err(lowerer.errors)
    }
}

/// Lowers a helper `FnDecl` using the same machinery as `lower_kernel`.
fn lower_fn_decl(
    decl: &FnDecl<'_>,
    symbols: &SymbolTable,
) -> Result<ForgeFunction, Vec<LowerError>> {
    let name = symbols
        .resolve(decl.name)
        .unwrap_or("unknown_fn")
        .to_owned();
    let mut lowerer = FunctionLowerer::new(name, symbols);
    for param in &decl.params {
        let ty = ast_type_to_ir(param.ty);
        let value = lowerer.func.add_block_param(lowerer.func.entry, ty.clone());
        lowerer.var_types.insert(param.name, ty);
        lowerer.ssa.write_variable(param.name, lowerer.func.entry, value);
    }
    lowerer.lower_block(&decl.body);
    if matches!(
        lowerer.func.block(lowerer.current_block).terminator,
        Terminator::Unreachable
    ) {
        lowerer.func.set_terminator(lowerer.current_block, Terminator::Return(None));
    }
    lowerer.ssa.seal_block(lowerer.current_block, &mut lowerer.func, &lowerer.var_types);
    if lowerer.errors.is_empty() {
        Ok(lowerer.func)
    } else {
        Err(lowerer.errors)
    }
}

struct FunctionLowerer<'a> {
    func: ForgeFunction,
    current_block: BlockId,
    ssa: SsaBuilder,
    var_types: HashMap<Symbol, IrType>,
    errors: Vec<LowerError>,
    symbols: &'a SymbolTable,
}

impl<'a> FunctionLowerer<'a> {
    fn new(name: String, symbols: &'a SymbolTable) -> Self {
        let func = ForgeFunction::new(name);
        let entry = func.entry;
        let mut ssa = SsaBuilder::default();
        ssa.sealed_blocks.insert(entry);
        Self {
            func,
            current_block: entry,
            ssa,
            var_types: HashMap::new(),
            errors: Vec::new(),
            symbols,
        }
    }

    fn lower_params(&mut self, kernel: &KernelDef<'_>) {
        for param in &kernel.params {
            let ty = ast_type_to_ir(param.ty);
            let value = self.func.add_block_param(self.func.entry, ty.clone());
            self.var_types.insert(param.name, ty);
            self.ssa.write_variable(param.name, self.func.entry, value);
        }
    }

    fn lower_block(&mut self, block: &Block<'_>) {
        for stmt in block.stmts {
            self.lower_stmt(stmt);
        }
    }

    fn lower_stmt(&mut self, stmt: &Stmt<'_>) {
        match stmt {
            Stmt::Let {
                name,
                ty,
                init,
                span: _,
                ..
            } => {
                let value = if let Some(init) = init {
                    self.lower_expr(init)
                } else {
                    let inferred_ty = ty.map(ast_type_to_ir).unwrap_or(IrType::U32);
                    let undef = self.func.fresh_value(inferred_ty.clone());
                    self.func.append_inst(
                        self.current_block,
                        Inst::Undef {
                            result: undef,
                            ty: inferred_ty,
                        },
                    );
                    Some(undef)
                };
                if let Some(v) = value {
                    let resolved_ty = ty
                        .map(ast_type_to_ir)
                        .unwrap_or_else(|| self.func.value_type(v).cloned().unwrap_or(IrType::U32));
                    self.var_types.insert(*name, resolved_ty);
                    self.ssa.write_variable(*name, self.current_block, v);
                }
            }
            Stmt::Assign {
                target, op, value, ..
            } => {
                self.lower_assign_stmt(target, *op, value);
            }
            Stmt::Expr(expr, _) => {
                let _ = self.lower_expr(expr);
            }
            Stmt::If {
                cond,
                then,
                else_,
                ..
            } => self.lower_if_stmt(cond, then, else_.as_ref()),
            Stmt::For {
                var, iter, body, ..
            } => self.lower_for_stmt(*var, iter, body),
            Stmt::While { cond, body, .. } => self.lower_while_stmt(cond, body),
            Stmt::Loop { body, .. } => self.lower_loop_stmt(body),
            Stmt::Return { value, .. } => {
                let ret = value.and_then(|e| self.lower_expr(e));
                self.func
                    .set_terminator(self.current_block, Terminator::Return(ret));
            }
            Stmt::Break { .. } => {
                self.func
                    .set_terminator(self.current_block, Terminator::Unreachable);
            }
            Stmt::Continue { .. } => {
                self.func
                    .set_terminator(self.current_block, Terminator::Unreachable);
            }
        }
    }

    fn lower_assign_stmt(&mut self, target: &Expr<'_>, op: Option<BinOp>, value: &Expr<'_>) {
        match target {
            Expr::Ident { symbol, .. } => {
                let rhs = self.lower_expr(value);
                let Some(rhs) = rhs else {
                    return;
                };
                let final_val = if let Some(op) = op {
                    let lhs = self.read_var(*symbol);
                    match lhs {
                        Some(lhs) => self.lower_binop_values(op, lhs, rhs),
                        None => None,
                    }
                } else {
                    Some(rhs)
                };
                if let Some(v) = final_val {
                    self.ssa.write_variable(*symbol, self.current_block, v);
                }
            }
            Expr::Index { .. } => {
                let ptr = self.lower_lvalue_ptr(target);
                let Some(ptr) = ptr else {
                    return;
                };
                let rhs = self.lower_expr(value);
                let Some(rhs) = rhs else {
                    return;
                };
                let stored = if let Some(op) = op {
                    let load_ty = self
                        .func
                        .value_type(ptr)
                        .and_then(IrType::pointee)
                        .cloned()
                        .unwrap_or(IrType::F32);
                    let old = self.func.fresh_value(load_ty.clone());
                    self.func.append_inst(
                        self.current_block,
                        Inst::Load {
                            result: old,
                            ty: load_ty,
                            ptr,
                            align: 4,
                        },
                    );
                    self.lower_binop_values(op, old, rhs)
                } else {
                    Some(rhs)
                };
                if let Some(v) = stored {
                    self.func.append_inst(
                        self.current_block,
                        Inst::Store {
                            ptr,
                            value: v,
                            align: 4,
                        },
                    );
                }
            }
            _ => {
                self.errors.push(LowerError::UnsupportedConstruct {
                    function: self.func.name.clone(),
                    detail: "assignment target is not an lvalue".to_owned(),
                });
            }
        }
    }

    fn lower_if_stmt(&mut self, cond: &Expr<'_>, then: &Block<'_>, else_: Option<&Block<'_>>) {
        let Some(cond_val) = self.lower_expr(cond) else {
            return;
        };
        let then_block = self.func.create_block();
        let else_block = self.func.create_block();
        let join_block = self.func.create_block();

        let live_symbols = self.live_symbols_in_current_block();
        let mut join_params = Vec::with_capacity(live_symbols.len());
        for symbol in &live_symbols {
            if let Some(ty) = self.var_types.get(symbol).cloned() {
                let param = self.func.add_block_param(join_block, ty);
                join_params.push(param);
            }
        }

        self.func.set_terminator(
            self.current_block,
            Terminator::Branch {
                cond: cond_val,
                then_block,
                then_args: Vec::new(),
                else_block,
                else_args: Vec::new(),
            },
        );
        self.ssa.add_predecessor(then_block, self.current_block);
        self.ssa.add_predecessor(else_block, self.current_block);
        self.ssa.seal_block(then_block, &mut self.func, &self.var_types);
        self.ssa.seal_block(else_block, &mut self.func, &self.var_types);

        self.current_block = then_block;
        self.lower_block(then);
        if matches!(
            self.func.block(self.current_block).terminator,
            Terminator::Unreachable
        ) {
            let args = live_symbols
                .iter()
                .filter_map(|sym| self.read_var(*sym))
                .collect::<Vec<_>>();
            self.func.set_terminator(
                self.current_block,
                Terminator::Jump {
                    target: join_block,
                    args,
                },
            );
            self.ssa.add_predecessor(join_block, self.current_block);
        }

        self.current_block = else_block;
        if let Some(else_block_ast) = else_ {
            self.lower_block(else_block_ast);
        }
        if matches!(
            self.func.block(self.current_block).terminator,
            Terminator::Unreachable
        ) {
            let args = live_symbols
                .iter()
                .filter_map(|sym| self.read_var(*sym))
                .collect::<Vec<_>>();
            self.func.set_terminator(
                self.current_block,
                Terminator::Jump {
                    target: join_block,
                    args,
                },
            );
            self.ssa.add_predecessor(join_block, self.current_block);
        }

        self.ssa.seal_block(join_block, &mut self.func, &self.var_types);
        self.current_block = join_block;
        for (idx, sym) in live_symbols.iter().enumerate() {
            if let Some(param) = join_params.get(idx).copied() {
                self.ssa.write_variable(*sym, join_block, param);
            }
        }
    }

    fn lower_for_stmt(&mut self, var: Symbol, iter: &Expr<'_>, body: &Block<'_>) {
        let Expr::Range {
            lo,
            hi,
            inclusive,
            ..
        } = iter
        else {
            self.errors.push(LowerError::UnsupportedConstruct {
                function: self.func.name.clone(),
                detail: "for-loop iterator must be a range expression".to_owned(),
            });
            return;
        };
        let Some(lo_expr) = lo else {
            self.errors.push(LowerError::UnsupportedConstruct {
                function: self.func.name.clone(),
                detail: "open-ended for-loop ranges are unsupported".to_owned(),
            });
            return;
        };
        let Some(hi_expr) = hi else {
            self.errors.push(LowerError::UnsupportedConstruct {
                function: self.func.name.clone(),
                detail: "open-ended for-loop ranges are unsupported".to_owned(),
            });
            return;
        };
        let Some(lo_val) = self.lower_expr(lo_expr) else {
            return;
        };
        let Some(hi_val) = self.lower_expr(hi_expr) else {
            return;
        };
        let iv_ty = self
            .func
            .value_type(lo_val)
            .cloned()
            .unwrap_or(IrType::U32);
        self.var_types.insert(var, iv_ty.clone());

        let preheader = self.current_block;
        let header = self.func.create_block();
        let body_block = self.func.create_block();
        let exit_block = self.func.create_block();

        let live_symbols = self.live_symbols_in_current_block();
        let mut loop_symbols = vec![var];
        for s in live_symbols {
            if s != var {
                loop_symbols.push(s);
            }
        }

        let mut header_params = HashMap::new();
        let mut exit_params = HashMap::new();
        let mut pre_args = Vec::new();
        for sym in &loop_symbols {
            let ty = self
                .var_types
                .get(sym)
                .cloned()
                .unwrap_or(IrType::U32);
            let hp = self.func.add_block_param(header, ty.clone());
            header_params.insert(*sym, hp);
            self.ssa.write_variable(*sym, header, hp);

            let ep = self.func.add_block_param(exit_block, ty);
            exit_params.insert(*sym, ep);

            if *sym == var {
                pre_args.push(lo_val);
            } else if let Some(v) = self.read_var(*sym) {
                pre_args.push(v);
            }
        }

        self.func.set_terminator(
            preheader,
            Terminator::Jump {
                target: header,
                args: pre_args,
            },
        );
        self.ssa.add_predecessor(header, preheader);

        self.current_block = header;
        let iv = *header_params.get(&var).expect("induction param exists");
        let cond = self.func.fresh_value(IrType::Bool);
        let pred = if *inclusive {
            int_pred_for_less_equal(self.func.value_type(iv).cloned())
        } else {
            int_pred_for_less(self.func.value_type(iv).cloned())
        };
        self.func.append_inst(
            header,
            Inst::ICmp {
                result: cond,
                pred,
                lhs: iv,
                rhs: hi_val,
            },
        );
        let else_args = loop_symbols
            .iter()
            .filter_map(|sym| header_params.get(sym).copied())
            .collect::<Vec<_>>();
        self.func.set_terminator(
            header,
            Terminator::Branch {
                cond,
                then_block: body_block,
                then_args: Vec::new(),
                else_block: exit_block,
                else_args,
            },
        );
        self.ssa.add_predecessor(body_block, header);
        self.ssa.add_predecessor(exit_block, header);
        self.ssa.seal_block(body_block, &mut self.func, &self.var_types);

        self.current_block = body_block;
        self.lower_block(body);
        if matches!(
            self.func.block(self.current_block).terminator,
            Terminator::Unreachable
        ) {
            let one = self.const_like(iv, 1);
            let next = self.func.fresh_value(self.func.value_type(iv).cloned().unwrap_or(IrType::U32));
            self.func.append_inst(
                self.current_block,
                Inst::IAdd {
                    result: next,
                    lhs: iv,
                    rhs: one,
                },
            );
            self.ssa.write_variable(var, self.current_block, next);
            let back_args = loop_symbols
                .iter()
                .filter_map(|sym| self.read_var(*sym))
                .collect::<Vec<_>>();
            self.func.set_terminator(
                self.current_block,
                Terminator::Jump {
                    target: header,
                    args: back_args,
                },
            );
            self.ssa.add_predecessor(header, self.current_block);
        }

        self.ssa.seal_block(header, &mut self.func, &self.var_types);
        self.ssa.seal_block(exit_block, &mut self.func, &self.var_types);
        self.current_block = exit_block;
        for sym in loop_symbols {
            if let Some(v) = exit_params.get(&sym).copied() {
                self.ssa.write_variable(sym, exit_block, v);
            }
        }
    }

    fn lower_while_stmt(&mut self, cond: &Expr<'_>, body: &Block<'_>) {
        let header = self.func.create_block();
        let body_block = self.func.create_block();
        let exit = self.func.create_block();
        let pre = self.current_block;
        self.func.set_terminator(
            pre,
            Terminator::Jump {
                target: header,
                args: Vec::new(),
            },
        );
        self.ssa.add_predecessor(header, pre);
        self.ssa.seal_block(header, &mut self.func, &self.var_types);

        self.current_block = header;
        let Some(cond_val) = self.lower_expr(cond) else {
            return;
        };
        self.func.set_terminator(
            header,
            Terminator::Branch {
                cond: cond_val,
                then_block: body_block,
                then_args: Vec::new(),
                else_block: exit,
                else_args: Vec::new(),
            },
        );
        self.ssa.add_predecessor(body_block, header);
        self.ssa.add_predecessor(exit, header);
        self.ssa.seal_block(body_block, &mut self.func, &self.var_types);

        self.current_block = body_block;
        self.lower_block(body);
        if matches!(
            self.func.block(self.current_block).terminator,
            Terminator::Unreachable
        ) {
            self.func.set_terminator(
                self.current_block,
                Terminator::Jump {
                    target: header,
                    args: Vec::new(),
                },
            );
            self.ssa.add_predecessor(header, self.current_block);
        }
        self.ssa.seal_block(exit, &mut self.func, &self.var_types);
        self.current_block = exit;
    }

    fn lower_loop_stmt(&mut self, body: &Block<'_>) {
        let body_block = self.func.create_block();
        let pre = self.current_block;
        self.func.set_terminator(
            pre,
            Terminator::Jump {
                target: body_block,
                args: Vec::new(),
            },
        );
        self.ssa.add_predecessor(body_block, pre);
        self.ssa.seal_block(body_block, &mut self.func, &self.var_types);
        self.current_block = body_block;
        self.lower_block(body);
        if matches!(
            self.func.block(self.current_block).terminator,
            Terminator::Unreachable
        ) {
            self.func.set_terminator(
                self.current_block,
                Terminator::Jump {
                    target: body_block,
                    args: Vec::new(),
                },
            );
            self.ssa.add_predecessor(body_block, self.current_block);
        }
        let exit = self.func.create_block();
        self.current_block = exit;
    }

    fn lower_expr(&mut self, expr: &Expr<'_>) -> Option<Value> {
        match expr {
            Expr::Lit { lit, .. } => Some(self.lower_lit(lit)),
            Expr::Ident { symbol, .. } => self.read_var(*symbol),
            Expr::Path { segments, .. } => {
                if segments.len() == 1 {
                    self.read_var(segments[0])
                } else {
                    self.errors.push(LowerError::UnsupportedConstruct {
                        function: self.func.name.clone(),
                        detail: "unresolved path expression in value position".to_owned(),
                    });
                    None
                }
            }
            Expr::Intrinsic { kind, .. } => self.lower_intrinsic_value(*kind, &[]),
            Expr::BinOp { op, lhs, rhs, .. } => {
                let lhs = self.lower_expr(lhs)?;
                let rhs = self.lower_expr(rhs)?;
                self.lower_binop_values(*op, lhs, rhs)
            }
            Expr::UnOp { op, expr, .. } => self.lower_unop(*op, expr),
            Expr::Call { func, args, .. } => self.lower_call_expr(func, args),
            Expr::Index { base, index, .. } => {
                let ptr = self.lower_index_ptr(base, index)?;
                let elem_ty = self
                    .func
                    .value_type(ptr)
                    .and_then(IrType::pointee)
                    .cloned()
                    .unwrap_or(IrType::F32);
                let result = self.func.fresh_value(elem_ty.clone());
                self.func.append_inst(
                    self.current_block,
                    Inst::Load {
                        result,
                        ty: elem_ty,
                        ptr,
                        align: 4,
                    },
                );
                Some(result)
            }
            Expr::Field { .. } => {
                self.errors.push(LowerError::UnsupportedConstruct {
                    function: self.func.name.clone(),
                    detail: "field lowering is not implemented yet".to_owned(),
                });
                None
            }
            Expr::Cast { expr, ty, .. } => self.lower_cast(expr, ty),
            Expr::Range { .. } => {
                self.errors.push(LowerError::UnsupportedConstruct {
                    function: self.func.name.clone(),
                    detail: "range expression only supported in `for`".to_owned(),
                });
                None
            }
        }
    }

    fn lower_call_expr(&mut self, func_expr: &Expr<'_>, args: &[&Expr<'_>]) -> Option<Value> {
        let mut lowered_args = Vec::with_capacity(args.len());
        for arg in args {
            lowered_args.push(self.lower_expr(arg)?);
        }
        match func_expr {
            Expr::Intrinsic { kind, .. } => self.lower_intrinsic_value(*kind, &lowered_args),
            Expr::Path { segments, .. } => {
                let name = segments
                    .iter()
                    .filter_map(|s| self.symbol_name(*s))
                    .collect::<Vec<_>>()
                    .join("::");
                self.emit_call(name, lowered_args, IrType::Void)
            }
            Expr::Ident { symbol, .. } => {
                let name = self
                    .symbol_name(*symbol)
                    .unwrap_or_else(|| format!("sym_{}", symbol.0));
                self.emit_call(name, lowered_args, IrType::Void)
            }
            _ => {
                self.errors.push(LowerError::UnsupportedConstruct {
                    function: self.func.name.clone(),
                    detail: "call target must be a path or intrinsic".to_owned(),
                });
                None
            }
        }
    }

    fn lower_intrinsic_value(&mut self, kind: IntrinsicKind, args: &[Value]) -> Option<Value> {
        match kind {
            IntrinsicKind::ThreadX => {
                let result = self.func.fresh_value(IrType::U32);
                self.func
                    .append_inst(self.current_block, Inst::ThreadIdxX { result });
                Some(result)
            }
            IntrinsicKind::ThreadY => {
                let result = self.func.fresh_value(IrType::U32);
                self.func
                    .append_inst(self.current_block, Inst::ThreadIdxY { result });
                Some(result)
            }
            IntrinsicKind::ThreadZ => {
                let result = self.func.fresh_value(IrType::U32);
                self.func
                    .append_inst(self.current_block, Inst::ThreadIdxZ { result });
                Some(result)
            }
            IntrinsicKind::BlockX => {
                let result = self.func.fresh_value(IrType::U32);
                self.func
                    .append_inst(self.current_block, Inst::BlockIdxX { result });
                Some(result)
            }
            IntrinsicKind::BlockY => {
                let result = self.func.fresh_value(IrType::U32);
                self.func
                    .append_inst(self.current_block, Inst::BlockIdxY { result });
                Some(result)
            }
            IntrinsicKind::BlockZ => {
                let result = self.func.fresh_value(IrType::U32);
                self.func
                    .append_inst(self.current_block, Inst::BlockIdxZ { result });
                Some(result)
            }
            IntrinsicKind::WarpLaneId => {
                let result = self.func.fresh_value(IrType::U32);
                self.func
                    .append_inst(self.current_block, Inst::WarpLaneId { result });
                Some(result)
            }
            IntrinsicKind::ForgeSyncBlock => {
                self.func.append_inst(self.current_block, Inst::SyncBlock);
                None
            }
            IntrinsicKind::ForgeSyncWarp => {
                self.func.append_inst(self.current_block, Inst::SyncWarp);
                None
            }
            IntrinsicKind::ForgeAtomicAdd => {
                if args.len() != 2 {
                    self.errors.push(LowerError::TypeFailure {
                        function: self.func.name.clone(),
                        detail: "forge::atomic_add expects 2 arguments".to_owned(),
                    });
                    return None;
                }
                let ty = self.func.value_type(args[1]).cloned().unwrap_or(IrType::F32);
                let result = self.func.fresh_value(ty);
                self.func.append_inst(
                    self.current_block,
                    Inst::AtomicAdd {
                        result,
                        ptr: args[0],
                        value: args[1],
                        ordering: MemOrdering::SeqCst,
                    },
                );
                Some(result)
            }
            IntrinsicKind::ForgeExp => {
                // Lower to an opaque external call so codegen can emit exp_ps.
                self.emit_call("forge_std::exp_approx::exp_f32".to_owned(), args.to_vec(), IrType::F32)
            }
            IntrinsicKind::ForgeSqrt => {
                if args.len() != 1 {
                    return None;
                }
                let ty = self.func.value_type(args[0]).cloned().unwrap_or(IrType::F32);
                let result = self.func.fresh_value(ty);
                self.func.append_inst(
                    self.current_block,
                    Inst::FSqrt { result, operand: args[0] },
                );
                Some(result)
            }
            IntrinsicKind::ForgeFma => {
                // a*b+c
                if args.len() != 3 {
                    return None;
                }
                let ty = self.func.value_type(args[0]).cloned().unwrap_or(IrType::F32);
                let result = self.func.fresh_value(ty);
                self.func.append_inst(
                    self.current_block,
                    Inst::FMA { result, a: args[0], b: args[1], c: args[2] },
                );
                Some(result)
            }
        }
    }

    fn lower_unop(&mut self, op: UnOp, expr: &Expr<'_>) -> Option<Value> {
        let value = self.lower_expr(expr)?;
        let ty = self.func.value_type(value).cloned().unwrap_or(IrType::U32);
        let out = self.func.fresh_value(ty.clone());
        let inst = match op {
            UnOp::LogicalNot => Inst::Not {
                result: out,
                operand: value,
            },
            UnOp::Neg => {
                if ty.is_float() {
                    Inst::FNeg {
                        result: out,
                        operand: value,
                    }
                } else {
                    Inst::INeg {
                        result: out,
                        operand: value,
                    }
                }
            }
            UnOp::BitNot => Inst::Not {
                result: out,
                operand: value,
            },
            UnOp::Ref | UnOp::RefMut => return Some(value),
            UnOp::Deref => {
                let elem_ty = ty.pointee().cloned().unwrap_or(IrType::F32);
                let load = self.func.fresh_value(elem_ty.clone());
                self.func.append_inst(
                    self.current_block,
                    Inst::Load {
                        result: load,
                        ty: elem_ty,
                        ptr: value,
                        align: 4,
                    },
                );
                return Some(load);
            }
        };
        self.func.append_inst(self.current_block, inst);
        Some(out)
    }

    fn lower_cast(&mut self, expr: &Expr<'_>, ty: &Type<'_>) -> Option<Value> {
        let from = self.lower_expr(expr)?;
        let from_ty = self.func.value_type(from).cloned().unwrap_or(IrType::U32);
        let to_ty = ast_type_to_ir(ty);
        if from_ty == to_ty {
            return Some(from);
        }
        let to_signed = matches!(to_ty, IrType::I32 | IrType::I64);
        let result = self.func.fresh_value(to_ty.clone());
        let inst = match (from_ty.is_integer(), to_ty.is_integer(), from_ty.is_float(), to_ty.is_float()) {
            (true, true, _, _) => Inst::Trunc {
                result,
                operand: from,
                ty: to_ty.clone(),
            },
            (true, _, _, true) => Inst::IToFP {
                result,
                operand: from,
                ty: to_ty.clone(),
                signed: matches!(from_ty, IrType::I32 | IrType::I64),
            },
            (_, true, true, _) => Inst::FPToI {
                result,
                operand: from,
                ty: to_ty.clone(),
                signed: to_signed,
            },
            (_, _, true, true) => Inst::FPTrunc {
                result,
                operand: from,
                ty: to_ty.clone(),
            },
            _ => Inst::Bitcast {
                result,
                operand: from,
                ty: to_ty,
            },
        };
        self.func.append_inst(self.current_block, inst);
        Some(result)
    }

    fn lower_binop_values(&mut self, op: BinOp, lhs: Value, rhs: Value) -> Option<Value> {
        let lhs_ty = self.func.value_type(lhs).cloned().unwrap_or(IrType::U32);
        let rhs_ty = self.func.value_type(rhs).cloned().unwrap_or(IrType::U32);
        let same_ty = if lhs_ty == rhs_ty { lhs_ty.clone() } else { IrType::U32 };
        let result_ty = if matches!(
            op,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge | BinOp::LogicalAnd | BinOp::LogicalOr
        ) {
            IrType::Bool
        } else {
            same_ty.clone()
        };
        let result = self.func.fresh_value(result_ty);
        let inst = if same_ty.is_float() {
            match op {
                BinOp::Add => Inst::FAdd { result, lhs, rhs },
                BinOp::Sub => Inst::FSub { result, lhs, rhs },
                BinOp::Mul => Inst::FMul { result, lhs, rhs },
                BinOp::Div => Inst::FDiv { result, lhs, rhs },
                BinOp::Eq => Inst::FCmp {
                    result,
                    pred: FloatPred::Oeq,
                    lhs,
                    rhs,
                },
                BinOp::Ne => Inst::FCmp {
                    result,
                    pred: FloatPred::One,
                    lhs,
                    rhs,
                },
                BinOp::Lt => Inst::FCmp {
                    result,
                    pred: FloatPred::Olt,
                    lhs,
                    rhs,
                },
                BinOp::Gt => Inst::FCmp {
                    result,
                    pred: FloatPred::Ogt,
                    lhs,
                    rhs,
                },
                BinOp::Le => Inst::FCmp {
                    result,
                    pred: FloatPred::Ole,
                    lhs,
                    rhs,
                },
                BinOp::Ge => Inst::FCmp {
                    result,
                    pred: FloatPred::Oge,
                    lhs,
                    rhs,
                },
                _ => {
                    self.errors.push(LowerError::UnsupportedConstruct {
                        function: self.func.name.clone(),
                        detail: format!("unsupported float operator {:?}", op),
                    });
                    return None;
                }
            }
        } else {
            match op {
                BinOp::Add => Inst::IAdd { result, lhs, rhs },
                BinOp::Sub => Inst::ISub { result, lhs, rhs },
                BinOp::Mul => Inst::IMul { result, lhs, rhs },
                BinOp::Div => Inst::IDiv {
                    result,
                    lhs,
                    rhs,
                    signed: matches!(same_ty, IrType::I32 | IrType::I64),
                },
                BinOp::Rem => Inst::IRem {
                    result,
                    lhs,
                    rhs,
                    signed: matches!(same_ty, IrType::I32 | IrType::I64),
                },
                BinOp::Eq => Inst::ICmp {
                    result,
                    pred: IntPred::Eq,
                    lhs,
                    rhs,
                },
                BinOp::Ne => Inst::ICmp {
                    result,
                    pred: IntPred::Ne,
                    lhs,
                    rhs,
                },
                BinOp::Lt => Inst::ICmp {
                    result,
                    pred: int_pred_for_less(Some(same_ty.clone())),
                    lhs,
                    rhs,
                },
                BinOp::Gt => Inst::ICmp {
                    result,
                    pred: int_pred_for_greater(Some(same_ty.clone())),
                    lhs,
                    rhs,
                },
                BinOp::Le => Inst::ICmp {
                    result,
                    pred: int_pred_for_less_equal(Some(same_ty.clone())),
                    lhs,
                    rhs,
                },
                BinOp::Ge => Inst::ICmp {
                    result,
                    pred: int_pred_for_greater_equal(Some(same_ty.clone())),
                    lhs,
                    rhs,
                },
                BinOp::LogicalAnd | BinOp::BitAnd => Inst::And { result, lhs, rhs },
                BinOp::LogicalOr | BinOp::BitOr => Inst::Or { result, lhs, rhs },
                BinOp::BitXor => Inst::Xor { result, lhs, rhs },
                BinOp::Shl => Inst::Shl { result, lhs, rhs },
                BinOp::Shr => Inst::Shr {
                    result,
                    lhs,
                    rhs,
                    signed: matches!(same_ty, IrType::I32 | IrType::I64),
                },
            }
        };
        self.func.append_inst(self.current_block, inst);
        Some(result)
    }

    fn lower_lit(&mut self, lit: &Lit) -> Value {
        match lit {
            Lit::Int(v) => {
                let ty = if u32::try_from(*v).is_ok() {
                    IrType::U32
                } else {
                    IrType::U64
                };
                let value = self.func.fresh_value(ty);
                self.func.append_inst(
                    self.current_block,
                    Inst::Const {
                        result: value,
                        value: if u32::try_from(*v).is_ok() {
                            ConstValue::U32(*v as u32)
                        } else {
                            ConstValue::U64(*v)
                        },
                    },
                );
                value
            }
            Lit::Float(v) => {
                let value = self.func.fresh_value(IrType::F64);
                self.func.append_inst(
                    self.current_block,
                    Inst::Const {
                        result: value,
                        value: ConstValue::F64(*v),
                    },
                );
                value
            }
            Lit::Bool(v) => {
                let value = self.func.fresh_value(IrType::Bool);
                self.func.append_inst(
                    self.current_block,
                    Inst::Const {
                        result: value,
                        value: ConstValue::Bool(*v),
                    },
                );
                value
            }
            Lit::String(_) => {
                let value = self.func.fresh_value(IrType::Ptr {
                    elem: Box::new(IrType::U32),
                    mutable: false,
                });
                self.func.append_inst(
                    self.current_block,
                    Inst::Undef {
                        result: value,
                        ty: IrType::Ptr {
                            elem: Box::new(IrType::U32),
                            mutable: false,
                        },
                    },
                );
                value
            }
        }
    }

    fn lower_lvalue_ptr(&mut self, expr: &Expr<'_>) -> Option<Value> {
        match expr {
            Expr::Index { base, index, .. } => self.lower_index_ptr(base, index),
            Expr::Ident { symbol, .. } => self.read_var(*symbol),
            _ => None,
        }
    }

    fn lower_index_ptr(&mut self, base: &Expr<'_>, index: &Expr<'_>) -> Option<Value> {
        let base_val = self.lower_expr(base)?;
        let idx_val = self.lower_expr(index)?;
        let base_ty = self.func.value_type(base_val).cloned().unwrap_or(IrType::Ptr {
            elem: Box::new(IrType::F32),
            mutable: false,
        });
        let elem_ty = base_ty.pointee().cloned().unwrap_or(IrType::F32);
        let ptr_ty = IrType::Ptr {
            elem: Box::new(elem_ty),
            mutable: true,
        };
        let result = self.func.fresh_value(ptr_ty);
        self.func.append_inst(
            self.current_block,
            Inst::GetElementPtr {
                result,
                base: base_val,
                indices: vec![idx_val],
            },
        );
        Some(result)
    }

    fn emit_call(&mut self, name: String, args: Vec<Value>, ret_ty: IrType) -> Option<Value> {
        let result = if matches!(ret_ty, IrType::Void) {
            None
        } else {
            Some(self.func.fresh_value(ret_ty))
        };
        self.func.append_inst(
            self.current_block,
            Inst::Call {
                result,
                func: FuncRef::new(name),
                args,
            },
        );
        result
    }

    fn read_var(&mut self, symbol: Symbol) -> Option<Value> {
        if !self.var_types.contains_key(&symbol) {
            self.errors.push(LowerError::UndefinedVariable {
                function: self.func.name.clone(),
                name: self
                    .symbol_name(symbol)
                    .unwrap_or_else(|| format!("sym_{}", symbol.0)),
            });
            return None;
        }
        Some(self.ssa.read_variable(
            symbol,
            self.current_block,
            &mut self.func,
            &self.var_types,
        ))
    }

    fn live_symbols_in_current_block(&mut self) -> Vec<Symbol> {
        let mut symbols = self.var_types.keys().copied().collect::<Vec<_>>();
        symbols.sort_by_key(|s| s.0);
        symbols
            .into_iter()
            .filter(|s| {
                self.ssa
                    .current_def
                    .contains_key(&(self.current_block, *s))
                    || self.read_var(*s).is_some()
            })
            .collect()
    }

    fn const_like(&mut self, like: Value, n: u32) -> Value {
        let ty = self.func.value_type(like).cloned().unwrap_or(IrType::U32);
        let c = self.func.fresh_value(ty.clone());
        let const_value = match ty {
            IrType::I32 => ConstValue::I32(n as i32),
            IrType::I64 => ConstValue::I64(n as i64),
            IrType::U64 => ConstValue::U64(n as u64),
            _ => ConstValue::U32(n),
        };
        self.func.append_inst(
            self.current_block,
            Inst::Const {
                result: c,
                value: const_value,
            },
        );
        c
    }

    fn symbol_name(&self, symbol: Symbol) -> Option<String> {
        self.symbols.resolve(symbol).map(ToOwned::to_owned)
    }
}

fn ast_type_to_ir(ty: &Type<'_>) -> IrType {
    match ty {
        Type::Named(_) => IrType::U32,
        Type::F32 => IrType::F32,
        Type::F64 => IrType::F64,
        Type::I32 => IrType::I32,
        Type::I64 => IrType::I64,
        Type::U32 => IrType::U32,
        Type::U64 => IrType::U64,
        Type::Bool => IrType::Bool,
        Type::Usize => IrType::U64,
        Type::Slice { elem, mutable } => IrType::Ptr {
            elem: Box::new(ast_type_to_ir(elem)),
            mutable: *mutable,
        },
        Type::Ptr { elem, mutable } => IrType::Ptr {
            elem: Box::new(ast_type_to_ir(elem)),
            mutable: *mutable,
        },
        Type::Vector { elem, lanes } => IrType::Vector {
            elem: Box::new(ast_type_to_ir(elem)),
            lanes: *lanes,
        },
        Type::Void => IrType::Void,
    }
}

fn kernel_name(kernel: &KernelDef<'_>, symbols: &SymbolTable) -> String {
    symbols
        .resolve(kernel.name)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("kernel_{}", kernel.name.0))
}

fn int_pred_for_less(ty: Option<IrType>) -> IntPred {
    match ty {
        Some(IrType::I32 | IrType::I64) => IntPred::Slt,
        _ => IntPred::Ult,
    }
}

fn int_pred_for_less_equal(ty: Option<IrType>) -> IntPred {
    match ty {
        Some(IrType::I32 | IrType::I64) => IntPred::Sle,
        _ => IntPred::Ule,
    }
}

fn int_pred_for_greater(ty: Option<IrType>) -> IntPred {
    match ty {
        Some(IrType::I32 | IrType::I64) => IntPred::Sgt,
        _ => IntPred::Ugt,
    }
}

fn int_pred_for_greater_equal(ty: Option<IrType>) -> IntPred {
    match ty {
        Some(IrType::I32 | IrType::I64) => IntPred::Sge,
        _ => IntPred::Uge,
    }
}

#[cfg(test)]
mod tests {
    use bumpalo::Bump;
    use forge_lang::{resolve_module, Lexer, ParseSession};

    use super::*;
    use crate::verify::verify_function;

    fn lower_src(src: &str) -> ForgeFunction {
        let arena = Bump::new();
        let (tokens, lex_errors) = Lexer::new(src).lex_all();
        assert!(lex_errors.is_empty(), "lex errors: {lex_errors:#?}");
        let tokens = arena.alloc_slice_fill_iter(tokens);
        let mut parser = ParseSession::new(&arena, tokens);
        let module = parser.parse_module();
        assert!(parser.errors.is_empty(), "parse errors: {:#?}", parser.errors);
        let (_scope_tree, resolve_errors) = resolve_module(&module, &parser.symbols);
        assert!(
            resolve_errors.is_empty(),
            "resolution errors: {resolve_errors:#?}"
        );
        let ir = lower_module(&module, &parser.symbols).expect("lowering should succeed");
        ir.functions[0].clone()
    }

    #[test]
    fn lowers_for_range_with_block_param_for_induction_var() {
        let src = r#"
@kernel
fn k(n: u32) {
    let mut acc: u32 = 0u32;
    for k in 0u32..n {
        acc += k;
    }
}
"#;
        let func = lower_src(src);
        let has_header_param = func
            .blocks
            .iter()
            .any(|b| !b.params.is_empty() && b.params.len() >= 1);
        assert!(has_header_param, "expected loop header block params");
    }

    #[test]
    fn lowers_compound_assign_index_in_expected_order() {
        let src = r#"
@kernel
fn k(a: &mut [f32], i: u32, x: f32) {
    a[i] += x;
}
"#;
        let func = lower_src(src);
        let mut saw_gep = false;
        let mut saw_load = false;
        let mut saw_add = false;
        let mut saw_store = false;
        for inst in &func.insts {
            match inst {
                Inst::GetElementPtr { .. } => saw_gep = true,
                Inst::Load { .. } => saw_load = saw_gep,
                Inst::FAdd { .. } | Inst::IAdd { .. } => saw_add = saw_load,
                Inst::Store { .. } => saw_store = saw_add,
                _ => {}
            }
        }
        assert!(saw_store, "expected GEP -> Load -> Add -> Store sequence");
    }

    #[test]
    fn lowered_ir_verifies_without_errors() {
        let src = r#"
@kernel
fn k(a: &[f32], b: &[f32], out: &mut [f32], n: u32) {
    let row = thread::x();
    if row < n {
        out[row] = a[row] + b[row];
    }
}
"#;
        let func = lower_src(src);
        let errors = verify_function(&func);
        assert!(errors.is_empty(), "verifier errors: {errors:#?}");
    }
}
