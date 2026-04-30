use std::collections::{HashMap, HashSet, VecDeque};

use thiserror::Error;

use crate::func::ForgeFunction;
use crate::inst::{Inst, Terminator};
use crate::types::{BlockId, InstId, IrType, Value};

/// Structural IR verifier diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifyError {
    /// A value is defined more than once.
    #[error("value v{value} has {count} definitions")]
    DuplicateDefinition { value: u32, count: usize },

    /// A value operand references no existing value.
    #[error("unknown value operand v{value}")]
    UnknownValue { value: u32 },

    /// A block cannot be reached from entry.
    #[error("unreachable block b{block}")]
    UnreachableBlock { block: u32 },

    /// Instruction operands violate required type rules.
    #[error("type mismatch at instruction i{inst}: {detail}")]
    TypeMismatch { inst: u32, detail: String },

    /// An operand is used before its definition in the same block.
    #[error("use-before-definition of v{value} at instruction i{inst}")]
    UseBeforeDefinition { value: u32, inst: u32 },

    /// Incoming edge argument count does not match target block params.
    #[error("block argument mismatch on edge b{pred} -> b{target}: expected {expected}, found {found}")]
    BlockParamArityMismatch {
        pred: u32,
        target: u32,
        expected: usize,
        found: usize,
    },
}

/// Verifies one function and returns all discovered diagnostics.
pub fn verify_function(func: &ForgeFunction) -> Vec<VerifyError> {
    let mut errors = Vec::new();
    verify_definitions(func, &mut errors);
    verify_operands_exist(func, &mut errors);
    verify_reachability(func, &mut errors);
    verify_block_arity(func, &mut errors);
    verify_type_rules(func, &mut errors);
    verify_use_before_def(func, &mut errors);
    errors
}

fn verify_definitions(func: &ForgeFunction, errors: &mut Vec<VerifyError>) {
    let mut defs: HashMap<Value, usize> = HashMap::new();
    for block in &func.blocks {
        for param in &block.params {
            *defs.entry(*param).or_insert(0) += 1;
        }
        for inst_id in &block.insts {
            if let Some(v) = func.inst(*inst_id).result() {
                *defs.entry(v).or_insert(0) += 1;
            }
        }
    }
    for (value, count) in defs {
        if count > 1 {
            errors.push(VerifyError::DuplicateDefinition {
                value: value.0,
                count,
            });
        }
    }
}

fn verify_operands_exist(func: &ForgeFunction, errors: &mut Vec<VerifyError>) {
    let max = func.values.len() as u32;
    for block in &func.blocks {
        for inst_id in &block.insts {
            for operand in func.inst(*inst_id).operands() {
                if operand.0 >= max {
                    errors.push(VerifyError::UnknownValue { value: operand.0 });
                }
            }
        }
        for operand in block.terminator.operands() {
            if operand.0 >= max {
                errors.push(VerifyError::UnknownValue { value: operand.0 });
            }
        }
    }
}

fn verify_reachability(func: &ForgeFunction, errors: &mut Vec<VerifyError>) {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(func.entry);
    visited.insert(func.entry);
    while let Some(block) = queue.pop_front() {
        for succ in func.block(block).terminator.successors() {
            if visited.insert(succ) {
                queue.push_back(succ);
            }
        }
    }
    for block in func.block_ids() {
        if !visited.contains(&block) {
            errors.push(VerifyError::UnreachableBlock { block: block.0 });
        }
    }
}

fn verify_block_arity(func: &ForgeFunction, errors: &mut Vec<VerifyError>) {
    for pred in func.block_ids() {
        match &func.block(pred).terminator {
            Terminator::Jump { target, args } => {
                let expected = func.block(*target).params.len();
                if expected != args.len() {
                    errors.push(VerifyError::BlockParamArityMismatch {
                        pred: pred.0,
                        target: target.0,
                        expected,
                        found: args.len(),
                    });
                }
            }
            Terminator::Branch {
                then_block,
                then_args,
                else_block,
                else_args,
                ..
            } => {
                let then_expected = func.block(*then_block).params.len();
                if then_expected != then_args.len() {
                    errors.push(VerifyError::BlockParamArityMismatch {
                        pred: pred.0,
                        target: then_block.0,
                        expected: then_expected,
                        found: then_args.len(),
                    });
                }
                let else_expected = func.block(*else_block).params.len();
                if else_expected != else_args.len() {
                    errors.push(VerifyError::BlockParamArityMismatch {
                        pred: pred.0,
                        target: else_block.0,
                        expected: else_expected,
                        found: else_args.len(),
                    });
                }
            }
            Terminator::Return(_) | Terminator::Unreachable => {}
        }
    }
}

fn verify_type_rules(func: &ForgeFunction, errors: &mut Vec<VerifyError>) {
    for (idx, inst) in func.insts.iter().enumerate() {
        match inst {
            Inst::FAdd { lhs, rhs, .. }
            | Inst::FSub { lhs, rhs, .. }
            | Inst::FMul { lhs, rhs, .. }
            | Inst::FDiv { lhs, rhs, .. } => {
                let lt = func.value_type(*lhs).cloned();
                let rt = func.value_type(*rhs).cloned();
                if !matches!(lt, Some(IrType::F32 | IrType::F64))
                    || !matches!(rt, Some(IrType::F32 | IrType::F64))
                    || lt != rt
                {
                    errors.push(VerifyError::TypeMismatch {
                        inst: idx as u32,
                        detail: "float binary operands must match and be floating".to_owned(),
                    });
                }
            }
            Inst::IAdd { lhs, rhs, .. }
            | Inst::ISub { lhs, rhs, .. }
            | Inst::IMul { lhs, rhs, .. }
            | Inst::And { lhs, rhs, .. }
            | Inst::Or { lhs, rhs, .. }
            | Inst::Xor { lhs, rhs, .. }
            | Inst::Shl { lhs, rhs, .. }
            | Inst::Shr { lhs, rhs, .. } => {
                let lt = func.value_type(*lhs).cloned();
                let rt = func.value_type(*rhs).cloned();
                if lt != rt || !lt.unwrap_or(IrType::Void).is_integer() {
                    errors.push(VerifyError::TypeMismatch {
                        inst: idx as u32,
                        detail: "integer binary operands must match and be integral".to_owned(),
                    });
                }
            }
            Inst::Load { ptr, ty, .. } => {
                let pt = func.value_type(*ptr).cloned();
                if pt.and_then(|t| t.pointee().cloned()) != Some(ty.clone()) {
                    errors.push(VerifyError::TypeMismatch {
                        inst: idx as u32,
                        detail: "load pointer type must match loaded type".to_owned(),
                    });
                }
            }
            _ => {}
        }
    }
}

fn verify_use_before_def(func: &ForgeFunction, errors: &mut Vec<VerifyError>) {
    let mut location: HashMap<InstId, (BlockId, usize)> = HashMap::new();
    for block in func.block_ids() {
        for (idx, inst_id) in func.block(block).insts.iter().enumerate() {
            location.insert(*inst_id, (block, idx));
        }
    }
    for block in func.block_ids() {
        for inst_id in &func.block(block).insts {
            let Some((_, use_pos)) = location.get(inst_id).copied() else {
                continue;
            };
            for operand in func.inst(*inst_id).operands() {
                let Some(def_id) = func
                    .values
                    .get(operand.0 as usize)
                    .and_then(|v| v.def)
                else {
                    continue;
                };
                if let Some((def_block, def_pos)) = location.get(&def_id).copied() {
                    if def_block == block && def_pos > use_pos {
                        errors.push(VerifyError::UseBeforeDefinition {
                            value: operand.0,
                            inst: inst_id.0,
                        });
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::func::ForgeFunction;
    use crate::inst::{Inst, Terminator};
    use crate::types::{ConstValue, IrType};

    use super::*;

    #[test]
    fn verifier_accepts_well_formed_function() {
        let mut f = ForgeFunction::new("ok");
        let lhs = f.fresh_value(IrType::F32);
        let rhs = f.fresh_value(IrType::F32);
        f.append_inst(f.entry, Inst::Const { result: lhs, value: ConstValue::F32(1.0) });
        f.append_inst(f.entry, Inst::Const { result: rhs, value: ConstValue::F32(2.0) });
        let out = f.fresh_value(IrType::F32);
        f.append_inst(f.entry, Inst::FAdd { result: out, lhs, rhs });
        f.set_terminator(f.entry, Terminator::Return(Some(out)));
        let errors = verify_function(&f);
        assert!(errors.is_empty(), "unexpected verifier errors: {errors:#?}");
    }

    #[test]
    fn verifier_reports_duplicate_definition() {
        let mut f = ForgeFunction::new("bad");
        let v = f.fresh_value(IrType::U32);
        f.append_inst(
            f.entry,
            Inst::Const {
                result: v,
                value: ConstValue::U32(1),
            },
        );
        f.append_inst(
            f.entry,
            Inst::Const {
                result: v,
                value: ConstValue::U32(2),
            },
        );
        f.set_terminator(f.entry, Terminator::Return(None));
        let errors = verify_function(&f);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, VerifyError::DuplicateDefinition { .. })),
            "expected duplicate definition error, got {errors:#?}"
        );
    }
}
