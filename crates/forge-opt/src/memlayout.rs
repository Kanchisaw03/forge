use std::collections::HashMap;

use forge_ir::{ConstValue, ForgeFunction, Inst};

use crate::pass::FunctionPass;

/// Memory layout tuning pass.
#[derive(Default)]
pub struct MemoryLayoutOptimization;

impl FunctionPass for MemoryLayoutOptimization {
    fn name(&self) -> &'static str {
        "MemoryLayoutOptimization"
    }

    fn run(&mut self, func: &mut ForgeFunction) -> bool {
        let mut alignment: HashMap<forge_ir::Value, u32> = HashMap::new();
        for inst in &func.insts {
            match inst {
                Inst::Alloca { result, .. } => {
                    alignment.insert(*result, 32);
                }
                Inst::GetElementPtr { result, base, indices } => {
                    let base_align = alignment.get(base).copied().unwrap_or(4);
                    let mut out = base_align;
                    if let Some(idx) = indices.first() {
                        if let Some(ConstValue::U32(0)) | Some(ConstValue::U64(0)) = constant_for(func, *idx) {
                            out = base_align;
                        }
                    }
                    alignment.insert(*result, out.max(4));
                }
                _ => {}
            }
        }

        let mut changed = false;
        for inst in &mut func.insts {
            match inst {
                Inst::Load { ptr, align, .. } => {
                    let desired = alignment.get(ptr).copied().unwrap_or(*align).max(*align);
                    if desired != *align {
                        *align = desired;
                        changed = true;
                    }
                }
                Inst::Store { ptr, align, .. } => {
                    let desired = alignment.get(ptr).copied().unwrap_or(*align).max(*align);
                    if desired != *align {
                        *align = desired;
                        changed = true;
                    }
                }
                _ => {}
            }
        }
        changed
    }
}

fn constant_for(func: &ForgeFunction, value: forge_ir::Value) -> Option<ConstValue> {
    let def = func.values.get(value.0 as usize)?.def?;
    match &func.insts[def.0 as usize] {
        Inst::Const { value, .. } => Some(value.clone()),
        _ => None,
    }
}
