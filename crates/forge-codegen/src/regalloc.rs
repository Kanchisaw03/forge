use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};

use forge_ir::{ForgeFunction, IrType, Value};

/// Physical AVX2 register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysReg {
    /// YMM register index.
    Ymm(u8),
}

/// Assigned location for one SSA value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Location {
    /// Register-resident value.
    Register(PhysReg),
    /// Spilled stack slot offset (bytes).
    Stack(u32),
}

/// One liveness interval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveInterval {
    /// Value being assigned.
    pub value: Value,
    /// First instruction index where value is live.
    pub start: u32,
    /// Last instruction index where value is live.
    pub end: u32,
}

impl Ord for LiveInterval {
    fn cmp(&self, other: &Self) -> Ordering {
        self.end
            .cmp(&other.end)
            .then_with(|| self.start.cmp(&other.start))
            .then_with(|| self.value.0.cmp(&other.value.0))
    }
}

impl PartialOrd for LiveInterval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Linear-scan allocator for YMM registers.
pub struct LinearScanAllocator {
    free_regs: Vec<PhysReg>,
    active: BTreeSet<LiveInterval>,
    allocation: HashMap<Value, Location>,
    next_spill_slot: u32,
}

impl LinearScanAllocator {
    /// Allocates locations for all non-void values in `func`.
    pub fn allocate(func: &ForgeFunction) -> HashMap<Value, Location> {
        let mut alloc = Self {
            free_regs: (0..14).rev().map(PhysReg::Ymm).collect(),
            active: BTreeSet::new(),
            allocation: HashMap::new(),
            next_spill_slot: 0,
        };
        let mut intervals = compute_live_intervals(func);
        intervals.sort_by_key(|i| i.start);
        for interval in intervals {
            alloc.expire_old_intervals(interval.start);
            if let Some(reg) = alloc.free_regs.pop() {
                alloc.allocation.insert(interval.value, Location::Register(reg));
                alloc.active.insert(interval);
            } else {
                alloc.spill_at_interval(interval);
            }
        }
        alloc.allocation
    }

    fn expire_old_intervals(&mut self, current_start: u32) {
        let expired: Vec<_> = self
            .active
            .iter()
            .filter(|i| i.end < current_start)
            .cloned()
            .collect();
        for interval in expired {
            self.active.remove(&interval);
            if let Some(Location::Register(reg)) = self.allocation.get(&interval.value).copied() {
                self.free_regs.push(reg);
            }
        }
    }

    fn spill_at_interval(&mut self, interval: LiveInterval) {
        let Some(last) = self.active.iter().next_back().cloned() else {
            let slot = self.allocate_spill_slot();
            self.allocation.insert(interval.value, Location::Stack(slot));
            return;
        };
        if last.end > interval.end {
            if let Some(loc) = self.allocation.get(&last.value).copied() {
                self.allocation.insert(interval.value, loc);
            }
            let slot = self.allocate_spill_slot();
            self.allocation.insert(last.value, Location::Stack(slot));
            self.active.remove(&last);
            self.active.insert(interval);
        } else {
            let slot = self.allocate_spill_slot();
            self.allocation.insert(interval.value, Location::Stack(slot));
        }
    }

    fn allocate_spill_slot(&mut self) -> u32 {
        let slot = self.next_spill_slot;
        self.next_spill_slot = self.next_spill_slot.saturating_add(32);
        slot
    }
}

fn compute_live_intervals(func: &ForgeFunction) -> Vec<LiveInterval> {
    let mut inst_number = HashMap::new();
    let mut counter = 0u32;
    for block in func.block_ids() {
        for inst_id in &func.block(block).insts {
            inst_number.insert(*inst_id, counter);
            counter += 1;
        }
    }

    let mut intervals = Vec::new();
    for value in &func.values {
        if matches!(value.ty, IrType::Void) {
            continue;
        }
        let start = value
            .def
            .and_then(|d| inst_number.get(&d).copied())
            .unwrap_or(0);
        let end = value
            .uses
            .iter()
            .filter_map(|u| inst_number.get(&u.inst).copied())
            .max()
            .unwrap_or(start);
        intervals.push(LiveInterval {
            value: value.id,
            start,
            end,
        });
    }
    intervals
}

#[cfg(test)]
mod tests {
    use forge_ir::{ConstValue, ForgeFunction, Inst, IrType, Terminator};

    use super::*;

    #[test]
    fn allocates_registers_for_simple_values() {
        let mut f = ForgeFunction::new("ra");
        let a = f.fresh_value(IrType::F32);
        let b = f.fresh_value(IrType::F32);
        let c = f.fresh_value(IrType::F32);
        f.append_inst(
            f.entry,
            Inst::Const {
                result: a,
                value: ConstValue::F32(1.0),
            },
        );
        f.append_inst(
            f.entry,
            Inst::Const {
                result: b,
                value: ConstValue::F32(2.0),
            },
        );
        f.append_inst(f.entry, Inst::FAdd { result: c, lhs: a, rhs: b });
        f.set_terminator(f.entry, Terminator::Return(Some(c)));
        f.rebuild_uses();
        let map = LinearScanAllocator::allocate(&f);
        assert!(matches!(map.get(&a), Some(Location::Register(_))));
        assert!(matches!(map.get(&b), Some(Location::Register(_))));
        assert!(matches!(map.get(&c), Some(Location::Register(_))));
    }
}
