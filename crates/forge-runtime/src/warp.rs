/// Number of lanes in Forge's AVX2 warp simulation.
pub const WARP_LANES: usize = 8;

/// Per-lane execution mask.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WarpMask {
    lanes: [bool; WARP_LANES],
}

impl WarpMask {
    /// Creates a fully-enabled mask.
    pub const fn full() -> Self {
        Self {
            lanes: [true; WARP_LANES],
        }
    }

    /// Creates a mask from lane booleans.
    pub const fn from_lanes(lanes: [bool; WARP_LANES]) -> Self {
        Self { lanes }
    }

    /// Returns whether lane `idx` is active.
    pub fn lane(self, idx: usize) -> bool {
        self.lanes[idx]
    }

    /// Computes lane-wise conjunction.
    pub fn and(self, other: Self) -> Self {
        let mut out = [false; WARP_LANES];
        for (i, lane) in out.iter_mut().enumerate().take(WARP_LANES) {
            *lane = self.lanes[i] && other.lanes[i];
        }
        Self { lanes: out }
    }

    /// Computes lane-wise negation.
    pub fn not(self) -> Self {
        let mut out = [false; WARP_LANES];
        for (i, lane) in out.iter_mut().enumerate().take(WARP_LANES) {
            *lane = !self.lanes[i];
        }
        Self { lanes: out }
    }
}

/// Applies branch masking to produce then/else execution masks.
pub fn branch_masks(active: WarpMask, cond: WarpMask) -> (WarpMask, WarpMask) {
    let then_mask = active.and(cond);
    let else_mask = active.and(cond.not());
    (then_mask, else_mask)
}

/// Lane-wise blend operation used for divergent branch reconvergence.
pub fn blend_lanes<T: Copy>(a: [T; WARP_LANES], b: [T; WARP_LANES], mask: WarpMask) -> [T; WARP_LANES] {
    let mut out = a;
    for i in 0..WARP_LANES {
        out[i] = if mask.lane(i) { b[i] } else { a[i] };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blends_by_mask() {
        let a = [1u32; WARP_LANES];
        let b = [2u32; WARP_LANES];
        let mask = WarpMask::from_lanes([true, false, true, false, true, false, true, false]);
        let out = blend_lanes(a, b, mask);
        assert_eq!(out, [2, 1, 2, 1, 2, 1, 2, 1]);
    }
}
