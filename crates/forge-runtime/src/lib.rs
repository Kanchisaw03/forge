//! Forge runtime primitives.

pub mod barrier;
pub mod launch;
pub mod pool;
pub mod shared;
pub mod tensor;
pub mod warp;

pub use barrier::SenseBarrier;
pub use launch::{launch_2d, launch_3d, LaunchConfig, ThreadCtx};
pub use pool::{deque, Stealer, Worker};
pub use shared::{ArgBuilder, KernelArgs, SharedMemory};
pub use tensor::{DType, Tensor, TensorAllocator, SIZE_CLASSES};
pub use warp::{blend_lanes, branch_masks, WarpMask, WARP_LANES};

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[test]
    fn end_to_end_runtime_launch_writes_expected_values() {
        let cfg = LaunchConfig::new_2d(2, 1, 4, 1);
        let out = Arc::new(Mutex::new(vec![0u32; 8]));
        launch_3d(cfg, {
            let out = Arc::clone(&out);
            move |ctx| {
                let idx = ctx.block_x * 4 + ctx.thread_x;
                out.lock().expect("lock should succeed")[idx as usize] = idx;
            }
        });
        let out = out.lock().expect("lock should succeed");
        assert_eq!(out.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    }
}
