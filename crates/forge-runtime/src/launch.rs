use rayon::prelude::*;

/// 3-dimensional kernel launch dimensions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LaunchConfig {
    /// Grid dimension X.
    pub grid_x: u32,
    /// Grid dimension Y.
    pub grid_y: u32,
    /// Grid dimension Z.
    pub grid_z: u32,
    /// Block dimension X.
    pub block_x: u32,
    /// Block dimension Y.
    pub block_y: u32,
    /// Block dimension Z.
    pub block_z: u32,
}

impl LaunchConfig {
    /// Constructs a simple 2-D configuration (z=1 everywhere).
    pub const fn new_2d(grid_x: u32, grid_y: u32, block_x: u32, block_y: u32) -> Self {
        Self { grid_x, grid_y, grid_z: 1, block_x, block_y, block_z: 1 }
    }

    /// Returns the total number of logical threads launched.
    pub const fn total_threads(self) -> u64 {
        self.grid_x as u64
            * self.grid_y as u64
            * self.grid_z as u64
            * self.block_x as u64
            * self.block_y as u64
            * self.block_z as u64
    }
}

/// Thread context passed to each kernel invocation.
///
/// Mirrors the GPU programming model: the kernel reads these to compute
/// per-thread memory indices.
#[derive(Clone, Copy, Debug)]
pub struct ThreadCtx {
    pub thread_x: u32,
    pub thread_y: u32,
    pub thread_z: u32,
    pub block_x:  u32,
    pub block_y:  u32,
    pub block_z:  u32,
    pub block_dim_x: u32,
    pub block_dim_y: u32,
    pub block_dim_z: u32,
    pub grid_dim_x:  u32,
    pub grid_dim_y:  u32,
    pub grid_dim_z:  u32,
}

impl ThreadCtx {
    /// Returns the flat global thread index along X.
    #[inline(always)]
    pub const fn global_x(self) -> u32 {
        self.block_x * self.block_dim_x + self.thread_x
    }

    /// Returns the flat global thread index along Y.
    #[inline(always)]
    pub const fn global_y(self) -> u32 {
        self.block_y * self.block_dim_y + self.thread_y
    }

    /// Returns the flat global thread index along Z.
    #[inline(always)]
    pub const fn global_z(self) -> u32 {
        self.block_z * self.block_dim_z + self.thread_z
    }
}

/// Launches a kernel closure over a 3D grid/block topology.
///
/// Blocks are distributed across the Rayon thread pool; threads within a
/// block execute sequentially to match the barrier-synchronisation model.
pub fn launch_3d<F>(cfg: LaunchConfig, kernel: F)
where
    F: Fn(ThreadCtx) + Sync + Send,
{
    let total_blocks = cfg.grid_x as usize * cfg.grid_y as usize * cfg.grid_z as usize;
    (0..total_blocks).into_par_iter().for_each(|block_id| {
        let bx = (block_id as u32) % cfg.grid_x;
        let by = ((block_id as u32) / cfg.grid_x) % cfg.grid_y;
        let bz = (block_id as u32) / (cfg.grid_x * cfg.grid_y);

        for tz in 0..cfg.block_z {
            for ty in 0..cfg.block_y {
                for tx in 0..cfg.block_x {
                    kernel(ThreadCtx {
                        thread_x: tx,
                        thread_y: ty,
                        thread_z: tz,
                        block_x: bx,
                        block_y: by,
                        block_z: bz,
                        block_dim_x: cfg.block_x,
                        block_dim_y: cfg.block_y,
                        block_dim_z: cfg.block_z,
                        grid_dim_x: cfg.grid_x,
                        grid_dim_y: cfg.grid_y,
                        grid_dim_z: cfg.grid_z,
                    });
                }
            }
        }
    });
}

/// Convenience 2-D launcher that wraps `launch_3d`.
pub fn launch_2d<F>(cfg: LaunchConfig, kernel: F)
where
    F: Fn(u32, u32, u32, u32) + Sync + Send,
{
    launch_3d(cfg, move |ctx| kernel(ctx.thread_x, ctx.thread_y, ctx.block_x, ctx.block_y));
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    #[test]
    fn launch_3d_covers_all_threads() {
        let cfg = LaunchConfig {
            grid_x: 4, grid_y: 2, grid_z: 2,
            block_x: 8, block_y: 2, block_z: 2,
        };
        let counter = AtomicU64::new(0);
        launch_3d(cfg, |_ctx| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), cfg.total_threads());
    }

    #[test]
    fn thread_ctx_global_indices_correct() {
        let ctx = ThreadCtx {
            thread_x: 3, thread_y: 1, thread_z: 0,
            block_x: 2, block_y: 1, block_z: 0,
            block_dim_x: 8, block_dim_y: 4, block_dim_z: 1,
            grid_dim_x: 4, grid_dim_y: 2, grid_dim_z: 1,
        };
        assert_eq!(ctx.global_x(), 2 * 8 + 3);  // 19
        assert_eq!(ctx.global_y(), 1 * 4 + 1);  // 5
        assert_eq!(ctx.global_z(), 0);
    }
}
