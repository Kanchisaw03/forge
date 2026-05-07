use rayon::prelude::*;
use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::cell::RefCell;
use std::ptr::NonNull;
use std::sync::{Once, OnceLock};

const MC_DEFAULT: usize = 192;
const KC_DEFAULT: usize = 256;
const NC_DEFAULT: usize = 2048;
const MR: usize = 6;
const NR_AVX2: usize = 16;
const NR_AVX512: usize = 32;
const PACK_ALIGN_BYTES: usize = 64;

static TILING_LOG_ONCE: Once = Once::new();
static TILE_BLOCKS: OnceLock<TileConfig> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KernelKind {
    Scalar,
    Avx2,
    Avx512,
}

impl KernelKind {
    #[inline(always)]
    fn nr(self) -> usize {
        match self {
            Self::Avx512 => NR_AVX512,
            Self::Avx2 | Self::Scalar => NR_AVX2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TileConfig {
    mc: usize,
    kc: usize,
    nc: usize,
}

thread_local! {
    static PACK_A_BUF: RefCell<AlignedPackBuffer> =
        RefCell::new(AlignedPackBuffer::new(PACK_ALIGN_BYTES));
}

struct AlignedPackBuffer {
    ptr: NonNull<f32>,
    capacity: usize,
    align: usize,
}

impl AlignedPackBuffer {
    const fn new(align: usize) -> Self {
        Self {
            ptr: NonNull::dangling(),
            capacity: 0,
            align,
        }
    }

    fn as_mut_slice(&mut self, len: usize) -> &mut [f32] {
        self.ensure_capacity(len);
        // SAFETY: `ensure_capacity` guarantees allocation for `len` f32 values.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), len) }
    }

    fn ensure_capacity(&mut self, len: usize) {
        if len <= self.capacity {
            return;
        }
        let new_capacity = len.next_power_of_two();
        self.grow(new_capacity);
    }

    fn grow(&mut self, new_capacity: usize) {
        debug_assert!(new_capacity > self.capacity);
        let new_layout = Self::layout(new_capacity, self.align);
        // SAFETY: layout is valid and non-zero-sized.
        let new_raw = unsafe { alloc(new_layout) };
        if new_raw.is_null() {
            handle_alloc_error(new_layout);
        }
        // SAFETY: null already handled above.
        let new_ptr = unsafe { NonNull::new_unchecked(new_raw as *mut f32) };

        if self.capacity != 0 {
            let old_layout = Self::layout(self.capacity, self.align);
            // SAFETY: pointer/layout pair matches previous allocation.
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, old_layout);
            }
        }

        self.ptr = new_ptr;
        self.capacity = new_capacity;
    }

    fn layout(capacity: usize, align: usize) -> Layout {
        let bytes = capacity
            .checked_mul(std::mem::size_of::<f32>())
            .expect("packed buffer size overflow");
        Layout::from_size_align(bytes, align).expect("packed buffer layout must be valid")
    }
}

impl Drop for AlignedPackBuffer {
    fn drop(&mut self) {
        if self.capacity == 0 {
            return;
        }
        let layout = Self::layout(self.capacity, self.align);
        // SAFETY: pointer/layout pair matches previous allocation.
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

/// SGEMM kernel: `C = alpha * A * B + beta * C` (row-major).
///
/// Invariants:
/// - `a.len() == m * k`
/// - `b.len() == k * n`
/// - `c.len() == m * n`
pub fn sgemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    if m == 0 || n == 0 {
        return;
    }

    if k == 0 {
        if beta == 0.0 {
            c.fill(0.0);
        } else if beta != 1.0 {
            c.iter_mut().for_each(|v| *v *= beta);
        }
        return;
    }

    if beta == 0.0 {
        c.fill(0.0);
    } else if beta != 1.0 {
        c.iter_mut().for_each(|v| *v *= beta);
    }

    if alpha == 0.0 {
        return;
    }

    let kernel = select_kernel_for_problem(detect_kernel_kind(), m, n, k);
    maybe_log_tiling(kernel);
    sgemm_with_kernel(a, b, c, m, n, k, alpha, kernel);
}

/// Compatibility wrapper for plain GEMM where `beta=0`.
///
/// Invariants:
/// - same as [`sgemm`]
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    c.fill(0.0);
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    matmul_packed(a, b, c, m, n, k);
}

#[inline(always)]
fn matmul_packed(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let kernel = select_kernel_for_problem(detect_kernel_kind(), m, n, k);
    maybe_log_tiling(kernel);
    sgemm_with_kernel(a, b, c, m, n, k, 1.0, kernel);
}

fn detect_kernel_kind() -> KernelKind {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return KernelKind::Avx512;
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return KernelKind::Avx2;
        }
    }
    KernelKind::Scalar
}

#[inline(always)]
fn select_kernel_for_problem(detected: KernelKind, m: usize, n: usize, k: usize) -> KernelKind {
    if std::env::var_os("FORGE_FORCE_AVX512").is_some() {
        return if detected == KernelKind::Avx512 {
            KernelKind::Avx512
        } else {
            detected
        };
    }

    if std::env::var_os("FORGE_FORCE_AVX2").is_some()
        || std::env::var_os("FORGE_DISABLE_AVX512").is_some()
    {
        return match detected {
            KernelKind::Avx512 | KernelKind::Avx2 => KernelKind::Avx2,
            KernelKind::Scalar => KernelKind::Scalar,
        };
    }

    let problem = m.saturating_mul(n).saturating_mul(k);
    match detected {
        KernelKind::Avx512 if problem < 512 * 512 * 512 => KernelKind::Avx2,
        other => other,
    }
}

#[allow(clippy::too_many_arguments)]
fn sgemm_with_kernel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    kernel: KernelKind,
) {
    let tiles = tile_blocks();
    let low_core_client = std::thread::available_parallelism()
        .map(|n| n.get() <= 4)
        .unwrap_or(false);
    let kc_block = if low_core_client && k >= 1024 {
        // Large reductions on low-core client CPUs sustain better clocks
        // with a shallower K block.
        tiles.kc.min(128)
    } else {
        tiles.kc
    };
    let nc_block = tiles.nc;
    let nr_actual = kernel.nr();

    // Dynamically shrink MC so we get enough parallel chunks for all cores
    // while keeping each chunk large enough for cache efficiency.
    let num_threads = rayon::current_num_threads().max(1);
    let min_chunks = if num_threads > 2 && m <= 256 {
        2
    } else {
        num_threads.max(4)
    };
    let mc_block = if m > 0 {
        let mc_max = tiles.mc;
        let mc_for_balance = align_down((m + min_chunks - 1) / min_chunks, MR).max(MR);
        mc_max.min(mc_for_balance)
    } else {
        tiles.mc
    };

    let max_kc = kc_block.min(k.max(1));
    let max_nc = nc_block.min(n.max(1));
    let mut packed_b_buf = AlignedPackBuffer::new(PACK_ALIGN_BYTES);
    packed_b_buf.ensure_capacity(div_ceil(max_nc, nr_actual) * max_kc * nr_actual);

    for jc in (0..n).step_by(nc_block) {
        let nc = (n - jc).min(nc_block);

        for pc in (0..k).step_by(kc_block) {
            let kc = (k - pc).min(kc_block);

            let packed_b_len = div_ceil(nc, nr_actual) * kc * nr_actual;
            let packed_b = packed_b_buf.as_mut_slice(packed_b_len);
            if nr_actual == NR_AVX512 {
                pack_b_avx512(b, n, pc, jc, kc, nc, packed_b);
            } else {
                pack_b_avx2(b, n, pc, jc, kc, nc, packed_b);
            }
            let packed_b: &[f32] = packed_b;

            let row_block_elems = n * mc_block;
            c.par_chunks_mut(row_block_elems)
                .enumerate()
                .for_each(|(tile_i, c_rows)| {
                    let ic = tile_i * mc_block;
                    let mc = c_rows.len() / n;

                    PACK_A_BUF.with(|pack_a_buf| {
                        let mut pack_a_buf = pack_a_buf.borrow_mut();
                        let packed_a_len = div_ceil(mc, MR) * kc * MR;
                        let packed_a = pack_a_buf.as_mut_slice(packed_a_len);
                        pack_a(a, k, ic, pc, mc, kc, alpha, packed_a);

                        gebp(
                            packed_a,
                            packed_b,
                            c_rows,
                            n,
                            jc,
                            mc,
                            nc,
                            kc,
                            nr_actual,
                            kernel,
                        );
                    });
                });
        }
    }

    debug_assert!(m * n == c.len());
}

#[inline]
fn pack_a(
    a: &[f32],
    lda: usize,
    ic: usize,
    pc: usize,
    mc: usize,
    kc: usize,
    alpha: f32,
    packed_a: &mut [f32],
) {
    let m_panels = div_ceil(mc, MR);
    for ip in 0..m_panels {
        let row_base = ip * MR;
        let panel_base = ip * kc * MR;
        for kk in 0..kc {
            let src_col = pc + kk;
            let dst_base = panel_base + kk * MR;
            for ii in 0..MR {
                let row = row_base + ii;
                packed_a[dst_base + ii] = if row < mc {
                    alpha * a[(ic + row) * lda + src_col]
                } else {
                    0.0
                };
            }
        }
    }
}

#[inline]
fn pack_b_avx2(
    b: &[f32],
    ldb: usize,
    pc: usize,
    jc: usize,
    kc: usize,
    nc: usize,
    packed_b: &mut [f32],
) {
    pack_b_with_nr(b, ldb, pc, jc, kc, nc, NR_AVX2, packed_b);
}

#[inline]
fn pack_b_avx512(
    b: &[f32],
    ldb: usize,
    pc: usize,
    jc: usize,
    kc: usize,
    nc: usize,
    packed_b: &mut [f32],
) {
    pack_b_with_nr(b, ldb, pc, jc, kc, nc, NR_AVX512, packed_b);
}

#[inline]
fn pack_b_with_nr(
    b: &[f32],
    ldb: usize,
    pc: usize,
    jc: usize,
    kc: usize,
    nc: usize,
    nr: usize,
    packed_b: &mut [f32],
) {
    let n_panels = div_ceil(nc, nr);
    let panel_stride = kc * nr;
    for jp in 0..n_panels {
        let panel_base = jp * panel_stride;
        let panel = &mut packed_b[panel_base..panel_base + panel_stride];
        let col_base = jp * nr;
        for kk in 0..kc {
            let src_row = pc + kk;
            let dst_base = kk * nr;
            for jj in 0..nr {
                let col = col_base + jj;
                panel[dst_base + jj] = if col < nc {
                    b[src_row * ldb + (jc + col)]
                } else {
                    0.0
                };
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gebp(
    packed_a: &[f32],
    packed_b: &[f32],
    c_rows: &mut [f32],
    ldc: usize,
    jc: usize,
    mc: usize,
    nc: usize,
    kc: usize,
    nr_actual: usize,
    kernel: KernelKind,
) {
    let m_panels = div_ceil(mc, MR);
    let n_panels = div_ceil(nc, nr_actual);

    for ip in 0..m_panels {
        let row = ip * MR;
        let a_panel = &packed_a[ip * kc * MR..(ip + 1) * kc * MR];

        for jp in 0..n_panels {
            let col = jp * nr_actual;
            let b_panel = &packed_b[jp * kc * nr_actual..(jp + 1) * kc * nr_actual];
            prefetch_t0(b_panel.as_ptr());
            if jp + 1 < n_panels {
                let next_panel_offset = (jp + 1) * kc * nr_actual;
                prefetch_t0(unsafe { packed_b.as_ptr().add(next_panel_offset) });
            }

            let tile_m = (mc - row).min(MR);
            let tile_n = (nc - col).min(nr_actual);
            let c_offset = row * ldc + (jc + col);

            if tile_m == MR && tile_n == nr_actual {
                match kernel {
                    KernelKind::Avx512 => {
                        #[cfg(target_arch = "x86_64")]
                        {
                            // SAFETY: call site guarantees full MRxNR tile, valid packed panel lengths,
                            // and AVX-512 support via runtime detection before dispatch.
                            unsafe {
                                microkernel_6x32_avx512(
                                    a_panel.as_ptr(),
                                    b_panel.as_ptr(),
                                    c_rows.as_mut_ptr().add(c_offset),
                                    kc,
                                    ldc,
                                );
                            }
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        unreachable!("AVX-512 kernel is only selected on x86_64");
                    }
                    KernelKind::Avx2 => {
                        #[cfg(target_arch = "x86_64")]
                        {
                            // SAFETY: call site guarantees full MRxNR tile, valid packed panel lengths,
                            // and AVX2/FMA support via runtime detection before dispatch.
                            unsafe {
                                microkernel_6x16_avx2(
                                    a_panel.as_ptr(),
                                    b_panel.as_ptr(),
                                    c_rows.as_mut_ptr().add(c_offset),
                                    kc,
                                    ldc,
                                );
                            }
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        unreachable!("AVX2 kernel is only selected on x86_64");
                    }
                    KernelKind::Scalar => {
                        // SAFETY: pointers are derived from in-bounds packed panel slices and C tile base.
                        unsafe {
                            microkernel_scalar_edge(
                                a_panel.as_ptr(),
                                b_panel.as_ptr(),
                                c_rows.as_mut_ptr().add(c_offset),
                                kc,
                                ldc,
                                tile_m,
                                tile_n,
                                nr_actual,
                            );
                        }
                    }
                }
            } else {
                // SAFETY: edge dimensions cap writes/reads to in-bounds regions; packed panels are zero-padded.
                unsafe {
                    microkernel_scalar_edge(
                        a_panel.as_ptr(),
                        b_panel.as_ptr(),
                        c_rows.as_mut_ptr().add(c_offset),
                        kc,
                        ldc,
                        tile_m,
                        tile_n,
                        nr_actual,
                    );
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn microkernel_scalar_edge(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    kc: usize,
    ldc: usize,
    m_tile: usize,
    n_tile: usize,
    nr_stride: usize,
) {
    for ii in 0..m_tile {
        for jj in 0..n_tile {
            let mut acc = 0.0f32;
            for kk in 0..kc {
                let a_val = *a.add(kk * MR + ii);
                let b_val = *b.add(kk * nr_stride + jj);
                acc += a_val * b_val;
            }
            let c_ptr = c.add(ii * ldc + jj);
            *c_ptr += acc;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn microkernel_6x16_avx2(a: *const f32, b: *const f32, c: *mut f32, kc: usize, ldc: usize) {
    use std::arch::x86_64::*;

    // ymm0..ymm5   -> C rows 0..5, cols 0..7
    // ymm6..ymm11  -> C rows 0..5, cols 8..15
    let mut c0 = _mm256_setzero_ps();
    let mut c1 = _mm256_setzero_ps();
    let mut c2 = _mm256_setzero_ps();
    let mut c3 = _mm256_setzero_ps();
    let mut c4 = _mm256_setzero_ps();
    let mut c5 = _mm256_setzero_ps();

    let mut c6 = _mm256_setzero_ps();
    let mut c7 = _mm256_setzero_ps();
    let mut c8 = _mm256_setzero_ps();
    let mut c9 = _mm256_setzero_ps();
    let mut c10 = _mm256_setzero_ps();
    let mut c11 = _mm256_setzero_ps();

    let row0 = c;
    let row1 = c.add(ldc);
    let row2 = c.add(2 * ldc);
    let row3 = c.add(3 * ldc);
    let row4 = c.add(4 * ldc);
    let row5 = c.add(5 * ldc);

    _mm_prefetch(row0 as *const i8, _MM_HINT_T0);
    _mm_prefetch(row1 as *const i8, _MM_HINT_T0);
    _mm_prefetch(row2 as *const i8, _MM_HINT_T0);
    _mm_prefetch(row3 as *const i8, _MM_HINT_T0);
    _mm_prefetch(row4 as *const i8, _MM_HINT_T0);
    _mm_prefetch(row5 as *const i8, _MM_HINT_T0);

    macro_rules! avx2_k_step {
        ($a_ptr:expr, $b_ptr:expr) => {{
            let b0 = _mm256_loadu_ps($b_ptr);
            let b1 = _mm256_loadu_ps($b_ptr.add(8));

            let a0 = _mm256_set1_ps(*$a_ptr);
            c0 = _mm256_fmadd_ps(b0, a0, c0);
            c6 = _mm256_fmadd_ps(b1, a0, c6);

            let a1 = _mm256_set1_ps(*$a_ptr.add(1));
            c1 = _mm256_fmadd_ps(b0, a1, c1);
            c7 = _mm256_fmadd_ps(b1, a1, c7);

            let a2 = _mm256_set1_ps(*$a_ptr.add(2));
            c2 = _mm256_fmadd_ps(b0, a2, c2);
            c8 = _mm256_fmadd_ps(b1, a2, c8);

            let a3 = _mm256_set1_ps(*$a_ptr.add(3));
            c3 = _mm256_fmadd_ps(b0, a3, c3);
            c9 = _mm256_fmadd_ps(b1, a3, c9);

            let a4 = _mm256_set1_ps(*$a_ptr.add(4));
            c4 = _mm256_fmadd_ps(b0, a4, c4);
            c10 = _mm256_fmadd_ps(b1, a4, c10);

            let a5 = _mm256_set1_ps(*$a_ptr.add(5));
            c5 = _mm256_fmadd_ps(b0, a5, c5);
            c11 = _mm256_fmadd_ps(b1, a5, c11);
        }};
    }

    const PREFETCH_K_DIST: usize = 8;
    let kc_main = kc & !3;
    let mut kk = 0usize;
    let mut a_ptr = a;
    let mut b_ptr = b;
    while kk < kc_main {
        if kk + PREFETCH_K_DIST < kc {
            _mm_prefetch(b_ptr.add(PREFETCH_K_DIST * NR_AVX2) as *const i8, _MM_HINT_T0);
            _mm_prefetch(a_ptr.add(PREFETCH_K_DIST * MR) as *const i8, _MM_HINT_T0);
        }

        avx2_k_step!(a_ptr, b_ptr);
        a_ptr = a_ptr.add(MR);
        b_ptr = b_ptr.add(NR_AVX2);

        avx2_k_step!(a_ptr, b_ptr);
        a_ptr = a_ptr.add(MR);
        b_ptr = b_ptr.add(NR_AVX2);

        avx2_k_step!(a_ptr, b_ptr);
        a_ptr = a_ptr.add(MR);
        b_ptr = b_ptr.add(NR_AVX2);

        avx2_k_step!(a_ptr, b_ptr);
        a_ptr = a_ptr.add(MR);
        b_ptr = b_ptr.add(NR_AVX2);

        kk += 4;
    }
    while kk < kc {
        if kk + PREFETCH_K_DIST < kc {
            _mm_prefetch(b_ptr.add(PREFETCH_K_DIST * NR_AVX2) as *const i8, _MM_HINT_T0);
            _mm_prefetch(a_ptr.add(PREFETCH_K_DIST * MR) as *const i8, _MM_HINT_T0);
        }
        avx2_k_step!(a_ptr, b_ptr);
        a_ptr = a_ptr.add(MR);
        b_ptr = b_ptr.add(NR_AVX2);
        kk += 1;
    }

    let old0a = _mm256_loadu_ps(row0);
    let old0b = _mm256_loadu_ps(row0.add(8));
    _mm256_storeu_ps(row0, _mm256_add_ps(old0a, c0));
    _mm256_storeu_ps(row0.add(8), _mm256_add_ps(old0b, c6));

    let old1a = _mm256_loadu_ps(row1);
    let old1b = _mm256_loadu_ps(row1.add(8));
    _mm256_storeu_ps(row1, _mm256_add_ps(old1a, c1));
    _mm256_storeu_ps(row1.add(8), _mm256_add_ps(old1b, c7));

    let old2a = _mm256_loadu_ps(row2);
    let old2b = _mm256_loadu_ps(row2.add(8));
    _mm256_storeu_ps(row2, _mm256_add_ps(old2a, c2));
    _mm256_storeu_ps(row2.add(8), _mm256_add_ps(old2b, c8));

    let old3a = _mm256_loadu_ps(row3);
    let old3b = _mm256_loadu_ps(row3.add(8));
    _mm256_storeu_ps(row3, _mm256_add_ps(old3a, c3));
    _mm256_storeu_ps(row3.add(8), _mm256_add_ps(old3b, c9));

    let old4a = _mm256_loadu_ps(row4);
    let old4b = _mm256_loadu_ps(row4.add(8));
    _mm256_storeu_ps(row4, _mm256_add_ps(old4a, c4));
    _mm256_storeu_ps(row4.add(8), _mm256_add_ps(old4b, c10));

    let old5a = _mm256_loadu_ps(row5);
    let old5b = _mm256_loadu_ps(row5.add(8));
    _mm256_storeu_ps(row5, _mm256_add_ps(old5a, c5));
    _mm256_storeu_ps(row5.add(8), _mm256_add_ps(old5b, c11));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn microkernel_6x32_avx512(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    kc: usize,
    ldc: usize,
) {
    use std::arch::x86_64::*;

    // zmm0..zmm5   -> C rows 0..5, cols 0..15
    // zmm6..zmm11  -> C rows 0..5, cols 16..31
    let mut c0 = _mm512_setzero_ps();
    let mut c1 = _mm512_setzero_ps();
    let mut c2 = _mm512_setzero_ps();
    let mut c3 = _mm512_setzero_ps();
    let mut c4 = _mm512_setzero_ps();
    let mut c5 = _mm512_setzero_ps();

    let mut c6 = _mm512_setzero_ps();
    let mut c7 = _mm512_setzero_ps();
    let mut c8 = _mm512_setzero_ps();
    let mut c9 = _mm512_setzero_ps();
    let mut c10 = _mm512_setzero_ps();
    let mut c11 = _mm512_setzero_ps();

    let row0 = c;
    let row1 = c.add(ldc);
    let row2 = c.add(2 * ldc);
    let row3 = c.add(3 * ldc);
    let row4 = c.add(4 * ldc);
    let row5 = c.add(5 * ldc);

    _mm_prefetch(row0 as *const i8, _MM_HINT_T0);
    _mm_prefetch(row1 as *const i8, _MM_HINT_T0);
    _mm_prefetch(row2 as *const i8, _MM_HINT_T0);
    _mm_prefetch(row3 as *const i8, _MM_HINT_T0);
    _mm_prefetch(row4 as *const i8, _MM_HINT_T0);
    _mm_prefetch(row5 as *const i8, _MM_HINT_T0);

    macro_rules! avx512_k_step {
        ($a_ptr:expr, $b_ptr:expr) => {{
            let b0 = _mm512_loadu_ps($b_ptr);
            let b1 = _mm512_loadu_ps($b_ptr.add(16));

            let a0 = _mm512_set1_ps(*$a_ptr);
            c0 = _mm512_fmadd_ps(b0, a0, c0);
            c6 = _mm512_fmadd_ps(b1, a0, c6);

            let a1 = _mm512_set1_ps(*$a_ptr.add(1));
            c1 = _mm512_fmadd_ps(b0, a1, c1);
            c7 = _mm512_fmadd_ps(b1, a1, c7);

            let a2 = _mm512_set1_ps(*$a_ptr.add(2));
            c2 = _mm512_fmadd_ps(b0, a2, c2);
            c8 = _mm512_fmadd_ps(b1, a2, c8);

            let a3 = _mm512_set1_ps(*$a_ptr.add(3));
            c3 = _mm512_fmadd_ps(b0, a3, c3);
            c9 = _mm512_fmadd_ps(b1, a3, c9);

            let a4 = _mm512_set1_ps(*$a_ptr.add(4));
            c4 = _mm512_fmadd_ps(b0, a4, c4);
            c10 = _mm512_fmadd_ps(b1, a4, c10);

            let a5 = _mm512_set1_ps(*$a_ptr.add(5));
            c5 = _mm512_fmadd_ps(b0, a5, c5);
            c11 = _mm512_fmadd_ps(b1, a5, c11);
        }};
    }

    const PREFETCH_K_DIST: usize = 8;
    let kc_main = kc & !3;
    let mut kk = 0usize;
    let mut a_ptr = a;
    let mut b_ptr = b;
    while kk < kc_main {
        if kk + PREFETCH_K_DIST < kc {
            _mm_prefetch(b_ptr.add(PREFETCH_K_DIST * NR_AVX512) as *const i8, _MM_HINT_T0);
            _mm_prefetch(a_ptr.add(PREFETCH_K_DIST * MR) as *const i8, _MM_HINT_T0);
        }

        avx512_k_step!(a_ptr, b_ptr);
        a_ptr = a_ptr.add(MR);
        b_ptr = b_ptr.add(NR_AVX512);

        avx512_k_step!(a_ptr, b_ptr);
        a_ptr = a_ptr.add(MR);
        b_ptr = b_ptr.add(NR_AVX512);

        avx512_k_step!(a_ptr, b_ptr);
        a_ptr = a_ptr.add(MR);
        b_ptr = b_ptr.add(NR_AVX512);

        avx512_k_step!(a_ptr, b_ptr);
        a_ptr = a_ptr.add(MR);
        b_ptr = b_ptr.add(NR_AVX512);

        kk += 4;
    }
    while kk < kc {
        if kk + PREFETCH_K_DIST < kc {
            _mm_prefetch(b_ptr.add(PREFETCH_K_DIST * NR_AVX512) as *const i8, _MM_HINT_T0);
            _mm_prefetch(a_ptr.add(PREFETCH_K_DIST * MR) as *const i8, _MM_HINT_T0);
        }
        avx512_k_step!(a_ptr, b_ptr);
        a_ptr = a_ptr.add(MR);
        b_ptr = b_ptr.add(NR_AVX512);
        kk += 1;
    }

    let old0a = _mm512_loadu_ps(row0);
    let old0b = _mm512_loadu_ps(row0.add(16));
    _mm512_storeu_ps(row0, _mm512_add_ps(old0a, c0));
    _mm512_storeu_ps(row0.add(16), _mm512_add_ps(old0b, c6));

    let old1a = _mm512_loadu_ps(row1);
    let old1b = _mm512_loadu_ps(row1.add(16));
    _mm512_storeu_ps(row1, _mm512_add_ps(old1a, c1));
    _mm512_storeu_ps(row1.add(16), _mm512_add_ps(old1b, c7));

    let old2a = _mm512_loadu_ps(row2);
    let old2b = _mm512_loadu_ps(row2.add(16));
    _mm512_storeu_ps(row2, _mm512_add_ps(old2a, c2));
    _mm512_storeu_ps(row2.add(16), _mm512_add_ps(old2b, c8));

    let old3a = _mm512_loadu_ps(row3);
    let old3b = _mm512_loadu_ps(row3.add(16));
    _mm512_storeu_ps(row3, _mm512_add_ps(old3a, c3));
    _mm512_storeu_ps(row3.add(16), _mm512_add_ps(old3b, c9));

    let old4a = _mm512_loadu_ps(row4);
    let old4b = _mm512_loadu_ps(row4.add(16));
    _mm512_storeu_ps(row4, _mm512_add_ps(old4a, c4));
    _mm512_storeu_ps(row4.add(16), _mm512_add_ps(old4b, c10));

    let old5a = _mm512_loadu_ps(row5);
    let old5b = _mm512_loadu_ps(row5.add(16));
    _mm512_storeu_ps(row5, _mm512_add_ps(old5a, c5));
    _mm512_storeu_ps(row5.add(16), _mm512_add_ps(old5b, c11));
}

fn maybe_log_tiling(kernel: KernelKind) {
    if std::env::var_os("FORGE_LOG_TILING").is_none() {
        return;
    }
    let tiles = tile_blocks();
    TILING_LOG_ONCE.call_once(|| {
        eprintln!(
            "[forge-std::matmul] kernel={kernel:?} mc={} kc={} nc={} mr={MR} nr_avx2={NR_AVX2} nr_avx512={NR_AVX512}",
            tiles.mc, tiles.kc, tiles.nc
        );
        eprintln!(
            "[forge-std::matmul] cache arithmetic: A_panel={}KB B_panel={}MB reg_tile={}B",
            (tiles.mc * tiles.kc * 4) / 1024,
            (tiles.kc * tiles.nc * 4) as f64 / (1024.0 * 1024.0),
            MR * kernel.nr() * 4
        );
        let (l2, l3) = detect_cache_sizes();
        if let Some(l2_bytes) = l2 {
            eprintln!("[forge-std::matmul] detected l2={}KB", l2_bytes / 1024);
        }
        if let Some(l3_bytes) = l3 {
            eprintln!(
                "[forge-std::matmul] detected l3={}MB",
                l3_bytes / (1024 * 1024)
            );
        }
    });
}

#[inline(always)]
fn tile_blocks() -> TileConfig {
    *TILE_BLOCKS.get_or_init(|| {
        let mut tiles = autotune_tiles();

        if let Some(mc_env) = parse_env_usize("FORGE_MC") {
            tiles.mc = align_down(mc_env.max(MR), MR).max(MR);
        }
        if let Some(kc_env) = parse_env_usize("FORGE_KC") {
            tiles.kc = kc_env.max(1);
        }
        if let Some(nc_env) = parse_env_usize("FORGE_NC") {
            tiles.nc = align_down(nc_env.max(NR_AVX2), NR_AVX2).max(NR_AVX2);
        }

        tiles
    })
}

fn autotune_tiles() -> TileConfig {
    let (l2, l3) = detect_cache_sizes();
    let l2_bytes = l2.unwrap_or(512 * 1024);
    let l3_bytes =
        l3.unwrap_or(NC_DEFAULT * KC_DEFAULT * std::mem::size_of::<f32>() * 2);

    let kc = if l2_bytes >= 2 * 1024 * 1024 {
        384
    } else if l2_bytes >= 512 * 1024 {
        KC_DEFAULT
    } else {
        128
    };

    let l2_budget = l2_bytes / 2;
    let mc_target = l2_budget / (kc * std::mem::size_of::<f32>());
    let mc_pref = if l2_bytes >= 2 * 1024 * 1024 {
        256
    } else {
        MC_DEFAULT
    };
    let mc = align_down(mc_target.min(mc_pref).max(MR * 4), MR).max(MR);

    let l3_budget = (l3_bytes / 2).max(kc * NR_AVX2 * std::mem::size_of::<f32>());
    let nc_target = l3_budget / (kc * std::mem::size_of::<f32>());
    let nc = align_down(nc_target.clamp(512, 4096), NR_AVX2).max(NR_AVX2);

    TileConfig { mc, kc, nc }
}

fn detect_cache_sizes() -> (Option<usize>, Option<usize>) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__cpuid_count;

        let mut l2 = None;
        let mut l3 = None;
        for leaf in 0..8u32 {
            // SAFETY: `cpuid` is available on x86_64 and this leaf/subleaf pair is query-only.
            let regs = unsafe { __cpuid_count(4, leaf) };
            let cache_type = regs.eax & 0x1F;
            if cache_type == 0 {
                break;
            }

            if cache_type != 1 && cache_type != 3 {
                continue;
            }

            let level = (regs.eax >> 5) & 0x7;
            let line_size = (regs.ebx & 0xFFF) + 1;
            let partitions = ((regs.ebx >> 12) & 0x3FF) + 1;
            let ways = ((regs.ebx >> 22) & 0x3FF) + 1;
            let sets = regs.ecx + 1;
            let size = (line_size as usize)
                .saturating_mul(partitions as usize)
                .saturating_mul(ways as usize)
                .saturating_mul(sets as usize);

            match level {
                2 => l2 = Some(size),
                3 => l3 = Some(size),
                _ => {}
            }
        }

        (l2, l3)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (None, None)
    }
}

#[inline(always)]
fn prefetch_t0(ptr: *const f32) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ptr;
    }
}

#[inline(always)]
const fn align_down(value: usize, align: usize) -> usize {
    (value / align) * align
}

#[inline(always)]
fn parse_env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

#[inline(always)]
const fn div_ceil(x: usize, y: usize) -> usize {
    x.div_ceil(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for kk in 0..k {
                let a_ik = a[i * k + kk];
                let b_row = &b[kk * n..(kk + 1) * n];
                let c_row = &mut out[i * n..(i + 1) * n];
                for j in 0..n {
                    c_row[j] += a_ik * b_row[j];
                }
            }
        }
        out
    }

    fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn test_matmul_4x4() {
        let a = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ];
        let b = vec![
            1.0, 0.0, 2.0, 0.0, //
            0.0, 1.0, 0.0, 2.0, //
            3.0, 0.0, 4.0, 0.0, //
            0.0, 3.0, 0.0, 4.0,
        ];
        let mut c = vec![0.0f32; 16];
        matmul(&a, &b, &mut c, 4, 4, 4);

        let expected = vec![
            10.0, 14.0, 14.0, 20.0, //
            26.0, 30.0, 38.0, 44.0, //
            42.0, 46.0, 62.0, 68.0, //
            58.0, 62.0, 86.0, 92.0,
        ];

        assert!(max_abs_diff(&c, &expected) < 1e-4);
    }

    #[test]
    fn test_matmul_256x256() {
        let n = 256usize;
        let a = (0..n * n)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.03125)
            .collect::<Vec<_>>();
        let b = (0..n * n)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.041)
            .collect::<Vec<_>>();

        let mut c = vec![0.0f32; n * n];
        matmul(&a, &b, &mut c, n, n, n);

        let reference = reference_matmul(&a, &b, n, n, n);
        assert!(max_abs_diff(&c, &reference) < 0.1);
    }

    #[test]
    fn test_matmul_512x512() {
        let n = 512usize;
        let a = (0..n * n)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.0625)
            .collect::<Vec<_>>();
        let b = (0..n * n)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.037)
            .collect::<Vec<_>>();

        let mut c = vec![0.0f32; n * n];
        matmul(&a, &b, &mut c, n, n, n);

        let reference = reference_matmul(&a, &b, n, n, n);
        assert!(max_abs_diff(&c, &reference) < 1.0);
    }

    #[test]
    fn test_matmul_non_square() {
        let (m, n, k) = (48usize, 64usize, 32usize);
        let a = (0..m * k)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.125)
            .collect::<Vec<_>>();
        let b = (0..k * n)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.075)
            .collect::<Vec<_>>();

        let mut c = vec![0.0f32; m * n];
        matmul(&a, &b, &mut c, m, n, k);

        let reference = reference_matmul(&a, &b, m, n, k);
        assert!(max_abs_diff(&c, &reference) < 1e-3);
    }

    #[test]
    fn test_matmul_edge_tiles() {
        let sizes = [7usize, 13, 17, 33, 65, 100, 200];
        for &n in &sizes {
            let a = (0..n * n)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.2)
                .collect::<Vec<_>>();
            let b = (0..n * n)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.3)
                .collect::<Vec<_>>();

            let mut c = vec![0.0f32; n * n];
            matmul(&a, &b, &mut c, n, n, n);

            let reference = reference_matmul(&a, &b, n, n, n);
            let tol = if n <= 65 { 1e-3 } else { 5e-2 };
            assert!(
                max_abs_diff(&c, &reference) < tol,
                "edge tile mismatch at n={n}"
            );
        }
    }

    #[test]
    fn test_avx512_matches_avx2() {
        #[cfg(target_arch = "x86_64")]
        {
            if !(std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("fma")
                && std::is_x86_feature_detected!("avx512f"))
            {
                return;
            }

            let n = 256usize;
            let a = (0..n * n)
                .map(|i| ((i % 41) as f32 - 20.0) * 0.017)
                .collect::<Vec<_>>();
            let b = (0..n * n)
                .map(|i| ((i % 37) as f32 - 18.0) * 0.019)
                .collect::<Vec<_>>();

            let mut c_avx2 = vec![0.0f32; n * n];
            let mut c_avx512 = vec![0.0f32; n * n];

            sgemm_with_kernel(&a, &b, &mut c_avx2, n, n, n, 1.0, KernelKind::Avx2);
            sgemm_with_kernel(&a, &b, &mut c_avx512, n, n, n, 1.0, KernelKind::Avx512);

            assert!(max_abs_diff(&c_avx2, &c_avx512) < 1e-3);
        }
    }

    #[test]
    fn test_threading_correctness() {
        let n = 512usize;
        let a = (0..n * n)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.125)
            .collect::<Vec<_>>();
        let b = (0..n * n)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.0625)
            .collect::<Vec<_>>();

        let pool_1 = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool creation should succeed");
        let pool_4 = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("pool creation should succeed");

        let mut out_1 = vec![0.0f32; n * n];
        let mut out_4 = vec![0.0f32; n * n];

        pool_1.install(|| matmul(&a, &b, &mut out_1, n, n, n));
        pool_4.install(|| matmul(&a, &b, &mut out_4, n, n, n));

        assert!(max_abs_diff(&out_1, &out_4) < 1e-3);
    }
}
