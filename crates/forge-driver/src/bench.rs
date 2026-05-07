use std::sync::Once;
use std::time::{Duration, Instant};

const BANNER_BORDER: &str = "────────────────────────────────────────────────────────────";

static AVX512_FREQ_WARMUP_ONCE: Once = Once::new();

/// Benchmark options.
#[derive(Clone, Copy, Debug)]
pub struct BenchConfig {
    /// Warmup iterations.
    pub warmup: usize,
    /// Timed iterations.
    pub iters: usize,
}

/// Bench result metrics.
#[derive(Clone, Copy, Debug)]
pub struct BenchResult {
    /// Arithmetic mean latency across timed iterations.
    pub mean_ms: f64,
    /// Throughput in GFLOP/s.
    pub gflops: f64,
}

/// Prints runtime diagnostics for SIMD dispatch and threading.
pub fn print_benchmark_header() {
    let rayon_threads = forge_runtime::init_rayon_global_pool();
    let physical = forge_runtime::physical_cpu_count();
    let logical = num_cpus::get().max(1);

    let avx2 = std::is_x86_feature_detected!("avx2");
    let fma = std::is_x86_feature_detected!("fma");
    let avx512 = std::is_x86_feature_detected!("avx512f");

    println!("{BANNER_BORDER}");
    println!();
    println!("  Forge SGEMM Benchmark");
    println!("  CPU:     {}", cpu_model_name());
    println!(
        "  Threads: {} (Rayon) / {} physical / {} logical",
        rayon_threads, physical, logical
    );
    println!("  AVX2:    {}    FMA: {}    AVX512: {}", avx2, fma, avx512);
    println!("{BANNER_BORDER}");
    println!();
}

/// Runs SGEMM benchmark for square `n x n`.
pub fn run_matmul_bench(config: BenchConfig, n: usize) -> BenchResult {
    let _ = forge_runtime::init_rayon_global_pool();
    AVX512_FREQ_WARMUP_ONCE.call_once(stabilize_avx512_frequency);
    let repeat = benchmark_repeat_count(n);

    let a = (0..n * n).map(|i| (i as f32).sin()).collect::<Vec<_>>();
    let b = (0..n * n).map(|i| (i as f32).cos()).collect::<Vec<_>>();
    let mut c = vec![0.0f32; n * n];

    for _ in 0..config.warmup {
        for _ in 0..repeat {
            forge_std::matmul(&a, &b, &mut c, n, n, n);
        }
    }

    let mut samples_ms = Vec::with_capacity(config.iters);
    for _ in 0..config.iters {
        let start = Instant::now();
        for _ in 0..repeat {
            forge_std::matmul(&a, &b, &mut c, n, n, n);
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0 / repeat as f64;
        samples_ms.push(elapsed_ms);
    }

    let mean_ms = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
    let flops = 2.0 * (n as f64).powi(3);
    let gflops = flops / (mean_ms * 1e-3 * 1e9);

    BenchResult { mean_ms, gflops }
}

pub fn theoretical_peak_gflops() -> f64 {
    let cores = num_cpus::get_physical().max(1) as f64;
    let base_freq_ghz = 3.0f64;
    let lanes = if std::is_x86_feature_detected!("avx512f") {
        16.0
    } else if std::is_x86_feature_detected!("avx2") {
        8.0
    } else {
        4.0
    };
    let fma_factor = if std::is_x86_feature_detected!("fma") {
        2.0
    } else {
        1.0
    };
    lanes * fma_factor * base_freq_ghz * cores
}

/// Stabilize AVX-512 frequency before timed runs.
///
/// Intel CPUs can downclock when AVX-512 first starts executing.
/// This warmup triggers the frequency transition before measurement.
/// On client chips (<=4 cores) AVX-512 usually hurts more than it helps,
/// so we skip the warmup to avoid throttling the AVX2 benchmark.
fn stabilize_avx512_frequency() {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }

        // Client chips (<=4 cores) rarely benefit from AVX-512 and the
        // warmup actively throttles AVX2 performance. Use AVX2 warmup instead
        // to bring the CPU to boost frequency without triggering AVX-512 throttling.
        let cores = num_cpus::get_physical();
        if cores <= 4 {
            println!("Warming up CPU frequency with AVX2 ({cores} cores)...");
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                // SAFETY: guarded by runtime feature detection.
                unsafe { avx2_frequency_warmup_loop() };
            }
            std::thread::sleep(Duration::from_millis(30));
            println!("Done. Starting benchmark.");
            return;
        }

        println!("Stabilizing CPU frequency (300ms)...");

        // SAFETY: guarded by runtime feature detection.
        unsafe { avx512_frequency_warmup_loop() };

        std::thread::sleep(Duration::from_millis(50));
        println!("Done. Starting benchmark.");
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn avx512_frequency_warmup_loop() {
    use std::arch::x86_64::*;

    let deadline = Instant::now() + Duration::from_millis(300);
    let mut acc = _mm512_setzero_ps();
    let ones = _mm512_set1_ps(1.001f32);

    while Instant::now() < deadline {
        for _ in 0..500 {
            acc = _mm512_fmadd_ps(acc, ones, ones);
        }
    }

    // Keep the work observable to avoid dead-code elimination.
    let mut sink = [0.0f32; 16];
    _mm512_storeu_ps(sink.as_mut_ptr(), acc);
    std::hint::black_box(sink);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_frequency_warmup_loop() {
    use std::arch::x86_64::*;

    let deadline = Instant::now() + Duration::from_millis(200);
    let mut acc = _mm256_setzero_ps();
    let ones = _mm256_set1_ps(1.001f32);

    while Instant::now() < deadline {
        for _ in 0..500 {
            acc = _mm256_fmadd_ps(acc, ones, ones);
        }
    }

    let mut sink = [0.0f32; 8];
    _mm256_storeu_ps(sink.as_mut_ptr(), acc);
    std::hint::black_box(sink);
}

fn cpu_model_name() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(name) = cpu_brand_from_cpuid() {
            return name;
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            if let Some(name) = cpuinfo
                .lines()
                .find_map(|line| line.strip_prefix("model name\t: "))
            {
                return clean_cpu_brand(name);
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Ok(identifier) = std::env::var("PROCESSOR_IDENTIFIER") {
            let name = clean_cpu_brand(&identifier);
            if !name.is_empty() {
                return name;
            }
        }
    }

    "unknown".to_owned()
}

#[cfg(target_arch = "x86_64")]
fn cpu_brand_from_cpuid() -> Option<String> {
    use std::arch::x86_64::__cpuid;

    let max_extended_leaf = unsafe { __cpuid(0x8000_0000) }.eax;
    if max_extended_leaf < 0x8000_0004 {
        return None;
    }

    let mut brand = Vec::with_capacity(48);
    for leaf in 0x8000_0002..=0x8000_0004 {
        let regs = unsafe { __cpuid(leaf) };
        brand.extend_from_slice(&regs.eax.to_le_bytes());
        brand.extend_from_slice(&regs.ebx.to_le_bytes());
        brand.extend_from_slice(&regs.ecx.to_le_bytes());
        brand.extend_from_slice(&regs.edx.to_le_bytes());
    }

    let raw = String::from_utf8_lossy(&brand).trim_matches(char::from(0)).trim().to_owned();
    if raw.is_empty() {
        None
    } else {
        Some(clean_cpu_brand(&raw))
    }
}

fn clean_cpu_brand(raw: &str) -> String {
    raw.replace("(R)", "")
        .replace("(TM)", "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[inline(always)]
fn benchmark_repeat_count(n: usize) -> usize {
    if n <= 256 {
        8
    } else if n <= 512 {
        4
    } else {
        1
    }
}
