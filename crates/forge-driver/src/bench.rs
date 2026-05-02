use std::fs;
use std::time::Instant;

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
    let _ = forge_runtime::init_rayon_global_pool();

    let avx2 = std::is_x86_feature_detected!("avx2");
    let fma = std::is_x86_feature_detected!("fma");
    let avx512 = std::is_x86_feature_detected!("avx512f");

    println!("--------------------------------------------");
    println!("Forge SGEMM Benchmark");
    println!("CPU:     {}", cpu_model_name());
    println!("Threads: {}", rayon::current_num_threads());
    println!("AVX2:    {}", avx2);
    println!("FMA:     {}", fma);
    println!("AVX512:  {}", avx512);
    println!("--------------------------------------------");
    forge_runtime::print_thread_diagnostics();
}

/// Runs SGEMM benchmark for square `n x n`.
pub fn run_matmul_bench(config: BenchConfig, n: usize) -> BenchResult {
    let _ = forge_runtime::init_rayon_global_pool();
    let repeat = benchmark_repeat_count(n);

    let a = (0..n * n).map(|i| (i as f32).sin()).collect::<Vec<_>>();
    let b = (0..n * n).map(|i| (i as f32).cos()).collect::<Vec<_>>();
    let mut c = vec![0.0f32; n * n];

    for _ in 0..config.warmup {
        for _ in 0..repeat {
            c.fill(0.0);
            forge_std::matmul(&a, &b, &mut c, n, n, n);
        }
    }

    let mut samples_ms = Vec::with_capacity(config.iters);
    for _ in 0..config.iters {
        let start = Instant::now();
        for _ in 0..repeat {
            c.fill(0.0);
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

fn cpu_model_name() -> String {
    let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") else {
        return "unknown".to_owned();
    };

    cpuinfo
        .lines()
        .find_map(|line| line.strip_prefix("model name\t: "))
        .map_or_else(|| "unknown".to_owned(), ToOwned::to_owned)
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
