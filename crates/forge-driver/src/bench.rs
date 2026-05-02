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
    /// Median latency.
    pub p50_ms: f64,
    /// 95th percentile latency.
    pub p95_ms: f64,
    /// 99th percentile latency.
    pub p99_ms: f64,
    /// Throughput in GFLOP/s.
    pub gflops: f64,
    /// Effective memory throughput in GB/s.
    pub gb_per_s: f64,
}

/// Prints runtime diagnostics for SIMD dispatch and threading.
pub fn print_diagnostics() {
    let before = rayon::current_num_threads();
    let after = forge_runtime::init_rayon_global_pool();

    println!("=== FORGE RUNTIME DIAGNOSTICS ===");
    println!(
        "AVX2:             {}",
        std::is_x86_feature_detected!("avx2")
    );
    println!("FMA:              {}", std::is_x86_feature_detected!("fma"));
    println!(
        "AVX-512F:         {}",
        std::is_x86_feature_detected!("avx512f")
    );
    println!("Rayon threads:    {}", after);
    println!("Rayon before init: {}", before);
    println!("Physical cores:   {}", forge_runtime::physical_cpu_count());
    println!("Logical cores:    {}", num_cpus::get());
    println!("=================================");
}

/// Runs SGEMM benchmark for square `n x n`.
pub fn run_matmul_bench(config: BenchConfig, n: usize) -> BenchResult {
    let _ = forge_runtime::init_rayon_global_pool();

    let a = (0..n * n).map(|i| (i as f32).sin()).collect::<Vec<_>>();
    let b = (0..n * n).map(|i| (i as f32).cos()).collect::<Vec<_>>();
    let mut c = vec![0.0f32; n * n];

    for _ in 0..config.warmup {
        c.fill(0.0);
        forge_std::matmul(&a, &b, &mut c, n, n, n);
    }

    let mut samples_ms = Vec::with_capacity(config.iters);
    for _ in 0..config.iters {
        c.fill(0.0);
        let start = Instant::now();
        forge_std::matmul(&a, &b, &mut c, n, n, n);
        samples_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
    }

    samples_ms.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).expect("finite sample"));

    let mean_ms = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
    let p50_ms = percentile(&samples_ms, 0.50);
    let p95_ms = percentile(&samples_ms, 0.95);
    let p99_ms = percentile(&samples_ms, 0.99);

    let flops = 2.0 * (n as f64).powi(3);
    let gflops = flops / (mean_ms * 1e-3 * 1e9);

    let bytes = 16.0 * (n as f64).powi(2);
    let gb_per_s = bytes / (mean_ms * 1e-3 * 1e9);

    BenchResult {
        mean_ms,
        p50_ms,
        p95_ms,
        p99_ms,
        gflops,
        gb_per_s,
    }
}

fn percentile(sorted_samples: &[f64], p: f64) -> f64 {
    let len = sorted_samples.len();
    if len == 0 {
        return 0.0;
    }
    let idx = ((len as f64 - 1.0) * p).round() as usize;
    sorted_samples[idx.min(len - 1)]
}
