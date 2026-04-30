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
    /// Average latency in milliseconds.
    pub latency_ms: f64,
    /// Throughput in GFLOP/s.
    pub gflops: f64,
}

/// Runs built-in SGEMM benchmark.
pub fn run_matmul_bench(config: BenchConfig, n: usize) -> BenchResult {
    let a = (0..n * n).map(|i| (i % 17) as f32 * 0.1).collect::<Vec<_>>();
    let b = (0..n * n).map(|i| (i % 11) as f32 * 0.2).collect::<Vec<_>>();
    let mut c = vec![0.0f32; n * n];

    for _ in 0..config.warmup {
        forge_std::matmul(&a, &b, &mut c, n, n, n);
    }

    let start = Instant::now();
    for _ in 0..config.iters {
        forge_std::matmul(&a, &b, &mut c, n, n, n);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let latency_ms = elapsed * 1000.0 / config.iters as f64;
    let flops = 2.0 * (n as f64).powi(3);
    let gflops = (flops * config.iters as f64) / elapsed / 1e9;
    BenchResult { latency_ms, gflops }
}
