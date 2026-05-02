use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use forge_std::matmul;

fn bench_sgemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("sgemm");
    for &n in &[256usize, 512, 1024] {
        let a = (0..n * n).map(|i| (i % 17) as f32 * 0.1).collect::<Vec<_>>();
        let b = (0..n * n).map(|i| (i % 11) as f32 * 0.2).collect::<Vec<_>>();
        let mut out = vec![0.0f32; n * n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                matmul(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut out),
                    n,
                    n,
                    n,
                );
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sgemm);
criterion_main!(benches);
