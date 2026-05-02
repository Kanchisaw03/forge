import numpy as np
import time

for N in [256, 512, 1024]:
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    for _ in range(5):  # warmup
        C = A @ B

    times = []
    for _ in range(20):  # timed
        t0 = time.perf_counter()
        C = A @ B
        times.append(time.perf_counter() - t0)

    ms = sum(times) / len(times) * 1000
    gflops = (2 * N ** 3) / (sum(times) / len(times) * 1e9)
    print(f"{N}×{N}: {ms:.2f}ms | {gflops:.1f} GFLOPS")
