import numpy as np
import time

sizes = [256, 512, 1024]
warmup = 5
iters = 20

print("NumPy SGEMM on this machine:")
for n in sizes:
    a = (np.arange(n * n, dtype=np.float32) % 17) * 0.1
    b = (np.arange(n * n, dtype=np.float32) % 11) * 0.2
    a = a.reshape((n, n))
    b = b.reshape((n, n))

    for _ in range(warmup):
        np.matmul(a, b)

    latencies = []
    for _ in range(iters):
        start = time.perf_counter()
        np.matmul(a, b)
        latencies.append(time.perf_counter() - start)

    mean_s = sum(latencies) / len(latencies)
    gflops = (2.0 * n**3) / (mean_s * 1e9)
    print(f"  {n}x{n}: {gflops:.1f} GFLOPS  ({mean_s*1000:.2f} ms)")
