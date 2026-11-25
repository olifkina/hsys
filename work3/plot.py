import json
import os
import plotly.graph_objects as go

base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir, "bench_results.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

naive, shmem = {}, {}
for b in data["benchmarks"]:
    n = int(b["name"].split("/")[1])
    if b["name"].startswith("BM_CudaMatMulNaive"):
        naive[n] = b["real_time"]
    elif b["name"].startswith("BM_CudaMatMulShmem"):
        shmem[n] = b["real_time"]

ns = sorted(shmem.keys())
t_naive = [naive[n] for n in ns]
t_shmem = [shmem[n] for n in ns]
speedup = [naive[n] / shmem[n] for n in ns]

print(f"{'N':>6} | {'Naive (ms)':>10} | {'Shmem (ms)':>10} | {'Speedup':>8} | {'Naive GFLOP/s':>13} | {'Shmem GFLOP/s':>13}")
print("-" * 76)
for n, tn, ts, sp in zip(ns, t_naive, t_shmem, speedup):
    flops = 2.0 * (n ** 3)
    gn = flops / (tn * 1e-3) / 1e9
    gs = flops / (ts * 1e-3) / 1e9
    print(f"{n:>6} | {tn:>10.5f} | {ts:>10.5f} | {sp:>8.3f} | {gn:>13.2f} | {gs:>13.2f}")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=ns, y=t_naive, mode="lines+markers", name="CUDA Matrix Multiplication (Naive, float)"))
fig1.add_trace(go.Scatter(x=ns, y=t_shmem, mode="lines+markers", name="CUDA Matrix Multiplication (Shared Memory, float)"))
fig1.update_layout(
    title="Real Complexity",
    xaxis_title="N", yaxis_title="Time, ms",
    xaxis_type="log", yaxis_type="log"
)
fig1.show()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ns, y=speedup, mode="lines+markers", name="Speedup (Shmem vs Naive)"))
fig2.update_layout(
    title="Speedup: CUDA Shared Memory vs CUDA Naive",
    xaxis_title="N", yaxis_title="Speedup",
    xaxis_type="log"
)
fig2.show()
