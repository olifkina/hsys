import json
import os
import plotly.graph_objects as go

base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir, "bench_results.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

shmem, wmma = {}, {}
for b in data["benchmarks"]:
    n = int(b["name"].split("/")[1])
    if b["name"].startswith("BM_CudaMatMulShmem"):
        shmem[n] = b["real_time"]
    elif b["name"].startswith("BM_CudaMatMulWmma"):
        wmma[n] = b["real_time"]

ns = sorted(wmma.keys())
t_shmem = [shmem[n] for n in ns]
t_wmma = [wmma[n] for n in ns]
speedup = [shmem[n] / wmma[n] for n in ns]

print(f"{'N':>6} | {'Shmem (ms)':>10} | {'WMMA (ms)':>10} | {'Speedup':>8} | {'Shmem GFLOP/s':>13} | {'WMMA GFLOP/s':>12}")
print("-" * 75)
for n, ts, tw, sp in zip(ns, t_shmem, t_wmma, speedup):
    flops = 2.0 * (n ** 3)
    gs = flops / (ts * 1e-3) / 1e9
    gw = flops / (tw * 1e-3) / 1e9
    print(f"{n:>6} | {ts:>10.5f} | {tw:>10.5f} | {sp:>8.3f} | {gs:>13.2f} | {gw:>12.2f}")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=ns, y=t_shmem, mode="lines+markers", name="CUDA Matrix Multiplication (Shared Memory, half)"))
fig1.add_trace(go.Scatter(x=ns, y=t_wmma, mode="lines+markers", name="CUDA Matrix Multiplication (WMMA Tensor Cores, half)"))
fig1.update_layout(
    title="Real Complexity",
    xaxis_title="N", yaxis_title="Time, ms",
    xaxis_type="log", yaxis_type="log"
)
fig1.show()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ns, y=speedup, mode="lines+markers", name="Speedup (WMMA vs Shmem)"))
fig2.update_layout(
    title="Speedup: CUDA WMMA vs CUDA Shared Memory",
    xaxis_title="N", yaxis_title="Speedup",
    xaxis_type="log"
)
fig2.show()
