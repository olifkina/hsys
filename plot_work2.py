import json
import plotly.graph_objects as go

with open("bench_results_work2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

cuda, eigen = {}, {}
for b in data["benchmarks"]:
    n = int(b["name"].split("/")[1])
    if b["name"].startswith("BM_CudaMatMulNaive"):
        cuda[n] = b["real_time"]
    elif b["name"].startswith("BM_EigenMatMul"):
        eigen[n] = b["real_time"]

ns = sorted(cuda.keys())
t_cuda = [cuda[n] for n in ns]
t_eigen = [eigen[n] for n in ns]
speedup = [eigen[n] / cuda[n] for n in ns]

print(f"{'N':>6} | {'CUDA (ms)':>10} | {'Eigen (ms)':>10} | {'Speedup':>8} | {'GPU GFLOP/s':>11} | {'CPU GFLOP/s':>11}")
print("-" * 70)
for n, tc, te, sp in zip(ns, t_cuda, t_eigen, speedup):
    flops = 2.0 * (n ** 3)
    gflops_gpu = flops / (tc * 1e-3) / 1e9
    gflops_cpu = flops / (te * 1e-3) / 1e9
    print(f"{n:>6} | {tc:>10.5f} | {te:>10.5f} | {sp:>8.3f} | {gflops_gpu:>11.2f} | {gflops_cpu:>11.2f}")

# График 1: Real Complexity
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=ns, y=t_eigen, mode="lines+markers", name="Eigen Matrix Multiplication (CPU, float)"))
fig1.add_trace(go.Scatter(x=ns, y=t_cuda, mode="lines+markers", name="CUDA Matrix Multiplication (GPU, Naive, float)"))
fig1.update_layout(
    title="Real Complexity",
    xaxis_title="N", yaxis_title="Time, ms",
    xaxis_type="log", yaxis_type="log"
)
fig1.show()

# График 2: Speedup
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ns, y=speedup, mode="lines+markers", name="Speedup"))
fig2.update_layout(
    title="Speedup: CUDA Matrix Multiplication (GPU, Naive, float) vs Eigen Matrix Multiplication (CPU, float)",
    xaxis_title="N", yaxis_title="Speedup",
    xaxis_type="log", yaxis_type="log"
)
fig2.show()