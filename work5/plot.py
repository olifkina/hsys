import json
import os
import plotly.graph_objects as go

base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir, "bench_results.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

nobr, br = {}, {}
for b in data["benchmarks"]:
    n = int(b["name"].split("/")[1])
    if b["name"].startswith("BM_CudaVecRedNoBr"):
        nobr[n] = b["real_time"]
    elif b["name"].startswith("BM_CudaVecRedBr"):
        br[n] = b["real_time"]

ns = sorted(br.keys())
t_nobr = [nobr[n] for n in ns]
t_br = [br[n] for n in ns]
speedup = [nobr[n] / br[n] for n in ns]

print(f"{'N':>12} | {'NoBr (ms)':>10} | {'Br (ms)':>10} | {'Speedup':>8} | {'NoBr GB/s':>10} | {'Br GB/s':>10}")
print("-" * 76)
for n, t1, t2, sp in zip(ns, t_nobr, t_br, speedup):
    bytes_read = n * 4.0
    bw1 = bytes_read / (t1 * 1e-3) / 1e9
    bw2 = bytes_read / (t2 * 1e-3) / 1e9
    print(f"{n:>12} | {t1:>10.5f} | {t2:>10.5f} | {sp:>8.3f} | {bw1:>10.2f} | {bw2:>10.2f}")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=ns, y=t_nobr, mode="lines+markers", name="CUDA Reduction (Shared Memory, NoBr)"))
fig1.add_trace(go.Scatter(x=ns, y=t_br, mode="lines+markers", name="CUDA Reduction (Warp Shuffle, Br)"))
fig1.update_layout(
    title="Real Complexity",
    xaxis_title="N", yaxis_title="Time, ms",
    xaxis_type="log", yaxis_type="log"
)
fig1.show()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ns, y=speedup, mode="lines+markers", name="Speedup (Br vs NoBr)"))
fig2.update_layout(
    title="Speedup: CUDA Warp Shuffle (Br) vs Shared Memory (NoBr)",
    xaxis_title="N", yaxis_title="Speedup",
    xaxis_type="log"
)
fig2.show()
