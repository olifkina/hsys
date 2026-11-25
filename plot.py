import plotly.graph_objects as go

ns = [8, 64, 512, 4096, 32768, 262144, 2097152, 16777216]
t_cuda = [0.0052293, 0.0066768, 0.0092061, 0.0100475, 0.0113564, 0.0319786, 0.1010715, 0.7515084]
t_eigen = [0.0000580, 0.0000701, 0.0001509, 0.0008344, 0.0065134, 0.1759314, 1.6928334, 21.0836189]
speedup = [e / c for e, c in zip(t_eigen, t_cuda)]

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=ns, y=t_eigen, mode="lines+markers", name="Eigen Vector Addition (CPU)"))
fig1.add_trace(go.Scatter(x=ns, y=t_cuda, mode="lines+markers", name="CUDA Vector Addition (GPU)"))
fig1.update_layout(title="Real Complexity", xaxis_title="N", yaxis_title="Time, ms", xaxis_type="log", yaxis_type="log")
fig1.show()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ns, y=speedup, mode="lines+markers", name="Speedup"))
fig2.update_layout(title="Speedup: CUDA (GPU) vs Eigen (CPU)", xaxis_title="N", yaxis_title="Speedup", xaxis_type="log", yaxis_type="log")
fig2.show()