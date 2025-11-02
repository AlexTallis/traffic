# app.py
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ------------- Page setup / styles -------------
st.set_page_config(page_title="Velocity–Time Water-Filling", layout="wide")
st.markdown("""
    <style>
      .main { padding-top: 1rem; }
      .stSlider > div > div { padding-top: 0.2rem; }
      .metric-row { display: flex; gap: 1rem; }
      .metric-box {
          background: #0e1117; border: 1px solid #2b2f36; border-radius: 12px;
          padding: 10px 14px; flex: 1;
      }
      .metric-title { color: #c7cbd1; font-size: 0.8rem; }
      .metric-value { color: white; font-weight: 700; font-size: 1.1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("Velocity–Time ‘Tank’ — Water-Filling with Level Lines of Slope −α")

# ------------- Sidebar controls -------------
with st.sidebar:
    st.header("Controls")
    vmax = st.slider("vmax", 0.5, 50.0, 10.0, 0.1)
    v0   = st.slider("v0 (must be < vmax)", 0.0, max(0.0, vmax - 1e-3), 4.0, 0.1)
    q    = st.slider("q (time wall)", 1.0, 50.0, 12.0, 0.1)
    alpha= st.slider("alpha (accel slope)", 0.1, 10.0, 1.0, 0.1)
    beta = st.slider("beta (decel slope)", 0.1, 10.0, 2.0, 0.1)
    d    = st.slider("d (filled area target)", 0.0, 300.0, 15.0, 0.1)

# ------------- Core helpers (same physics as your last version) -------------
EPS = 1e-9

def accel_segment(v0, alpha, vmax, q):
    """Accel wall (orange) from (0,v0) to hitting vmax, clipped by q."""
    if alpha <= 0: return np.array([]), np.array([])
    t_hit = (vmax - v0) / alpha
    t_end = max(0.0, min(q, t_hit))
    if t_end <= 0: return np.array([]), np.array([])
    t = np.linspace(0.0, t_end, 400)
    v = np.clip(v0 + alpha * t, 0.0, vmax)
    return t, v

def decel_segment(v0, beta, vmax, q):
    """Decel wall (green) from (0,v0) to 0, clipped by q."""
    if beta <= 0: return np.array([]), np.array([])
    t_hit = v0 / beta
    t_end = max(0.0, min(q, t_hit))
    if t_end <= 0: return np.array([]), np.array([])
    t = np.linspace(0.0, t_end, 400)
    v = np.clip(v0 - beta * t, 0.0, vmax)
    return t, v

def t_left_of_tank(v, v0, alpha, beta, vmax):
    """
    For each v, minimal allowed time inside the tank:
      t_left(v) = max(0, t_acc(v), t_dec(v))
    with t_acc valid on [v0, vmax], t_dec on [0, v0].
    """
    t_acc = np.full_like(v, -np.inf, dtype=float)
    t_dec = np.full_like(v, -np.inf, dtype=float)
    if alpha > 0:
        mask_a = (v >= v0) & (v <= vmax)
        t_acc[mask_a] = (v[mask_a] - v0) / alpha
    if beta > 0:
        mask_d = (v >= 0) & (v <= v0)
        t_dec[mask_d] = (v0 - v[mask_d]) / beta
    return np.maximum(0.0, np.maximum(t_acc, t_dec))

def capacity_and_target(d_target, v0, vmax, q, alpha, beta, V):
    t_left = t_left_of_tank(V, v0, alpha, beta, vmax)
    cap = np.trapz(np.maximum(0.0, q - t_left), V)
    d_star = np.clip(d_target, 0.0, cap + 1e-12)
    return cap, d_star, t_left

def filled_area_for_s(s, V, t_left, q, alpha):
    t_water = np.minimum(q, s - V/alpha)
    width   = np.maximum(0.0, t_water - t_left)
    return np.trapz(width, V)

def solve_s_for_area(d_star, V, t_left, q, alpha):
    if d_star <= 1e-12:
        return -1e9  # effectively empty
    span = q + (V.max() if V.size else 0.0) / max(alpha, 1e-6) + 5.0
    s_lo, s_hi = -span, span + q

    def A(s): return filled_area_for_s(s, V, t_left, q, alpha)

    A_lo, A_hi = A(s_lo), A(s_hi)
    # Expand until [s_lo,s_hi] brackets d_star
    tries = 0
    while A_lo > d_star + 1e-9 and tries < 20:
        s_hi = s_lo
        s_lo -= span
        A_lo = A(s_lo)
        tries += 1
    tries = 0
    while A_hi < d_star - 1e-9 and tries < 20:
        s_lo = s_hi
        s_hi += span
        A_hi = A(s_hi)
        tries += 1

    for _ in range(60):
        sm = 0.5 * (s_lo + s_hi)
        Am = A(sm)
        if Am < d_star:
            s_lo = sm
        else:
            s_hi = sm
    return 0.5 * (s_lo + s_hi)

# ------------- Build curves and fill -------------
# Walls
ta, va = accel_segment(v0, alpha, vmax, q)
td, vd = decel_segment(v0, beta,  vmax, q)

# Discretise vertical levels for the tank
V = np.linspace(0.0, vmax, 1600)
capacity, d_star, t_left = capacity_and_target(d, v0, vmax, q, alpha, beta, V)

# Solve for water level intercept s so area ≈ d_star
s_star = solve_s_for_area(d_star, V, t_left, q, alpha)

# Water line: only show where it's to the RIGHT of walls and within [0, q]
V_line = np.linspace(0.0, vmax, 1200)
t_left_interp = np.interp(V_line, V, t_left)
t_line = s_star - V_line / alpha
mask_line = (t_line >= t_left_interp + 1e-12) & (t_line >= 0.0) & (t_line <= q)
t_line_plot = t_line[mask_line]
V_line_plot = V_line[mask_line]

# Shaded region: polygon to the RIGHT of walls, up to t_water = min(q, s - v/alpha)
t_water = np.minimum(q, s_star - V/alpha)
width   = np.maximum(0.0, t_water - t_left)
mask_fill = width > 1e-12
V_fill = V[mask_fill]
x_left = t_left[mask_fill]
x_right= (t_left + width)[mask_fill]

# Polygon path (x then back along the other boundary at same V)
poly_x = np.concatenate([x_left, x_right[::-1]])
poly_y = np.concatenate([V_fill, V_fill[::-1]])

# ------------- Plotly figure -------------
x_pad = max(0.1 * max(q, 1.0), 1.0)
x_max = q + x_pad
y_max = max(1.0, vmax, v0) * 1.2

fig = go.Figure()

# Shaded water polygon
if poly_x.size >= 3:
    fig.add_trace(go.Scatter(
        x=poly_x, y=poly_y,
        fill='toself', mode='lines', line=dict(width=0),
        fillcolor='rgba(31,119,180,0.25)',
        name='filled water'
    ))

# Accel (orange) and decel (green) walls
if ta.size:
    fig.add_trace(go.Scatter(x=ta, y=va, mode='lines',
                             line=dict(width=3, color='#ff7f0e'),
                             name='accel to vmax'))
if td.size:
    fig.add_trace(go.Scatter(x=td, y=vd, mode='lines',
                             line=dict(width=3, color='#2ca02c'),
                             name='decel to 0'))

# Water level line (blue), only where right of walls
if t_line_plot.size:
    fig.add_trace(go.Scatter(x=t_line_plot, y=V_line_plot, mode='lines',
                             line=dict(width=3, color='#1f77b4'),
                             name='water level (slope = -α)'))

# vmax dashed guide (illustrative)
fig.add_trace(go.Scatter(
    x=[0, x_max], y=[vmax, vmax],
    mode='lines', line=dict(width=2, dash='dash', color='#9aa0a6'),
    name='vmax (guide)'
))
# q vertical guide + label
fig.add_trace(go.Scatter(
    x=[q, q], y=[0, y_max],
    mode='lines', line=dict(width=2, dash='dash', color='#9aa0a6'),
    name='q (guide)', showlegend=False
))
fig.add_annotation(x=q, y=0, text="q", showarrow=False, yshift=-12)

# v0 marker + label on y-axis
fig.add_trace(go.Scatter(
    x=[0], y=[v0], mode='markers',
    marker=dict(size=8, color='white'), name='v₀', showlegend=False
))
fig.add_annotation(x=0, y=v0, text="v₀", xanchor='right', yanchor='middle', showarrow=False, xshift=-16)

# Layout polish
fig.update_layout(
    template="plotly_dark",
    margin=dict(l=30, r=20, t=50, b=40),
    xaxis=dict(title="t", range=[0, x_max], zeroline=False, showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
    yaxis=dict(title="v", range=[0, y_max], zeroline=False, showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, bgcolor="rgba(0,0,0,0)")
)

# ------------- Metrics row -------------
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.markdown('<div class="metric-box"><div class="metric-title">Target d</div>'
                f'<div class="metric-value">{d:.3f}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-box"><div class="metric-title">Clamped d*</div>'
                f'<div class="metric-value">{min(d, capacity):.3f}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box"><div class="metric-title">Tank capacity</div>'
                f'<div class="metric-value">{capacity:.3f}</div></div>', unsafe_allow_html=True)

# ------------- Plot -------------
st.plotly_chart(fig, use_container_width=True)

# ------------- Footnote -------------
st.caption(
    "Shaded region is always to the **right** of the orange/green walls and below the blue level line, "
    "bounded by the dashed guides at **q** and **vmax**. The blue −α line is only drawn where it lies to the right of the walls."
)
