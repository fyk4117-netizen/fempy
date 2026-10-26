# TRUSS.EXE - step-by-step FEM tutor for 2D/3D bar (truss) elements
# Pure Python: numpy for the math, streamlit for the screen. No AI at runtime.
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt

plt.style.use("dark_background")
st.set_page_config(page_title="TRUSS.EXE", layout="wide")
st.markdown("<style>*{font-family:'Courier New',monospace !important} .stApp{border:3px double #33ff66}</style>",
            unsafe_allow_html=True)

NC = ["x", "y", "z", "fix_x", "fix_y", "fix_z", "Fx_kN", "Fy_kN", "Fz_kN"]
EC = ["node_i", "node_j", "A_m2", "E_GPa"]
F_, T_ = False, True
PRESETS = {
    "EX 3.9 (Logan) - 3D space truss": dict(dim=3,
        nodes=[[12, -3, -4, F_, F_, F_, 20, 0, 0], [0, 0, 0, T_, T_, T_, 0, 0, 0],
               [12, -3, -7, T_, T_, T_, 0, 0, 0], [14, 6, 0, T_, T_, T_, 0, 0, 0]],
        elems=[[1, 2, 0.001, 210], [1, 3, 0.001, 210], [1, 4, 0.001, 210]]),
    "2D three-bar truss": dict(dim=2,
        nodes=[[0, 0, 0, T_, T_, F_, 0, 0, 0], [4, 0, 0, F_, T_, F_, 0, 0, 0], [2, 3, 0, F_, F_, F_, 10, -20, 0]],
        elems=[[1, 2, 0.001, 200], [1, 3, 0.001, 200], [2, 3, 0.001, 200]]),
    "2D seven-bar bridge": dict(dim=2,
        nodes=[[0, 0, 0, T_, T_, F_, 0, 0, 0], [3, 0, 0, F_, F_, F_, 0, -30, 0], [6, 0, 0, F_, T_, F_, 0, 0, 0],
               [1.5, 2, 0, F_, F_, F_, 0, 0, 0], [4.5, 2, 0, F_, F_, F_, 5, 0, 0]],
        elems=[[1, 2, 0.002, 200], [2, 3, 0.002, 200], [1, 4, 0.002, 200], [4, 2, 0.002, 200],
               [2, 5, 0.002, 200], [5, 3, 0.002, 200], [4, 5, 0.002, 200]]),
}
STEPS = ["1 PROBLEM", "2 ELEMENT [k]", "3 ASSEMBLY [K]", "4 BOUNDARY CONDITIONS", "5 DISPLACEMENTS", "6 REACTIONS + STRESS"]


def show(M, rows, cols, unit="", hl=None):
    """Print a matrix with a common power-of-ten factored out, like a textbook."""
    M = np.atleast_2d(M)
    m = np.abs(M).max()
    p = int(np.floor(np.log10(m))) if m > 0 else 0
    df = pd.DataFrame(M / 10.0**p, index=rows, columns=cols)
    sty = df.style.format("{:.4f}")
    if hl is not None:
        css = pd.DataFrame("", index=df.index, columns=df.columns)
        css.iloc[hl, hl] = "background-color:#225522"
        sty = sty.apply(lambda _: css, axis=None)
    st.caption(f"values x 10^{p} {unit}")
    st.dataframe(sty)


def gauss(A, b):
    """Gaussian elimination with partial pivoting, logging every row operation."""
    M, n, log = np.hstack([A, b[:, None]]).astype(float), len(b), []
    for k in range(n):
        p = k + np.argmax(abs(M[k:, k]))
        if p != k:
            M[[k, p]] = M[[p, k]]
            log.append((f"swap R{k+1} <-> R{p+1}", M.copy()))
        for i in range(k + 1, n):
            f = M[i, k] / M[k, k]
            if f:
                M[i] -= f * M[k]
                log.append((f"R{i+1} = R{i+1} - ({f:.4g}) R{k+1}", M.copy()))
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - M[i, i + 1:n] @ x[i + 1:]) / M[i, i]
    return x, log


def draw(X, conn, fixed, F, dim, u=None, scale=1.0):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(projection="3d" if dim == 3 else None)
    for e, (i, j) in enumerate(conn):
        ax.plot(*X[[i, j]].T, color="#33ff66", lw=2)
        ax.text(*X[[i, j]].mean(0), f"({e+1})", color="#ffcc00")
        if u is not None:
            Y = X + scale * u.reshape(-1, dim)
            ax.plot(*Y[[i, j]].T, "--", color="#ff6633")
    for n, p in enumerate(X):
        ax.text(*p, f" {n+1}", color="white", fontsize=12)
    sup = fixed.reshape(-1, dim).any(1)
    ax.scatter(*X[sup].T, marker="^", s=120, color="red", label="support")
    Fn = F.reshape(-1, dim)
    ld = np.linalg.norm(Fn, axis=1) > 0
    if ld.any():
        span = np.ptp(X, axis=0).max() or 1
        V = Fn[ld] / np.abs(Fn).max() * 0.3 * span
        kw = dict(angles="xy", scale_units="xy", scale=1) if dim == 2 else {}
        ax.quiver(*X[ld].T, *V.T, color="cyan", label="load", **kw)
    if dim == 2:
        ax.set_aspect("equal", "datalim")
    ax.legend(loc="best", fontsize=8)
    st.pyplot(fig)


# ---------------- input deck ----------------
st.title("TRUSS.EXE  v1.0")
st.text("FINITE ELEMENT TUTOR // BAR + TRUSS ELEMENTS // NO AI, JUST LINEAR ALGEBRA")
preset = st.sidebar.selectbox("LOAD PROBLEM", list(PRESETS))
P = PRESETS[preset]
if "step" not in st.session_state:
    st.session_state.step = STEPS[0]
step = st.sidebar.radio("GO TO STEP", STEPS, key="step")
st.sidebar.text("UNITS: m, m^2, GPa, kN\nOUTPUT: m, mm, kN, MPa")

with st.expander(">> INPUT DECK (edit any cell, add rows; row n = node n)", expanded=step == STEPS[0]):
    dim = st.radio("dimension", [2, 3], index=P["dim"] - 2, horizontal=True, key="dim" + preset)
    nd = st.data_editor(pd.DataFrame(P["nodes"], columns=NC), num_rows="dynamic", key="n" + preset)
    el = st.data_editor(pd.DataFrame(P["elems"], columns=EC), num_rows="dynamic", key="e" + preset)

# ---------------- the FEM engine ----------------
nd, el = nd.fillna(0), el.dropna()
X = nd[NC[:dim]].to_numpy(float)
fixed = nd[NC[3:3 + dim]].astype(bool).to_numpy().ravel()
F = nd[NC[6:6 + dim]].to_numpy(float).ravel() * 1e3          # kN -> N
N, ndof, ax_ = len(X), dim * len(X), "xyz"[:dim]
lab = [f"d{n+1}{a}" for n in range(N) for a in ax_]
E_ = []
for e, (i, j, A, E) in enumerate(el.itertuples(index=False)):
    i, j = int(i) - 1, int(j) - 1
    if not (0 <= i < N and 0 <= j < N) or i == j:
        st.error(f"element {e+1}: bad node numbers"); st.stop()
    L = float(np.linalg.norm(X[j] - X[i]))
    if L == 0:
        st.error(f"element {e+1}: zero length"); st.stop()
    c = (X[j] - X[i]) / L
    lam = np.outer(c, c)
    k = A * E * 1e9 / L * np.block([[lam, -lam], [-lam, lam]])
    dofs = [dim * i + a for a in range(dim)] + [dim * j + a for a in range(dim)]
    E_.append(dict(i=i, j=j, A=A, E=E * 1e9, L=L, c=c, lam=lam, k=k, dofs=dofs))
K = np.zeros((ndof, ndof))
for d in E_:
    K[np.ix_(d["dofs"], d["dofs"])] += d["k"]
free, sup = np.where(~fixed)[0], np.where(fixed)[0]
Kff, Ff = K[np.ix_(free, free)], F[free]
if len(free) == 0 or abs(np.linalg.det(Kff / np.abs(Kff).max())) < 1e-12:
    st.error("reduced [K] is singular: the structure is a mechanism. add supports or members."); st.stop()
uf, glog = gauss(Kff, Ff)
u = np.zeros(ndof); u[free] = uf
R = K @ u - F
for d in E_:
    ue = u[d["dofs"]]
    d["strain"] = d["c"] @ (ue[dim:] - ue[:dim]) / d["L"]
    d["stress"] = d["E"] * d["strain"]
conn = [(d["i"], d["j"]) for d in E_]
pick = lambda: st.selectbox("element", range(1, len(E_) + 1), format_func=lambda e: f"element ({e})") - 1

# ---------------- the six screens ----------------
st.header(step)
if step == STEPS[0]:
    draw(X, conn, fixed, F, dim)
    st.subheader("nodes + degrees of freedom")
    st.dataframe(pd.DataFrame({"coords (m)": [tuple(p) for p in X],
        "DOF labels": [", ".join(lab[dim*n:dim*n+dim]) for n in range(N)],
        "global DOF #": [", ".join(str(dim*n+a+1) for a in range(dim)) for n in range(N)],
        "fixed": [", ".join(a for a, f in zip(ax_, fixed[dim*n:dim*n+dim]) if f) or "-" for n in range(N)],
        "load (kN)": [tuple(F[dim*n:dim*n+dim] / 1e3) for n in range(N)]}, index=range(1, N + 1)))
    st.subheader("element connectivity + geometry")
    st.latex(r"L=\sqrt{(x_j-x_i)^2+(y_j-y_i)^2" + (r"+(z_j-z_i)^2" if dim == 3 else "") + r"}\qquad C_x=\frac{x_j-x_i}{L},\;C_y=\frac{y_j-y_i}{L}" + (r",\;C_z=\frac{z_j-z_i}{L}" if dim == 3 else ""))
    st.dataframe(pd.DataFrame([dict(i=d["i"]+1, j=d["j"]+1, L_m=round(d["L"], 4), A_m2=d["A"], E_GPa=d["E"]/1e9,
        **{f"C{a}": round(v, 4) for a, v in zip(ax_, d["c"])}) for d in E_], index=range(1, len(E_) + 1)))

elif step == STEPS[1]:
    e = pick(); d = E_[e]; C = d["c"]
    st.text("1) length + direction cosines from the node coordinates")
    st.latex(rf"L^{{({e+1})}}={d['L']:.4f}\,\text{{m}},\qquad " + r",\;".join(rf"C_{a}={v:.4f}" for a, v in zip(ax_, C)))
    st.text("2) in LOCAL coords the bar is a 1D spring")
    st.latex(rf"\hat k=\frac{{AE}}{{L}}\begin{{bmatrix}}1&-1\\-1&1\end{{bmatrix}},\qquad \frac{{AE}}{{L}}=\frac{{({d['A']})({d['E']:.3g})}}{{{d['L']:.4f}}}={d['A']*d['E']/d['L']:.4e}\,\text{{N/m}}")
    st.text("3) rotate to GLOBAL coords with the transformation matrix T*")
    st.latex(r"T^*=\begin{bmatrix}" + "&".join(f"C_{a}" for a in ax_) + "&" + "&".join(["0"] * dim) + r"\\" + "&".join(["0"] * dim) + "&" + "&".join(f"C_{a}" for a in ax_) + r"\end{bmatrix},\qquad k=T^{*T}\hat k\,T^*=\frac{AE}{L}\begin{bmatrix}\lambda&-\lambda\\-\lambda&\lambda\end{bmatrix}")
    st.latex(r"\lambda=\{C\}\{C\}^T=\begin{bmatrix}" + r"\\".join("&".join(f"{v:.4f}" for v in r) for r in d["lam"]) + r"\end{bmatrix}")
    st.text("4) element stiffness matrix in global coordinates")
    show(d["k"], [lab[q] for q in d["dofs"]], [lab[q] for q in d["dofs"]], "N/m")

elif step == STEPS[2]:
    st.latex(r"[K]=\sum_{e}[k^{(e)}]\quad\text{(each } k_{ab}\text{ is added to } K_{pq}\text{, where } p,q \text{ are the global DOFs of local } a,b)")
    st.dataframe(pd.DataFrame({"element": range(1, len(E_) + 1), "nodes": [f"{d['i']+1}-{d['j']+1}" for d in E_],
        "global DOFs": [", ".join(f"{lab[q]}(#{q+1})" for q in d["dofs"]) for d in E_]}).set_index("element"))
    e = st.slider("add elements one by one: show [K] after element", 1, len(E_), len(E_))
    Kp = np.zeros_like(K)
    for d in E_[:e]:
        Kp[np.ix_(d["dofs"], d["dofs"])] += d["k"]
    st.text(f"green = entries touched by element ({e}). [K] so far = k(1)" + "".join(f" + k({q})" for q in range(2, e + 1)))
    show(Kp, lab, lab, "N/m", hl=E_[e - 1]["dofs"])
    if e == len(E_):
        st.text(f"COMPLETE global [K]: {ndof} x {ndof}, symmetric = {np.allclose(K, K.T)}, singular before BCs (rigid-body motion).")

elif step == STEPS[3]:
    st.text(f"constrained DOFs (d = 0): {', '.join(lab[q] for q in sup) or 'none'}")
    st.text(f"free DOFs (unknown):     {', '.join(lab[q] for q in free)}")
    st.latex(r"\begin{Bmatrix}F_f\\R_s\end{Bmatrix}=\begin{bmatrix}K_{ff}&K_{fs}\\K_{sf}&K_{ss}\end{bmatrix}\begin{Bmatrix}d_f\\d_s=0\end{Bmatrix}\;\Rightarrow\;[K_{ff}]\{d_f\}=\{F_f\}")
    st.text("so: DELETE every row and column of a constrained DOF. what is left is the reduced system:")
    c1, c2 = st.columns([3, 1])
    with c1:
        show(Kff, [lab[q] for q in free], [lab[q] for q in free], "N/m")
    with c2:
        show(Ff[:, None], [lab[q].replace("d", "F") for q in free], ["F"], "N")

elif step == STEPS[4]:
    st.latex(r"[K]\{d\}=\{F\}\;\rightarrow\;\{d_f\}=[K_{ff}]^{-1}\{F_f\}")
    with st.expander(f"Gaussian elimination: {len(glog)} row operations, then back substitution"):
        show(np.hstack([Kff, Ff[:, None]]), [f"R{q+1}" for q in range(len(free))], [lab[q] for q in free] + ["F"], "(start)")
        for msg, M in glog[:40]:
            st.text(msg)
            show(M, [f"R{q+1}" for q in range(len(free))], [lab[q] for q in free] + ["F"])
        st.text("back substitution, bottom row up:  d_i = (F_i - sum_{j>i} K_ij d_j) / K_ii")
    st.dataframe(pd.DataFrame({"d (m)": [f"{v:.4e}" for v in u], "d (mm)": [f"{v*1e3:.4f}" for v in u],
        "type": ["fixed" if f else "solved" for f in fixed]}, index=lab))
    st.text(f"check vs numpy.linalg.solve: max error = {np.abs(uf - np.linalg.solve(Kff, Ff)).max():.2e} m")
    sc = st.slider("deformed shape magnification", 1, 2000, 200)
    draw(X, conn, fixed, F, dim, u, sc)

else:
    st.subheader("reaction forces")
    st.latex(r"\{R\}=[K]\{d\}-\{F_{applied}\}\quad\text{(evaluated at the constrained DOFs)}")
    st.dataframe(pd.DataFrame({"R (kN)": [f"{R[q]/1e3:.4f}" for q in sup]}, index=[lab[q].replace("d", "R") for q in sup]))
    tot = (R + F).reshape(-1, dim).sum(0) / 1e3
    st.text("equilibrium check, sum of (loads + reactions) = " + ", ".join(f"{a}: {v:.2e} kN" for a, v in zip(ax_, tot)))
    st.subheader("element strain + stress")
    st.latex(r"\varepsilon=\frac{1}{L}\begin{bmatrix}-C&C\end{bmatrix}\begin{Bmatrix}d_i\\d_j\end{Bmatrix},\qquad \sigma=E\,\varepsilon")
    e = pick(); d = E_[e]; ue = u[d["dofs"]]
    st.latex(rf"\sigma^{{({e+1})}}=\frac{{{d['E']:.3g}}}{{{d['L']:.4f}}}\begin{{bmatrix}}" + "&".join(f"{v:.4f}" for v in np.r_[-d["c"], d["c"]]) + r"\end{bmatrix}\begin{Bmatrix}" + r"\\".join(f"{v:.4e}" for v in ue) + rf"\end{{Bmatrix}}={d['stress']/1e6:.3f}\,\text{{MPa}}")
    st.dataframe(pd.DataFrame([dict(L_m=round(d["L"], 4), strain=f"{d['strain']:.4e}", stress_MPa=round(d["stress"]/1e6, 3),
        force_kN=round(d["stress"]*d["A"]/1e3, 3), state="TENSION" if d["stress"] > 0 else "COMPRESSION" if d["stress"] < 0 else "zero")
        for d in E_], index=[f"({q})" for q in range(1, len(E_) + 1)]))
    rep = f"TRUSS.EXE REPORT - {preset}\n\nGLOBAL K (N/m)\n{np.array2string(K, precision=4)}\n\nREDUCED Kff\n{np.array2string(Kff, precision=4)}\nFf (N) {Ff}\n\n"
    rep += "\n".join(f"{lab[q]} = {u[q]:.4e} m" for q in range(ndof)) + "\n\n" + "\n".join(f"{lab[q].replace('d','R')} = {R[q]/1e3:.4f} kN" for q in sup)
    rep += "\n\n" + "\n".join(f"element {q+1}: stress = {d['stress']/1e6:.3f} MPa" for q, d in enumerate(E_))
    st.download_button("SAVE REPORT.TXT", rep, "report.txt")

c1, _, c2 = st.columns([1, 4, 1])
go = lambda s: st.session_state.update(step=STEPS[(STEPS.index(st.session_state.step) + s) % len(STEPS)])
c1.button("<< PREV", on_click=go, args=(-1,))
c2.button("NEXT >>", on_click=go, args=(1,))
