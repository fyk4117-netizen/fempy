# TRUSS.EXE v2.0 - a step-by-step Finite Element tutor for 2D/3D truss (bar) structures.
# Pure Python. numpy does the math, streamlit draws the screen. No AI is used while it runs.
import re
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt

st.set_page_config(page_title="TRUSS.EXE", layout="wide")
INK, PAPER, CARD, HI = "#111111", "#efeadd", "#fffdf6", "#ffd966"
st.markdown(f"""<style>
.stApp {{background:{PAPER};}}
h1, h2, h3, p, li, label, input, textarea, pre, code {{font-family:"Courier New", Courier, monospace !important; color:{INK};}}
.win {{font-family:"Courier New", Courier, monospace; border:2px solid {INK}; background:{CARD};
       box-shadow:5px 5px 0 {INK}; margin:6px 0 20px 0;}}
.win .tb {{background:{INK}; color:{PAPER}; padding:3px 10px; font-weight:bold;}}
.win .bd {{padding:10px 14px; line-height:1.5; color:{INK};}}
.stButton button {{border:2px solid {INK}; border-radius:0; background:#d9d3c2; box-shadow:3px 3px 0 {INK};}}
</style>""", unsafe_allow_html=True)
plt.rcParams.update({"font.family": "monospace", "axes.facecolor": CARD, "figure.facecolor": CARD})


def box(title, text):
    """A retro 'window' with a black title bar. Plain text inside; new lines become line breaks."""
    st.markdown(f'<div class="win"><div class="tb">{title}</div><div class="bd">{text.strip().replace(chr(10), "<br>")}</div></div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ examples
SUP = {"free": "free (moves any direction)", "pin": "pin (cannot move at all)",
       "rollx": "roller (slides along x only)", "rolly": "roller (slides along y only)",
       "rollz": "roller (slides in x-y plane, z held)"}
FIX = {"free": (0, 0, 0), "pin": (1, 1, 1), "rollx": (0, 1, 1), "rolly": (1, 0, 1), "rollz": (0, 0, 1)}
EX = {
    "Roof truss (2D, 3 bars)": dict(dim=2, A=10.0, E=200.0, bars="1-2, 1-3, 2-3",
        nodes=[(0, 0, 0, "pin", 0, 0, 0), (4, 0, 0, "rollx", 0, 0, 0), (2, 3, 0, "free", 10, -20, 0)]),
    "Bridge (2D, 7 bars)": dict(dim=2, A=20.0, E=200.0, bars="1-2, 2-3, 1-4, 4-2, 2-5, 5-3, 4-5",
        nodes=[(0, 0, 0, "pin", 0, 0, 0), (3, 0, 0, "free", 0, -30, 0), (6, 0, 0, "rollx", 0, 0, 0),
               (1.5, 2, 0, "free", 0, 0, 0), (4.5, 2, 0, "free", 5, 0, 0)]),
    "Space tripod (3D, Logan Example 3.9)": dict(dim=3, A=10.0, E=210.0, bars="1-2, 1-3, 1-4",
        nodes=[(12, -3, -4, "free", 20, 0, 0), (0, 0, 0, "pin", 0, 0, 0), (12, -3, -7, "pin", 0, 0, 0),
               (14, 6, 0, "pin", 0, 0, 0)]),
}
STEPS = ["0 SET UP", "1 THE STRUCTURE", "2 ONE BAR", "3 ASSEMBLE", "4 SUPPORTS", "5 SOLVE", "6 FORCES"]

S = st.session_state
for k in list(S.keys()):          # keep typed values alive while the user visits other steps
    if k.startswith("w_"):
        S[k] = S[k]


def v(key, default):
    return S.setdefault("w_" + key, default)


# ------------------------------------------------------------------ helpers
def show(M, rows, cols, unit="", hl=None):
    """Print a matrix with one power of ten pulled out front, the way textbooks do."""
    M = np.atleast_2d(M)
    m = np.abs(M).max()
    p = int(np.floor(np.log10(m))) if m > 0 else 0
    df = pd.DataFrame(M / 10.0**p, index=rows, columns=cols)
    sty = df.style.format("{:.4f}")
    if hl is not None:
        css = pd.DataFrame("", index=df.index, columns=df.columns)
        css.iloc[hl, hl] = f"background-color:{HI}"
        sty = sty.apply(lambda _: css, axis=None)
    st.text(f"every number below is x 10^{p} {unit}")
    st.dataframe(sty)


def gauss(A, b):
    """Gaussian elimination with partial pivoting. Logs every row operation for display."""
    M, n, log = np.hstack([A, b[:, None]]).astype(float), len(b), []
    for k in range(n):
        p = k + np.argmax(abs(M[k:, k]))
        if p != k:
            M[[k, p]] = M[[p, k]]
            log.append((f"swap row {k+1} and row {p+1}", M.copy()))
        for i in range(k + 1, n):
            f = M[i, k] / M[k, k]
            if f:
                M[i] -= f * M[k]
                log.append((f"row {i+1} = row {i+1} - ({f:.4g}) x row {k+1}", M.copy()))
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - M[i, i + 1:n] @ x[i + 1:]) / M[i, i]
    return x, log


def draw(X, conn, fixed, F, dim, u=None, scale=1.0):
    fig = plt.figure(figsize=(6.5, 4.2))
    ax = fig.add_subplot(projection="3d" if dim == 3 else None)
    span = np.ptp(X, axis=0).max() or 1
    for e, (i, j) in enumerate(conn):
        ax.plot(*X[[i, j]].T, color=INK, lw=2.5)
        ax.text(*X[[i, j]].mean(0), f"[{e+1}]", color="#1f4e9e", fontsize=10, weight="bold")
        if u is not None:
            Y = X + scale * u.reshape(-1, dim)
            ax.plot(*Y[[i, j]].T, "--", color="#c0392b", lw=1.8)
    for n, p in enumerate(X):
        ax.text(*p, f"  ({n+1})", color=INK, fontsize=11, weight="bold")
    sup = fixed.reshape(-1, dim).any(1)
    ax.scatter(*X[sup].T, marker="^", s=170, color="#888888", edgecolor=INK, zorder=3, label="support")
    Fn = F.reshape(-1, dim)
    ld = np.linalg.norm(Fn, axis=1) > 0
    if ld.any():
        V = Fn[ld] / np.abs(Fn).max() * 0.3 * span
        kw = dict(angles="xy", scale_units="xy", scale=1, width=0.007) if dim == 2 else {}
        ax.quiver(*X[ld].T, *V.T, color="#c0392b", label="load", **kw)
        for p, vv, f in zip(X[ld], V, Fn[ld]):
            ax.text(*(p + vv), f" {np.linalg.norm(f)/1e3:g} kN", color="#c0392b", fontsize=9)
    if dim == 2:
        ax.set_aspect("equal", "datalim")
        ax.margins(0.2)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    if dim == 3:
        ax.set_zlabel("z (m)")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ------------------------------------------------------------------ header + navigation
st.markdown("# TRUSS.EXE v2.0")
st.text("A FINITE ELEMENT TUTOR FOR TRUSSES.  NO AI.  JUST ARITHMETIC, ONE STEP AT A TIME.")
S.setdefault("step", STEPS[0])
step = st.radio("STEP", STEPS, horizontal=True, key="step")
go = lambda s: S.update(step=STEPS[min(max(STEPS.index(S.step) + s, 0), len(STEPS) - 1)])

# ------------------------------------------------------------------ inputs (step 0)
ex = st.session_state.get("w_ex", list(EX)[0])
if step == STEPS[0]:
    box("WELCOME", """
A truss is a set of straight bars joined by pins, like a roof frame, a bridge or a tower crane.
This program finds how far each joint moves and how hard each bar is pulled or squeezed,
using the Finite Element Method (FEM). You do not need to know FEM. Just follow steps 0 to 6.
STEP 0: pick an example, change any number you like, then press NEXT.""")
    v("ex", list(EX)[0])
    ex = st.selectbox("Starting example", list(EX), key="w_ex")
P, tag = EX[ex], ex[:6]
dim = P["dim"]
nn = int(v(tag + "nn", len(P["nodes"])))
if step == STEPS[0]:
    c1, c2, c3 = st.columns(3)
    nn = c1.number_input("How many joints (nodes)?", 2, 12, step=1, key=f"w_{tag}nn")
    c2.number_input("Bar cross-section area A (cm^2)", 0.01, 1e4, key="w_" + tag + "A" if v(tag + "A", P["A"]) else "")
    c3.number_input("Material stiffness E (GPa).  steel ~200, aluminum ~70, wood ~11", 0.01, 1e4,
                    key="w_" + tag + "E" if v(tag + "E", P["E"]) else "")
    st.markdown("#### Joints")
    st.text("x, y" + (", z" if dim == 3 else "") + " = position in metres.  load = force pushed on the joint in kN"
            " (+ is the positive axis direction, - is the opposite).")
    hdr = st.columns([1, 2, 2] + ([2] if dim == 3 else []) + [5, 2, 2] + ([2] if dim == 3 else []))
    for c, t in zip(hdr, ["joint", "x (m)", "y (m)"] + (["z (m)"] if dim == 3 else []) + ["support type", "load x (kN)", "load y (kN)"] + (["load z (kN)"] if dim == 3 else [])):
        c.markdown(f"**{t}**")
sup_opts = ["free", "pin", "rollx", "rolly"] if dim == 2 else ["free", "pin", "rollz"]
rows = []
for n in range(int(nn)):
    d = P["nodes"][n] if n < len(P["nodes"]) else (0, 0, 0, "free", 0, 0, 0)
    keys = [f"{tag}{q}{n}" for q in ("x", "y", "z", "s", "fx", "fy", "fz")]
    vals = [v(k, dv) for k, dv in zip(keys, d)]
    if step == STEPS[0]:
        cols = st.columns([1, 2, 2] + ([2] if dim == 3 else []) + [5, 2, 2] + ([2] if dim == 3 else []))
        cols[0].markdown(f"**({n+1})**")
        ci = 1
        for q in range(3):
            if q < dim:
                cols[ci].number_input(keys[q], key="w_" + keys[q], label_visibility="collapsed", format="%g"); ci += 1
        cols[ci].selectbox(keys[3], sup_opts, format_func=SUP.get, key="w_" + keys[3], label_visibility="collapsed"); ci += 1
        for q in range(4, 4 + dim):
            cols[ci].number_input(keys[q], key="w_" + keys[q], label_visibility="collapsed", format="%g"); ci += 1
        vals = [S["w_" + k] for k in keys]
    rows.append(vals)
if step == STEPS[0]:
    st.markdown("#### Bars")
    st.text_input("Which joints does each bar connect? Write pairs like 1-2, 1-3, 2-3", key="w_" + tag + "bars"
                  if v(tag + "bars", P["bars"]) else "")
A_, E_mod = v(tag + "A", P["A"]) * 1e-4, v(tag + "E", P["E"]) * 1e9   # cm^2 -> m^2, GPa -> Pa
pairs = [(int(a) - 1, int(b) - 1) for a, b in re.findall(r"(\d+)\s*-\s*(\d+)", v(tag + "bars", P["bars"]))]

# ------------------------------------------------------------------ the FEM engine
X = np.array([[r[q] for q in range(dim)] for r in rows], float)
fixed = np.array([FIX[r[3] if r[3] in sup_opts else "free"][q] for r in rows for q in range(dim)], bool)
F = np.array([r[4 + q] for r in rows for q in range(dim)], float) * 1e3        # kN -> N
N, ndof, AX = len(X), dim * len(X), "xyz"[:dim]
lab = [f"d{n+1}{a}" for n in range(N) for a in AX]


def fail(msg):
    box("CANNOT SOLVE YET", msg + "\nGo to step 0 and fix it.")
    if step == STEPS[0]:
        draw(X, [], fixed, F, dim)
    st.stop()


if not pairs:
    fail("No bars found. Type pairs of joint numbers, for example 1-2, 2-3.")
B = []
for e, (i, j) in enumerate(pairs):
    if not (0 <= i < N and 0 <= j < N) or i == j:
        fail(f"Bar [{e+1}] connects joints {i+1} and {j+1}, but only joints 1 to {N} exist.")
    L = float(np.linalg.norm(X[j] - X[i]))
    if L == 0:
        fail(f"Bar [{e+1}] has zero length: joints {i+1} and {j+1} are at the same spot.")
    c = (X[j] - X[i]) / L
    lam = np.outer(c, c)
    dofs = [dim * i + a for a in range(dim)] + [dim * j + a for a in range(dim)]
    B.append(dict(i=i, j=j, L=L, c=c, lam=lam, k=A_ * E_mod / L * np.block([[lam, -lam], [-lam, lam]]), dofs=dofs))
K = np.zeros((ndof, ndof))
for b in B:
    K[np.ix_(b["dofs"], b["dofs"])] += b["k"]
free, sup = np.where(~fixed)[0], np.where(fixed)[0]
Kff, Ff = K[np.ix_(free, free)], F[free]
if len(free) == 0:
    fail("Every joint is pinned, so nothing can move. Make at least one joint free.")
if np.linalg.matrix_rank(Kff, tol=1e-9 * np.abs(Kff).max()) < len(free):
    fail("The structure can wobble or slide freely (a 'mechanism'). Add a bar or a support.")
uf, glog = gauss(Kff, Ff)
u = np.zeros(ndof); u[free] = uf
R = K @ u - F
for b in B:
    ue = u[b["dofs"]]
    b["strain"] = b["c"] @ (ue[dim:] - ue[:dim]) / b["L"]
    b["stress"] = E_mod * b["strain"]
conn = [(b["i"], b["j"]) for b in B]
pick = lambda key: st.selectbox("Look at bar", range(1, len(B) + 1), format_func=lambda e: f"bar [{e}]  (joint {B[e-1]['i']+1} to joint {B[e-1]['j']+1})", key=key) - 1

# ------------------------------------------------------------------ the steps
st.markdown(f"## {step}")
if step == STEPS[0]:
    st.markdown("#### Live picture of your truss")
    draw(X, conn, fixed, F, dim)
    st.text("Looks right? Press NEXT >> at the bottom.")

elif step == STEPS[1]:
    box("WHAT IS HAPPENING", f"""
Each joint can move a tiny bit in {', '.join(AX)}. Each of those possible moves is called a
DEGREE OF FREEDOM (DOF). This truss has {N} joints x {dim} directions = {ndof} DOFs.
We name them d1x, d1y ... (d = displacement, then joint number, then direction).
Each bar is described by its two end joints, its length L, and its DIRECTION COSINES C:
how much the bar points along x, y{', z' if dim == 3 else ''} (each C is between -1 and 1).""")
    c1, c2 = st.columns([1, 1])
    with c1:
        draw(X, conn, fixed, F, dim)
    with c2:
        st.markdown("#### Joints and their DOFs")
        st.dataframe(pd.DataFrame({"position (m)": [", ".join(f"{p:g}" for p in X[n]) for n in range(N)],
            "DOF names": [", ".join(lab[dim*n:dim*n+dim]) for n in range(N)],
            "DOF numbers": [", ".join(str(dim*n+a+1) for a in range(dim)) for n in range(N)],
            "held by support": [", ".join(a for a, f in zip(AX, fixed[dim*n:dim*n+dim]) if f) or "-" for n in range(N)],
            "load (kN)": [", ".join(f"{f/1e3:g}" for f in F[dim*n:dim*n+dim]) for n in range(N)]},
            index=[f"({n+1})" for n in range(N)]))
    st.markdown("#### Bars: connectivity and geometry")
    st.latex(r"L=\sqrt{(x_j-x_i)^2+(y_j-y_i)^2" + (r"+(z_j-z_i)^2" if dim == 3 else "") +
             r"}\qquad C_x=\frac{x_j-x_i}{L}\quad C_y=\frac{y_j-y_i}{L}" + (r"\quad C_z=\frac{z_j-z_i}{L}" if dim == 3 else ""))
    st.dataframe(pd.DataFrame([dict(from_joint=b["i"]+1, to_joint=b["j"]+1, L_m=round(b["L"], 4), A_cm2=A_*1e4, E_GPa=E_mod/1e9,
        **{f"C{a}": round(c, 4) for a, c in zip(AX, b["c"])}) for b in B], index=[f"[{e+1}]" for e in range(len(B))]))

elif step == STEPS[2]:
    box("WHAT IS HAPPENING", """
Every bar behaves like a very stiff spring. Along its own length its spring constant is AE/L
(area x material stiffness / length). But the bar is tilted, so we rotate that 1D spring into
the x-y(-z) axes using its direction cosines. The result is the bar's STIFFNESS MATRIX [k]:
it says what forces appear at the bar's two ends when those ends move.""")
    e = pick("bar2"); b = B[e]; C = b["c"]
    st.markdown("**A. Length and direction**")
    st.latex(rf"L^{{[{e+1}]}}={b['L']:.4f}\ \text{{m}},\qquad " + r",\quad ".join(rf"C_{a}={c:.4f}" for a, c in zip(AX, C)))
    st.markdown("**B. The bar as a 1D spring, in its own (local) direction**")
    st.latex(rf"\hat k=\frac{{AE}}{{L}}\begin{{bmatrix}}1&-1\\-1&1\end{{bmatrix}},\qquad \frac{{AE}}{{L}}=\frac{{({A_:.4g}\ \text{{m}}^2)({E_mod:.4g}\ \text{{Pa}})}}{{{b['L']:.4f}\ \text{{m}}}}={A_*E_mod/b['L']:.4e}\ \text{{N/m}}")
    st.markdown("**C. Rotate into global x-y" + ("-z" if dim == 3 else "") + " with the transformation matrix T***")
    st.latex(r"T^*=\begin{bmatrix}" + "&".join(f"C_{a}" for a in AX) + "&" + "&".join(["0"] * dim) + r"\\" +
             "&".join(["0"] * dim) + "&" + "&".join(f"C_{a}" for a in AX) +
             r"\end{bmatrix},\qquad [k]=T^{*T}\,\hat k\,T^*=\frac{AE}{L}\begin{bmatrix}\lambda&-\lambda\\-\lambda&\lambda\end{bmatrix}")
    st.latex(r"\lambda=\begin{Bmatrix}" + r"\\".join(f"C_{a}" for a in AX) + r"\end{Bmatrix}\begin{Bmatrix}" + "&".join(f"C_{a}" for a in AX) +
             r"\end{Bmatrix}=\begin{bmatrix}" + r"\\".join("&".join(f"{x:.4f}" for x in r) for r in b["lam"]) + r"\end{bmatrix}")
    st.markdown(f"**D. Result: [k] for bar [{e+1}], rows and columns labeled by the DOFs it touches**")
    show(b["k"], [lab[q] for q in b["dofs"]], [lab[q] for q in b["dofs"]], "N/m")

elif step == STEPS[3]:
    box("WHAT IS HAPPENING", f"""
Now we build ONE big stiffness matrix [K] for the whole truss ({ndof} x {ndof}, one row and column per DOF).
Rule: take each bar's small [k] and ADD each number into the big [K] at the row and column
of the DOF it belongs to. Where two bars share a joint, their numbers stack up.
Use the slider to add the bars one at a time. Yellow cells = where the newest bar was added.""")
    st.latex(r"[K]=\sum_{\text{bars}}[k]\qquad k_{ab}\ \text{goes into}\ K_{pq},\ \ p,q=\text{global DOF numbers of local}\ a,b")
    st.dataframe(pd.DataFrame({"joints": [f"{b['i']+1} to {b['j']+1}" for b in B],
        "its DOFs go to rows/columns": [", ".join(f"{lab[q]} (#{q+1})" for q in b["dofs"]) for b in B]},
        index=[f"[{e+1}]" for e in range(len(B))]))
    e = st.slider("Bars added so far", 1, len(B), 1)
    Kp = np.zeros_like(K)
    for b in B[:e]:
        Kp[np.ix_(b["dofs"], b["dofs"])] += b["k"]
    st.text("[K] = " + " + ".join(f"[k{q}]" for q in range(1, e + 1)) + ("    <- COMPLETE" if e == len(B) else ""))
    show(Kp, lab, lab, "N/m", hl=B[e - 1]["dofs"])

elif step == STEPS[4]:
    box("WHAT IS HAPPENING", f"""
Supports hold some DOFs at exactly zero. We already know those moves (they are 0), so we
cross out their rows and columns. What is left is a smaller system with only the unknown moves.
Held at zero:  {', '.join(lab[q] for q in sup) or 'none'}
Free to move:  {', '.join(lab[q] for q in free)}""")
    st.latex(r"\begin{Bmatrix}F_{free}\\R_{held}\end{Bmatrix}=\begin{bmatrix}K_{ff}&K_{fs}\\K_{sf}&K_{ss}\end{bmatrix}\begin{Bmatrix}d_{free}\\0\end{Bmatrix}\ \Rightarrow\ [K_{ff}]\{d_{free}\}=\{F_{free}\}")
    st.markdown("**Full [K], crossed-out rows/columns hidden. Reduced stiffness [Kff]:**")
    show(Kff, [lab[q] for q in free], [lab[q] for q in free], "N/m")
    st.markdown("**Reduced load vector {F free} (the loads on the free DOFs):**")
    show(Ff[:, None], [lab[q].replace("d", "F") for q in free], ["F"], "N")

elif step == STEPS[5]:
    box("WHAT IS HAPPENING", """
We now have [K]{d} = {F}: stiffness x displacement = force. Same idea as F = k x for one spring,
just with many springs at once. We solve it with Gaussian elimination (the method from algebra
class): clear out numbers below the diagonal, then solve from the bottom row up.""")
    st.latex(r"[K_{ff}]\{d_{free}\}=\{F_{free}\}")
    with st.expander(f"Show all {len(glog)} elimination steps"):
        hdr = [lab[q] for q in free] + ["F"]
        rr = [f"row {q+1}" for q in range(len(free))]
        st.text("start: [K | F]")
        show(np.hstack([Kff, Ff[:, None]]), rr, hdr)
        for msg, M in glog[:40]:
            st.text(msg)
            show(M, rr, hdr)
        st.text("then back-substitute, last row first:  d_i = (F_i - sum of K_ij d_j for j > i) / K_ii")
    st.markdown("#### Joint displacements")
    st.dataframe(pd.DataFrame({"displacement (m)": [f"{x:.4e}" for x in u], "displacement (mm)": [f"{x*1e3:.4f}" for x in u],
        "how found": ["held by support = 0" if f else "solved" for f in fixed]}, index=lab))
    st.text(f"cross-check against numpy.linalg.solve: biggest difference = {np.abs(uf - np.linalg.solve(Kff, Ff)).max():.1e} m")
    sc = st.slider("Exaggerate the bending in the picture (x times)", 1, 3000, 300)
    draw(X, conn, fixed, F, dim, u, sc)
    st.text("solid black = original shape.  dashed red = deformed shape (exaggerated).")

else:
    box("WHAT IS HAPPENING", """
1) REACTIONS: the forces the supports push back with. Multiply the full [K] by all displacements
   and subtract the applied loads. The supports must exactly balance the loads.
2) STRESS in each bar: how much the bar got longer or shorter (strain) x E.
   + means TENSION (bar is being stretched).  - means COMPRESSION (bar is being squeezed).""")
    st.markdown("#### 1) Reaction forces at the supports")
    st.latex(r"\{R\}=[K]\{d\}-\{F_{applied}\}")
    st.dataframe(pd.DataFrame({"reaction (kN)": [f"{R[q]/1e3:.4f}" for q in sup]}, index=[lab[q].replace("d", "R") for q in sup]))
    tot = (R + F).reshape(-1, dim).sum(0) / 1e3
    st.text("check, loads + reactions should add to zero:  " + ",  ".join(f"sum {a} = {t:.1e} kN" for a, t in zip(AX, tot)))
    st.markdown("#### 2) Strain and stress in each bar")
    st.latex(r"\varepsilon=\frac{1}{L}\begin{bmatrix}-C&C\end{bmatrix}\begin{Bmatrix}d_i\\d_j\end{Bmatrix},\qquad \sigma=E\,\varepsilon,\qquad \text{force}=\sigma A")
    e = pick("bar6"); b = B[e]; ue = u[b["dofs"]]
    st.latex(rf"\sigma^{{[{e+1}]}}=\frac{{{E_mod:.3g}}}{{{b['L']:.4f}}}\begin{{bmatrix}}" + "&".join(f"{x:.4f}" for x in np.r_[-b["c"], b["c"]]) +
             r"\end{bmatrix}\begin{Bmatrix}" + r"\\".join(f"{x:.4e}" for x in ue) + rf"\end{{Bmatrix}}={b['stress']/1e6:.3f}\ \text{{MPa}}")
    st.dataframe(pd.DataFrame([dict(length_m=round(b["L"], 4), strain=f"{b['strain']:.4e}", stress_MPa=round(b["stress"]/1e6, 3),
        force_kN=round(b["stress"]*A_/1e3, 3), state="TENSION (stretched)" if b["stress"] > 0 else "COMPRESSION (squeezed)" if b["stress"] < 0 else "no force")
        for b in B], index=[f"[{e+1}]" for e in range(len(B))]))
    rep = f"TRUSS.EXE REPORT - {ex}\n\nGLOBAL [K] (N/m)\n{np.array2string(K, precision=4, max_line_width=200)}\n\n"
    rep += f"REDUCED [Kff] (N/m)\n{np.array2string(Kff, precision=4, max_line_width=200)}\n{{F free}} (N) = {Ff}\n\n"
    rep += "\n".join(f"{lab[q]} = {u[q]:.4e} m" for q in range(ndof)) + "\n\n"
    rep += "\n".join(f"{lab[q].replace('d', 'R')} = {R[q]/1e3:.4f} kN" for q in sup) + "\n\n"
    rep += "\n".join(f"bar [{q+1}]: stress = {b['stress']/1e6:.3f} MPa" for q, b in enumerate(B))
    st.download_button("SAVE FULL REPORT (.txt)", rep, "truss_report.txt")

st.text("")
c1, c2, _ = st.columns([1, 1, 5])
c1.button("<< BACK", on_click=go, args=(-1,), disabled=step == STEPS[0])
c2.button("NEXT >>", on_click=go, args=(1,), disabled=step == STEPS[-1])
