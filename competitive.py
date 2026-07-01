"""Point 2: head-to-head vs RandNLA preconditioning (Chen-Tan) and RRQR pruning (DD-RFM).

Honest question: our contribution designs the POINTS (rows); RandNLA preconditions the solve and RRQR
prunes FEATURES (columns) -- both act on a given system. What does point design buy OVER them?
  Part A: points-to-accuracy -- min #points to reach a target error (uniform vs christoffel vs greedy).
  Part B: solve cost at fixed N (overdetermined) -- iterations + wall-time for direct SVD / raw-LSQR /
          RandNLA sketch-precond / RRQR / ours(whitened) / christoffel+RandNLA (composability).
High-frequency Poisson u=sin(4 pi x)sin(4 pi y), M=1200, ws=8 (best-fit ~1e-6).
"""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src")))
import numpy as np
from rfm.features import RandomFeatureMap
from rfm.domains import Rectangle
from rfm.operators import Poisson, SineProduct
from rfm.christoffel import build_reference_quad, ResidualSpace
from rfm import samplers as S
from rfm.solver import assemble, errors
from rfm import baselines as BL
from rfm.utils import save_json, results_dir

dom = Rectangle(0, 1, 0, 1); exact = SineProduct((4, 4))
Xt = dom.sample_interior(6000, np.random.default_rng(99))
M, ws = 1200, 8.0
fmap = RandomFeatureMap(d=2, M=M, activation="tanh", w_scale=ws, b_scale=ws, seed=7)
prob = Poisson(fmap=fmap, exact=exact)
rq = build_reference_quad(dom, 9000, 400, np.random.default_rng(1))
rs = ResidualSpace(prob, rq); T = rs.whitening(); r = rs.numerical_rank()
bdry = S.make_boundary(dom, 400, np.random.default_rng(2), w_bc=1.0)
print(f"M={M} rank r={r}", flush=True)

def design_matrix(design):
    A, b, w = assemble(prob, S.rebalance(design)); sw = np.sqrt(w)
    return sw[:, None] * A, sw * b

def err_of(c): return errors(prob, c, Xt)["rel_l2"]

# ---------- Part A: points-to-accuracy (direct solve) ----------
print("=== Part A: error vs #points (direct SVD) ===")
A_rows = {"N": [], "uniform": [], "christoffel": [], "greedy": []}
for cf in (1.0, 1.5, 2.0, 3.0, 5.0):
    N = int(round(cf * r)); eu, ec = [], []
    for sd in range(3):
        rr = np.random.default_rng(50 + sd); pool = S.build_interior_pool(prob, dom, max(20 * N, 6000), rr)
        Au, bu = design_matrix(S.UniformSampler().build(prob, dom, N, rr, bdry))
        eu.append(err_of(BL.direct_svd(Au, bu)[0]))
        Ac, bc = design_matrix(S.ChristoffelSampler(rs).build(prob, dom, N, rr, bdry, pool=pool))
        ec.append(err_of(BL.direct_svd(Ac, bc)[0]))
    rg = np.random.default_rng(7); poolg = S.build_interior_pool(prob, dom, min(max(20 * N, 6000), 8000), rg)
    Ag, bg = design_matrix(S.GreedySampler(rs, lam=1e-6).build(prob, dom, N, rg, bdry, pool=poolg))
    eg = err_of(BL.direct_svd(Ag, bg)[0])
    A_rows["N"].append(N); A_rows["uniform"].append(float(np.median(eu)))
    A_rows["christoffel"].append(float(np.median(ec))); A_rows["greedy"].append(float(eg))
    print(f"  N={N:4d} (N/r={cf}) unif={A_rows['uniform'][-1]:.2e} chri={A_rows['christoffel'][-1]:.2e} "
          f"gree={A_rows['greedy'][-1]:.2e}", flush=True)

# ---------- Part B: solve cost at fixed N (overdetermined so sketch has s>=M) ----------
N = int(round(2.5 * r))
print(f"\n=== Part B: solve cost at N={N} (n>=M, M={M}) ===")
rr = np.random.default_rng(123); pool = S.build_interior_pool(prob, dom, max(20 * N, 8000), rr)
du = S.UniformSampler().build(prob, dom, N, rr, bdry)
dc = S.ChristoffelSampler(rs).build(prob, dom, N, rr, bdry, pool=pool)
Au, bu = design_matrix(du); Ac, bc = design_matrix(dc)
from scipy.sparse.linalg import lsqr as _lsqr
Tridge = rs.whitening_ridge(rho=1e-8)        # full-rank ridge preconditioner (point-independent)
B_rows = []
def record(name, c, itn, sec):
    e = err_of(c); B_rows.append({"method": name, "rel_l2": float(e), "iters": itn, "sec": round(sec, 3)})
    print(f"  {name:42s} err={e:.2e} iters={str(itn):>5s} time={sec:.3f}s", flush=True)
def ridge_lsqr(A, b):
    t0 = time.perf_counter(); o = _lsqr(A @ Tridge, b, atol=1e-10, btol=1e-10, iter_lim=3000, conlim=1e18)
    return Tridge @ o[0], int(o[2]), time.perf_counter() - t0
# uniform points + various solvers
c, sec = BL.direct_svd(Au, bu);                          record("uniform + direct SVD (gold; O(NM^2))", c, "-", sec)
c, itn, sec = BL.raw_lsqr(Au, bu, iter_lim=3000);        record("uniform + raw LSQR (no precond)", c, itn, sec)
c, itn, sec = BL.sketch_precond_lsqr(Au, bu, seed=1);    record("uniform + RandNLA LSRN sketch-precond", c, itn, sec)
c, itn, sec = ridge_lsqr(Au, bu);                        record("uniform + ridge-whiten precond LSQR", c, itn, sec)
c, cols, sec = BL.rrqr_prune_solve(Au, bu, k=r);         record(f"uniform + RRQR prune (k=r={r})", c, "-", sec)
# our points
c, sec = BL.direct_svd(Ac, bc);                          record("christoffel + direct SVD", c, "-", sec)
c, itn, sec = ridge_lsqr(Ac, bc);                        record("christoffel + ridge-whiten LSQR (ours)", c, itn, sec)
c, itn, sec = BL.sketch_precond_lsqr(Ac, bc, seed=1);    record("christoffel + RandNLA (composes)", c, itn, sec)

save_json(os.path.join(results_dir(), "competitive.json"),
          {"M": M, "r": r, "part_a_points_to_accuracy": A_rows, "part_b_solve_cost": B_rows})
print("[competitive] done")
