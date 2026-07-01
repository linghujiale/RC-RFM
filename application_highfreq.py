"""Point 4 (application): high-frequency Helmholtz at REAL accuracy with Fourier (sin) random features.

Helmholtz -Delta u - k^2 u = f, k=16, u=sin(5 pi x)sin(5 pi y) (oscillatory; |w|~22 to represent).
tanh features are poor at high frequency; sin (random Fourier) features are the right basis. We show:
the optimally-sampled, whitened system enables accurate, ITERATIVELY-solvable high-frequency RFM at
a feature count M where naive uniform RFM is ill-conditioned / iteratively unsolvable.
"""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src")))
import numpy as np
from scipy.sparse.linalg import lsqr
from rfm.features import RandomFeatureMap
from rfm.domains import Rectangle
from rfm.operators import Helmholtz, SineProduct
from rfm.christoffel import build_reference_quad, ResidualSpace
from rfm import samplers as S
from rfm.solver import assemble, errors
from rfm import baselines as BL
from rfm import metrics

dom = Rectangle(0, 1, 0, 1); exact = SineProduct((5, 5)); K = 16.0
Xt = dom.sample_interior(8000, np.random.default_rng(99))

def design_matrix(prob, design):
    A, b, w = assemble(prob, S.rebalance(design)); sw = np.sqrt(w)
    return sw[:, None] * A, sw * b

def main():
    t0 = time.time()
    # capacity check across (M, ws) for sin features
    print("=== capacity (best-fit, large-N direct SVD) for Helmholtz k=16, sin features ===")
    best_cfg = None
    for M, ws in [(1500, 25.0), (2500, 25.0), (2500, 30.0), (4000, 28.0)]:
        fmap = RandomFeatureMap(d=2, M=M, activation="sin", w_scale=ws, b_scale=np.pi, seed=7)
        prob = Helmholtz(fmap=fmap, exact=exact, k=K)
        bdry = S.make_boundary(dom, 600, np.random.default_rng(2), w_bc=1.0)
        big = S.UniformSampler().build(prob, dom, 9000, np.random.default_rng(0), bdry)
        A, b = design_matrix(prob, big); c, _ = BL.direct_svd(A, b, rcond=1e-13)
        bf = errors(prob, c, Xt)["rel_l2"]
        print(f"  M={M:4d} ws={ws:4.0f} -> best-fit={bf:.2e}", flush=True)
        if best_cfg is None or bf < best_cfg[2]:
            best_cfg = (M, ws, bf)
    M, ws, bf = best_cfg
    print(f"  --> chosen M={M} ws={ws} best-fit={bf:.2e}", flush=True)

    fmap = RandomFeatureMap(d=2, M=M, activation="sin", w_scale=ws, b_scale=np.pi, seed=7)
    prob = Helmholtz(fmap=fmap, exact=exact, k=K)
    rq = build_reference_quad(dom, 12000, 800, np.random.default_rng(1))
    rs = ResidualSpace(prob, rq); T = rs.whitening(); r = rs.numerical_rank()
    bdry = S.make_boundary(dom, 800, np.random.default_rng(2), w_bc=1.0)
    print(f"\n=== Helmholtz k=16: M={M} rank r={r}, best-fit={bf:.2e} ===")

    # conditioning + iterative solvability
    du = S.rebalance(S.UniformSampler().build(prob, dom, 3000, np.random.default_rng(5), bdry))
    ku = metrics.design_condition_number(prob, du, whiten=None)
    Au, bu = design_matrix(prob, du); _, itu, _ = BL.raw_lsqr(Au, bu, iter_lim=4000)
    dc = S.rebalance(S.ChristoffelSampler(rs).build(prob, dom, 3000, np.random.default_rng(6), bdry))
    Ac, bc = design_matrix(prob, dc); ow = lsqr(Ac @ T, bc, atol=1e-10, btol=1e-10, iter_lim=4000, conlim=1e18)
    kc = float(np.linalg.cond(Ac @ T))
    print(f"  conditioning: uniform-raw kappa={ku:.2e} (raw LSQR iters={itu}) | "
          f"christoffel-whitened kappa={kc:.2e} (LSQR iters={int(ow[2])})", flush=True)

    # accuracy-per-point
    print("  accuracy-per-point (full-space direct):")
    acc = {m: [] for m in ("uniform", "christoffel", "greedy")}; Ns = []
    for cf in (1.2, 1.8, 3.0, 5.0):
        N = int(round(cf * r)); Ns.append(N); eu, ec = [], []
        for sd in range(3):
            rr = np.random.default_rng(60 + sd); pool = S.build_interior_pool(prob, dom, max(20 * N, 8000), rr)
            Auu, buu = design_matrix(prob, S.UniformSampler().build(prob, dom, N, rr, bdry))
            eu.append(errors(prob, BL.direct_svd(Auu, buu)[0], Xt)["rel_l2"])
            Acc, bcc = design_matrix(prob, S.ChristoffelSampler(rs).build(prob, dom, N, rr, bdry, pool=pool))
            ec.append(errors(prob, BL.direct_svd(Acc, bcc)[0], Xt)["rel_l2"])
        rg = np.random.default_rng(7); poolg = S.build_interior_pool(prob, dom, min(max(20 * N, 8000), 9000), rg)
        Ag, bg = design_matrix(prob, S.GreedySampler(rs, lam=1e-6).build(prob, dom, N, rg, bdry, pool=poolg))
        eg = errors(prob, BL.direct_svd(Ag, bg)[0], Xt)["rel_l2"]
        acc["uniform"].append(float(np.median(eu))); acc["christoffel"].append(float(np.median(ec))); acc["greedy"].append(float(eg))
        print(f"    N={N:4d} (N/r={cf}) unif={acc['uniform'][-1]:.2e} chri={acc['christoffel'][-1]:.2e} gree={acc['greedy'][-1]:.2e}", flush=True)

    from rfm.utils import save_json, results_dir
    save_json(os.path.join(results_dir(), "application_highfreq.json"),
              {"k": K, "M": M, "ws": ws, "best_fit": bf, "rank": r,
               "kappa_uniform_raw": float(ku), "kappa_christoffel": kc,
               "lsqr_uniform_raw": itu, "lsqr_christoffel": int(ow[2]),
               "acc_N": Ns, "acc": acc, "wall_s": time.time() - t0})
    print(f"[application] done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
