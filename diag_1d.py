"""Diagnostic: isolate approximation power vs conditioning for 1D Poisson RFM.

Questions:
 (A) Can the RFM feature space represent sin(pi x) at all? (large well-sampled robust LS, raw basis)
 (B) numerical rank / spectrum of the residual Gram G vs (M, w_scale).
 (C) accuracy under different solve modes: raw robust-lstsq vs whitened(truncated) vs ridge.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src")))
import numpy as np
from rfm.features import RandomFeatureMap
from rfm.domains import Interval
from rfm.operators import Poisson, SineProduct
from rfm.christoffel import build_reference_quad, ResidualSpace
from rfm import samplers as S
from rfm.solver import solve, errors, assemble

rng = np.random.default_rng(0)
dom = Interval(0, 1)
Xt = np.linspace(0, 1, 1000).reshape(-1, 1)

def big_uniform_solve(prob, n=4000, ridge=0.0):
    Xi = dom.sample_interior(n, rng); Xb = np.array([[0.0],[1.0]]); nb=np.array([[-1.],[1.]])
    d = S.Design(Xi=Xi, wi=np.full(n, 1.0/n), Xb=Xb, nb=nb, wb=np.full(2, 0.5))
    res = solve(prob, d, whiten=None, ridge=ridge)
    return res, errors(prob, res.coeffs, Xt)

print("=== (A)/(B) feature space capacity + Gram spectrum ===")
for M in (40, 80, 150):
    for ws in (3.0, 6.0, 10.0):
        fmap = RandomFeatureMap(d=1, M=M, activation="tanh", w_scale=ws, b_scale=ws, seed=7)
        prob = Poisson(fmap=fmap, exact=SineProduct((1,)))
        quad = build_reference_quad(dom, 4000, 2, rng, alpha=0.5)
        rs = ResidualSpace(prob, quad, rcond=1e-12)
        res, err = big_uniform_solve(prob, n=4000, ridge=1e-10)
        print(f"M={M:4d} ws={ws:4.1f} | Grank={rs.numerical_rank():3d}/{M} "
              f"kappa(G)={rs.gram_condition_number():.1e} | bigLS rel_l2={err['rel_l2']:.2e} cond={res.cond:.1e}")

print("\n=== (C) solve modes at M=80, ws=6, christoffel sampling ===")
M, ws = 80, 6.0
fmap = RandomFeatureMap(d=1, M=M, activation="tanh", w_scale=ws, b_scale=ws, seed=7)
prob = Poisson(fmap=fmap, exact=SineProduct((1,)))
for alpha in (0.3, 0.5, 0.7):
    quad = build_reference_quad(dom, 4000, 2, rng, alpha=alpha)
    for rcond in (1e-6, 1e-8, 1e-10, 1e-12):
        rs = ResidualSpace(prob, quad, rcond=rcond)
        boundary = S.make_boundary(dom, 60, rng, w_bc=1.0)
        pool = S.build_interior_pool(prob, dom, 8000, rng)
        d = S.rebalance(S.ChristoffelSampler(rs).build(prob, dom, 400, rng, boundary, pool=pool))
        T = rs.whitening()
        res_w = solve(prob, d, whiten=T); err_w = errors(prob, res_w.coeffs, Xt)
        res_r = solve(prob, d, whiten=None, ridge=1e-8); err_r = errors(prob, res_r.coeffs, Xt)
        print(f"alpha={alpha} rcond={rcond:.0e} r={T.shape[1]:3d} | "
              f"whiten rel_l2={err_w['rel_l2']:.2e} cond={res_w.cond:.1e} | "
              f"raw+ridge rel_l2={err_r['rel_l2']:.2e} cond={res_r.cond:.1e}")
