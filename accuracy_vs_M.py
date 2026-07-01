"""Legacy diagnostic: relative L2 error vs #features M at fixed N.

This script separates accuracy from conditioning. Direct full-space SVD with many
uniform rows can be very accurate in some small diagnostics even when the raw system
is severely ill-conditioned. The manuscript's main evidence therefore comes from
conditioning, LSQR, multiseed, and feature-capacity-controlled experiments rather
than from this diagnostic alone.
"""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src")))
import numpy as np
from rfm.features import RandomFeatureMap
from rfm.domains import Interval, Rectangle
from rfm.operators import Poisson, SineProduct
from rfm.christoffel import build_reference_quad, ResidualSpace
from rfm import samplers as S
from rfm.solver import solve, errors
from rfm.plotting import new_ax, plot_series, save
from rfm.utils import save_json, results_dir, figures_dir


def run(dim, Ms, N, n_bd, w_scale, qsize, seeds, tag):
    if dim == 1:
        dom, exact, d = Interval(0, 1), SineProduct((2,)), 1
    else:
        dom, exact, d = Rectangle(0, 1, 0, 1), SineProduct((2, 2)), 2
    Xt = (np.linspace(0, 1, 1000).reshape(-1, 1) if dim == 1
          else np.column_stack([m.ravel() for m in np.meshgrid(np.linspace(.02, .98, 300),
                                                               np.linspace(.02, .98, 300))]))
    out = {"M": list(Ms), "uniform_raw": [], "christoffel": [], "greedy": [], "rank": []}
    for M in Ms:
        fmap = RandomFeatureMap(d=d, M=M, activation="tanh", w_scale=w_scale, b_scale=w_scale, seed=7)
        prob = Poisson(fmap=fmap, exact=exact)
        rq = build_reference_quad(dom, qsize, max(2, qsize // 10), np.random.default_rng(1))
        rs = ResidualSpace(prob, rq); T = rs.whitening()
        bdry = S.make_boundary(dom, n_bd, np.random.default_rng(2), w_bc=1.0)
        eu, ec = [], []
        for sd in seeds:
            r = np.random.default_rng(3000 + sd)
            pool = S.build_interior_pool(prob, dom, max(20 * N, 4000), r)
            du = S.UniformSampler().build(prob, dom, N, r, bdry)
            eu.append(errors(prob, solve(prob, S.rebalance(du), whiten=None, lstsq_rcond=1e-14).coeffs, Xt)["rel_l2"])
            dc = S.ChristoffelSampler(rs).build(prob, dom, N, r, bdry, pool=pool)
            ec.append(errors(prob, solve(prob, S.rebalance(dc), whiten=T).coeffs, Xt)["rel_l2"])
        rg = np.random.default_rng(7)
        poolg = S.build_interior_pool(prob, dom, min(max(20 * N, 4000), 8000), rg)
        dg = S.GreedySampler(rs, lam=1e-6).build(prob, dom, N, rg, bdry, pool=poolg)
        eg = errors(prob, solve(prob, S.rebalance(dg), whiten=T).coeffs, Xt)["rel_l2"]
        out["uniform_raw"].append(float(np.median(eu)))
        out["christoffel"].append(float(np.median(ec)))
        out["greedy"].append(float(eg)); out["rank"].append(rs.numerical_rank())
        print(f"[acc_vs_M {tag}] M={M:4d} rank={rs.numerical_rank():3d} "
              f"uRAW={np.median(eu):.2e} chri={np.median(ec):.2e} gree={eg:.2e}", flush=True)
    fig, ax = new_ax("# features M", "relative L2 error", f"{tag}: accuracy vs features (N={N})")
    plot_series(ax, Ms, out["uniform_raw"], "uniform_raw")
    plot_series(ax, Ms, out["christoffel"], "christoffel")
    plot_series(ax, Ms, out["greedy"], "greedy")
    ax.legend(fontsize=8); save(fig, os.path.join(figures_dir(), f"acc_vs_M_{tag}.png"))
    return out


def main():
    t0 = time.time(); seeds = list(range(6))
    res = {}
    res["poisson1d"] = run(1, [20, 40, 80, 160, 320, 640], N=600, n_bd=2, w_scale=6.0, qsize=4000, seeds=seeds, tag="poisson1d")
    res["poisson2d"] = run(2, [40, 80, 160, 320, 640], N=1200, n_bd=300, w_scale=5.0, qsize=8000, seeds=seeds, tag="poisson2d")
    res["wall_s"] = time.time() - t0
    save_json(os.path.join(results_dir(), "accuracy_vs_M.json"), res)
    print(f"[acc_vs_M] done in {res['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
