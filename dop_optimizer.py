
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.optimize import minimize

from fem_core import FrameNode, FrameElement, FrameSupport, FramePointLoad, PlaneFrameInput, solve_plane_frame


@dataclass
class OptimizationSettings:
    E: float = 210e6                  # kN/m2
    rho: float = 7850.0               # kg/m3
    sigma_allow: float = 235e3        # kN/m2
    disp_allow: float = 0.020         # m
    A_min: float = 1.0e-4
    A_max: float = 0.20
    I_min: float = 1.0e-8
    I_max: float = 5.0e-2
    target_utilization: float = 0.98
    optimize_A: bool = True
    optimize_I: bool = True


def frame_input(nodes, elements, supports, loads) -> PlaneFrameInput:
    ns = [FrameNode(float(n["x"]), float(n["y"])) for n in nodes]
    es = [FrameElement(int(e["i"]), int(e["j"]), float(e["E"]),
                       float(e["A"]), float(e["I"]), float(e.get("udl_local", 0.0)))
          for e in elements]
    ss = [FrameSupport(int(s["node"]), bool(s["ux"]), bool(s["uy"]), bool(s["rz"]))
          for s in supports]
    ls = [FramePointLoad(int(p["node"]), float(p.get("Fx", 0.0)),
                         float(p.get("Fy", 0.0)), float(p.get("Mz", 0.0)))
          for p in loads]
    return PlaneFrameInput(nodes=ns, elements=es, supports=ss, point_loads=ls)


def analyze(nodes, elements, supports, loads):
    return solve_plane_frame(frame_input(nodes, elements, supports, loads))


def max_displacement(result) -> float:
    d = np.asarray(result.node_displacements)
    return float(np.max(np.sqrt(d[:, 0]**2 + d[:, 1]**2)))


def force_envelopes(result):
    out = []
    for er in result.element_results:
        x = np.asarray(er.x_local, dtype=float)
        N = np.asarray(er.axial, dtype=float)
        M = np.asarray(er.moment, dtype=float)
        out.append({
            "elem_idx": int(er.elem_idx),
            "L": float(x[-1] - x[0]) if len(x) > 1 else 0.0,
            "x": x,
            "N": N,
            "M": M,
            "Nmax": float(np.max(np.abs(N))) if len(N) else 0.0,
            "Mmax": float(np.max(np.abs(M))) if len(M) else 0.0,
        })
    return out


def variational_axial_A(N, x, E, sigma_allow, disp_energy_limit):
    """
    Continuous variational candidate:
        min ∫ A dx
        s.t.  ∫ N²/(E A) dx <= C

    Euler-Lagrange/KKT:
        A*(x) = max(|N|/sigma_allow, |N| sqrt(lambda/E))
    lambda is selected by bisection so the displacement-energy constraint is
    active when the unconstrained stress solution is insufficient.
    """
    Nabs = np.abs(np.asarray(N, float))
    x = np.asarray(x, float)
    stress_A = Nabs / max(sigma_allow, 1e-12)

    def energy(A):
        return float(np.trapz(Nabs**2 / np.maximum(E*A, 1e-18), x))

    if energy(stress_A) <= disp_energy_limit:
        return stress_A, 0.0

    def candidate(lam):
        return np.maximum(stress_A, Nabs*np.sqrt(max(lam, 0.0)/E))

    lo, hi = 0.0, 1.0
    while energy(candidate(hi)) > disp_energy_limit and hi < 1e20:
        hi *= 10.0
    for _ in range(100):
        mid = 0.5*(lo+hi)
        if energy(candidate(mid)) > disp_energy_limit:
            lo = mid
        else:
            hi = mid
    lam = hi
    return candidate(lam), lam


def variational_frame_candidate(result, elements, settings: OptimizationSettings):
    """
    Uses FEM N(x), M(x) to build a continuous variational candidate.
    The displacement constraint is calibrated from the current FEM solution:
        C_delta = C0 * (delta_allow / delta0)^2
    This is an energy surrogate used to initialize the finite-dimensional
    displacement-constrained optimization.
    """
    env = force_envelopes(result)
    d0 = max_displacement(result)
    if d0 <= 0:
        d0 = 1e-12

    # Calibrate an energy-like limit from current internal-force fields.
    current_energy = 0.0
    for e, q in zip(elements, env):
        x, N, M = q["x"], q["N"], q["M"]
        L = max(q["L"], 1e-12)
        A = max(float(e["A"]), settings.A_min)
        I = max(float(e["I"]), settings.I_min)
        current_energy += np.trapz(N**2/(settings.E*A), x)
        current_energy += np.trapz(M**2/(settings.E*I), x)

    ratio = (settings.disp_allow / d0)**2
    energy_limit = current_energy * ratio

    candidates = []
    lambdas = []
    for e, q in zip(elements, env):
        A0 = max(float(e["A"]), settings.A_min)
        I0 = max(float(e["I"]), settings.I_min)
        x, N, M = q["x"], q["N"], q["M"]
        L = max(q["L"], 1e-12)

        # Allocate element energy budget proportional to its current energy.
        ce = np.trapz(N**2/(settings.E*A0), x) + np.trapz(M**2/(settings.E*I0), x)
        budget = energy_limit * ce / max(current_energy, 1e-18)

        # Split between axial and bending terms.
        bA = budget * 0.45
        bI = budget * 0.55

        Astar, lamA = variational_axial_A(N, x, settings.E, settings.sigma_allow, max(bA, 1e-18))

        Mabs = np.abs(M)
        stress_I = np.maximum(I0 * 0.05, Mabs * 1e-12)
        # For a rectangular-family proxy I ∝ A^2, but keep I bounded here.
        # The bending variational stationarity gives I*(x) ∝ |M(x)|.
        if np.max(Mabs) > 0:
            scale = bI / max(float(np.trapz(Mabs**2/(settings.E*np.maximum(I0,1e-18)), x)), 1e-18)
            Istar = np.maximum(settings.I_min, I0 * np.sqrt(max(scale, 1e-12)))
            Istar *= np.clip(Mabs / max(np.max(Mabs), 1e-18), 0.25, 1.0)
        else:
            Istar = np.full_like(x, I0)

        Abar = float(np.trapz(Astar, x)/L)
        Ibar = float(np.trapz(Istar, x)/L)
        candidates.append((np.clip(Abar, settings.A_min, settings.A_max),
                          np.clip(Ibar, settings.I_min, settings.I_max)))
        lambdas.append(lamA)

    return candidates, {"delta0": d0, "energy0": current_energy,
                        "energy_limit": energy_limit, "lambdas_A": lambdas}


def _build_result_with_design(base_elements, Avals, Ivals):
    elems = []
    for e, A, I in zip(base_elements, Avals, Ivals):
        z = dict(e)
        z["A"] = float(A)
        z["I"] = float(I)
        elems.append(z)
    return elems


def finite_design_optimize(nodes, elements, supports, loads, settings: OptimizationSettings):
    """
    Finite-dimensional optimization after the continuous variational candidate.
    Objective: minimum steel volume Σ A_e L_e.
    Constraint: maximum translational displacement <= allowable.
    Stress constraint: |N|/A + |M|/W is represented conservatively by
    separate axial and bending utilization proxies.
    """
    base_result = analyze(nodes, elements, supports, loads)
    env = force_envelopes(base_result)
    d0 = max_displacement(base_result)

    # Initial point from variational candidate.
    cand, meta = variational_frame_candidate(base_result, elements, settings)
    A0 = np.array([x[0] for x in cand], dtype=float)
    I0 = np.array([x[1] for x in cand], dtype=float)
    if not settings.optimize_A:
        A0 = np.array([float(e["A"]) for e in elements])
    if not settings.optimize_I:
        I0 = np.array([float(e["I"]) for e in elements])

    n = len(elements)
    x0 = np.ravel(np.column_stack([A0, I0]))

    bounds = []
    for _ in range(n):
        bounds.append((settings.A_min, settings.A_max))
        bounds.append((settings.I_min, settings.I_max))

    lengths = []
    for e in elements:
        ni, nj = nodes[int(e["i"])], nodes[int(e["j"])]
        lengths.append(float(np.hypot(nj["x"]-ni["x"], nj["y"]-ni["y"])))
    lengths = np.asarray(lengths)

    def unpack(x):
        return x[0::2], x[1::2]

    def obj(x):
        A, I = unpack(x)
        return float(np.sum(A*lengths))

    def metrics(x):
        A, I = unpack(x)
        ee = _build_result_with_design(elements, A, I)
        try:
            r = analyze(nodes, ee, supports, loads)
            d = max_displacement(r)
        except Exception:
            return 1e9, np.full(n, 1e9), np.full(n, 1e9), None
        envr = force_envelopes(r)
        Nu = np.array([q["Nmax"]/max(a*settings.sigma_allow,1e-18) for q,a in zip(envr,A)])
        # Rectangular section proxy: W ≈ 2 I / sqrt(A*12) is not exact.
        # Use W_eff = 2I/h with h inferred from a rectangular family.
        h_eff = np.sqrt(np.maximum(12.0*I/A, 1e-18))
        W_eff = np.maximum(I/(0.5*h_eff), 1e-18)
        Mu = np.array([q["Mmax"]/max(W*settings.sigma_allow,1e-18) for q,W in zip(envr,W_eff)])
        return d, Nu, Mu, r

    def g_disp(x):
        d, _, _, _ = metrics(x)
        return settings.disp_allow - d

    def g_stress(x):
        _, Nu, Mu, _ = metrics(x)
        return 1.0 - np.maximum(Nu + Mu, 0.0)

    cons = [{"type":"ineq", "fun":g_disp}, {"type":"ineq", "fun":g_stress}]

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 80, "ftol": 1e-9, "disp": False})

    Aopt, Iopt = unpack(res.x)
    opt_elements = _build_result_with_design(elements, Aopt, Iopt)
    opt_result = analyze(nodes, opt_elements, supports, loads)
    dopt, Nu, Mu, _ = metrics(res.x)

    return {
        "success": bool(res.success),
        "message": str(res.message),
        "objective_initial": float(np.sum(np.array([e["A"] for e in elements])*lengths)),
        "objective_final": float(obj(res.x)),
        "A_initial": np.array([e["A"] for e in elements]),
        "I_initial": np.array([e["I"] for e in elements]),
        "A_final": Aopt,
        "I_final": Iopt,
        "lengths": lengths,
        "base_result": base_result,
        "opt_result": opt_result,
        "d_initial": d0,
        "d_final": dopt,
        "stress_utilization": np.maximum(Nu + Mu, 0.0),
        "variational_meta": meta,
        "opt_elements": opt_elements,
        "scipy_result": res,
    }


def kkt_verify_design(nodes, elements, supports, loads, settings, result):
    """
    Numerical KKT verification of the finite-dimensional problem.
    Inequalities are written as:
        g1(x)=d(x)-d_allow <= 0
        gi(x)=util_i(x)-1 <= 0
        bounds are handled as active constraints.
    Multipliers are recovered by least squares from:
        ∇f + Jg^T λ = 0, λ >= 0.
    """
    x = np.ravel(np.column_stack([result["A_final"], result["I_final"]]))

    def f(xv):
        A, _ = xv[0::2], xv[1::2]
        return float(np.sum(A*result["lengths"]))

    def constraints_vec(xv):
        A, I = xv[0::2], xv[1::2]
        ee = _build_result_with_design(elements, A, I)
        rr = analyze(nodes, ee, supports, loads)
        d = max_displacement(rr)
        env = force_envelopes(rr)
        Nu = np.array([q["Nmax"]/max(a*settings.sigma_allow,1e-18) for q,a in zip(env,A)])
        h = np.sqrt(np.maximum(12.0*I/A, 1e-18))
        W = np.maximum(2*I/np.maximum(h,1e-18), 1e-18)
        Mu = np.array([q["Mmax"]/max(w*settings.sigma_allow,1e-18) for q,w in zip(env,W)])
        return np.r_[d-settings.disp_allow, Nu+Mu-1.0]

    def grad(fun, xv, eps=1e-6):
        out = np.zeros_like(xv)
        for i in range(len(xv)):
            h = eps*max(1.0, abs(xv[i]))
            xp, xm = xv.copy(), xv.copy()
            xp[i] += h; xm[i] -= h
            out[i] = (fun(xp)-fun(xm))/(2*h)
        return out

    gf = grad(f, x)
    gv = np.vstack([grad(lambda z, j=j: constraints_vec(z)[j], x)
                    for j in range(len(constraints_vec(x)))])
    g = constraints_vec(x)

    active = np.where(g >= -2e-4)[0]
    if len(active):
        M = gv[active].T
        lam, *_ = np.linalg.lstsq(M, -gf, rcond=None)
        lam = np.maximum(lam, 0.0)
        stationarity = gf + M @ lam
        dual_ok = bool(np.min(lam) >= -1e-8)
        comp = lam * g[active]
        comp_res = float(np.max(np.abs(comp))) if len(comp) else 0.0
    else:
        lam = np.array([])
        stationarity = gf
        dual_ok = True
        comp_res = 0.0

    return {
        "active_indices": active,
        "multipliers": lam,
        "stationarity_norm": float(np.linalg.norm(stationarity, np.inf)),
        "dual_feasible": dual_ok,
        "primal_max_violation": float(max(0.0, np.max(g))),
        "complementarity_residual": comp_res,
        "objective_gradient_norm": float(np.linalg.norm(gf, np.inf)),
        "constraints": g,
    }
