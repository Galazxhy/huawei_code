"""Multistart bounded conditional optimization; no global certificate.

Budget-preserving coordinates keep evaluations feasible without assuming
monotonic Loss in D. Every reduced training hull includes the reference p.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.optimize import brentq, minimize


class EvaluationLimit(RuntimeError):
    pass


from q3_model_contract import ContractConfig, Q3ModelContract


@dataclass(frozen=True)
class OptimizerConfig:
    n_bounds: tuple[float, float] = (1e6, 1e11)
    d_bounds: tuple[float, float] = (1e9, 1e13)
    anchor_count: int = 24
    starts: int = 3
    maxiter: int = 450
    seed: int = 20260924
    q0_override: float | None = None
    normalize_cost_endpoint: bool = False
    solver: str = "SLSQP"
    max_model_evaluations_per_start: int | None = None


def farthest_anchors(points, count):
    if not 1 <= count <= len(points):
        raise ValueError("anchor_count outside training row count")
    chosen = [int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
    distance = np.full(len(points), np.inf)
    while len(chosen) < count:
        distance = np.minimum(distance, np.linalg.norm(points - points[chosen[-1]], axis=1))
        distance[chosen] = -1
        chosen.append(int(np.argmax(distance)))
    return np.vstack([points.mean(axis=0), points[chosen]])


class JointOptimizer:
    def __init__(self, config=None, bridge_mode="ratio", kappa=1.0):
        self.config = config or OptimizerConfig()
        if self.config.starts < 2 or self.config.maxiter < 1:
            raise ValueError("at least two starts and positive maxiter are required")
        if self.config.solver not in ("SLSQP", "COBYQA"):
            raise ValueError("unsupported solver")
        if self.config.max_model_evaluations_per_start is not None and self.config.max_model_evaluations_per_start < 1:
            raise ValueError("model evaluation limit must be positive")
        self.model = Q3ModelContract(ContractConfig(
            quality_bridge_mode=bridge_mode, quality_bridge_kappa=kappa,
            n_bounds=self.config.n_bounds, d_bounds=self.config.d_bounds,
            q0_override=self.config.q0_override))
        self.anchors = farthest_anchors(self.model.support_points, self.config.anchor_count)
        self.p_reference = self.model.support_points.mean(axis=0)

    def solve_mode(self, budget, context, form, mode="joint_reduced_hull"):
        if mode not in ("joint_reduced_hull", "fixed_p_q_baseline", "fixed_p", "fixed_q"):
            raise ValueError("unknown decision mode")
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError("budget must be finite and positive")
        model = self.model
        model._validate_context(context)
        if form not in model.quality_cost_payload["forms"]:
            raise ValueError("unknown cost form")
        if self.config.normalize_cost_endpoint:
            # Remove the prior scale first; one optimizer may serve several forms.
            from dataclasses import replace
            model.config = replace(model.config, quality_cost_scale=1.0)
            reference = model._g(1., "exponential") - model._g(model.q0, "exponential")
            current = model._g(1., form) - model._g(model.q0, form)
            if current <= 0:
                raise ValueError("cannot normalize zero endpoint quality increment")
            model.config = replace(model.config, quality_cost_scale=reference / current)
        n_lo, n_hi = self.config.n_bounds
        d_lo, d_hi = self.config.d_bounds
        a = 6 + model.config.eta_attention * context
        if budget <= a * n_lo * d_lo:
            raise ValueError("budget has no interior feasible point within scenario bounds")
        h = lambda q: max(model._g(q, form) - model._g(model.q0, form), 0.0)
        q_hi = 1.0
        if d_lo * (a * n_lo + h(q_hi)) > budget:
            q_hi = brentq(lambda q: d_lo * (a * n_lo + h(q)) - budget, model.q0, 1.0)
        free_p = mode in ("joint_reduced_hull", "fixed_q")
        free_q = mode in ("joint_reduced_hull", "fixed_p")
        weight_count = len(self.anchors) if free_p else 0
        bounds = [(0., 1.), (0., 1.), (model.q0, q_hi if free_q else model.q0)]
        bounds += [(0., 1.)] * weight_count

        def unpack(x):
            q = float(x[2])
            n_cap = min(n_hi, max(n_lo, (budget / d_lo - h(q)) / a))
            n = float(np.exp(np.log(n_lo) + x[0] * np.log(n_cap / n_lo)))
            d_cap = min(d_hi, max(d_lo, budget / (a * n + h(q))))
            d = float(np.exp(np.log(d_lo) + x[1] * np.log(d_cap / d_lo)))
            if free_p:
                w = np.maximum(x[3:], 0)
                w = w / w.sum() if w.sum() > 0 else np.eye(1, weight_count)[0]
                p = w @ self.anchors
            else:
                p = self.p_reference
            return n, d, q, p

        evaluation_cache = {}
        count = 0
        last_evaluation, best_evaluation = None, None

        def losses(x):
            nonlocal count, last_evaluation, best_evaluation
            key = np.asarray(x, dtype=float).tobytes()
            if key in evaluation_cache:
                return evaluation_cache[key]
            if self.config.max_model_evaluations_per_start is not None and count >= self.config.max_model_evaluations_per_start:
                raise EvaluationLimit("registered model evaluation allowance exhausted")
            n, d, q, p = unpack(x)
            value = model.predict_losses(n, d, p, q)[0]
            last_evaluation = {"vector": np.asarray(x).tolist(), "N_params":n,"D_tokens":d,"Q":q,"p":p.tolist(),
                               "objective":float(value.mean()) if np.isfinite(value).all() else None,
                               "loss_positive":bool(np.isfinite(value).all() and min(value)>1e-6)}
            if last_evaluation["loss_positive"] and (best_evaluation is None or last_evaluation["objective"]<best_evaluation["objective"]):
                best_evaluation = dict(last_evaluation)
            evaluation_cache[key] = value
            count += 1
            return value

        def objective(x):
            values = losses(x)
            return float(np.mean(values)) if np.isfinite(values).all() else 1e6

        constraints = [{"type": "ineq", "fun": lambda x: losses(x) - 1e-8}]
        if free_p:
            constraints.append({"type": "eq", "fun": lambda x: np.sum(x[3:]) - 1})
        rng = np.random.default_rng(self.config.seed)
        starts = []
        for i in range(self.config.starts):
            if i == 0:
                w = np.eye(1, weight_count)[0] if free_p else np.array([])
                start = np.r_[0.5, 0.95, model.q0, w]
            else:
                w = rng.dirichlet(np.ones(weight_count)) if free_p else np.array([])
                start = np.r_[rng.uniform(.1, .9), rng.uniform(.5, 1),
                              rng.uniform(model.q0, q_hi) if free_q else model.q0, w]
            starts.append(start)
        attempts, candidates = [], []
        for i, start in enumerate(starts):
            evaluation_cache.clear()
            count = 0
            last_evaluation, best_evaluation = None, None
            started = time.perf_counter()
            options = {"maxiter": self.config.maxiter, "disp": False}
            if self.config.solver == "SLSQP":
                options["ftol"] = 1e-9
            else:
                options.update(feasibility_tol=1e-9, final_tr_radius=1e-7)
            try:
                fit = minimize(objective, start, method=self.config.solver, bounds=bounds,
                               constraints=constraints, options=options)
            except EvaluationLimit as exc:
                attempts.append({"start": i, "success": False, "accepted": False,
                                 "message": str(exc), "initial_vector": start.tolist(),
                                 "last_evaluation": last_evaluation, "best_evaluation": best_evaluation,
                                 "diagnostic_only_not_converged": True,
                                 "model_evaluations": count, "elapsed_seconds": time.perf_counter()-started})
                continue
            n, d, q, p = unpack(fit.x)
            attempt = {"start": i, "success": bool(fit.success), "message": str(fit.message),
                       "iterations": int(fit.nit), "objective": float(fit.fun),
                       "initial_vector": start.tolist(), "final_vector": fit.x.tolist(),
                       "N_params": n, "D_tokens": d, "Q": q, "p": p.tolist(),
                       "function_evaluations": int(fit.nfev), "model_evaluations": count,
                       "elapsed_seconds": time.perf_counter()-started}
            try:
                output = model.evaluate(n, d, p, q, context, form, budget_flops=budget)
                if min(output["loss_by_domain"].values()) < 1e-6:
                    raise ValueError("candidate exploits positivity boundary of predictor")
                if not fit.success:
                    raise ValueError("solver did not converge; feasible point is not a solution")
                if free_p and abs(np.sum(fit.x[3:]) - 1) > 1e-7:
                    raise ValueError("weight simplex equality did not converge")
                attempt["accepted"] = True
                candidates.append((output, i))
            except ValueError as exc:
                attempt.update(accepted=False, rejection=str(exc))
            attempts.append(attempt)
        if not candidates:
            raise RuntimeError(f"no converged admissible candidate: {attempts}")
        output, best_start = min(candidates, key=lambda pair: pair[0]["mean_loss"])
        best_x = np.asarray(attempts[best_start]["final_vector"])
        perturbations = []
        # Feasible directions check the actual piecewise budget map, including its corners.
        for step in (1e-4, 1e-3):
            trials = []
            for axis in range(3):
                for sign in (-1, 1):
                    x = best_x.copy()
                    x[axis] = np.clip(x[axis] + sign*step, *bounds[axis])
                    if not np.array_equal(x, best_x):
                        trials.append(x)
            if free_p:
                for index in range(weight_count):
                    x = best_x.copy()
                    x[3:] *= 1-step
                    x[3+index] += step
                    trials.append(x)
            gains = []
            for x in trials:
                tn, td, tq, tp = unpack(x)
                value = model.predict_losses(tn, td, tp, tq)[0]
                if np.isfinite(value).all() and min(value) >= 1e-6:
                    gains.append(output["mean_loss"] - float(value.mean()))
            perturbations.append({"step": step, "directions_checked": len(gains),
                                  "largest_loss_improvement": max([0.] + gains)})
        active = []
        for name, val, lo, hi in (("N", output["N_params"], n_lo, n_hi),
                                  ("D", output["D_tokens"], d_lo, d_hi),
                                  ("Q", output["Q"], model.q0, 1.0)):
            if np.isclose(val, lo, rtol=1e-5):
                active.append(name + "_lower")
            if np.isclose(val, hi, rtol=1e-5):
                active.append(name + "_upper")
        if abs(output["budget_margin_flops"]) / budget < 1e-6:
            active.append("budget")
        cost = output["costs"]
        n, d, q, p = output["N_params"], output["D_tokens"], output["Q"], np.array(output["p"])
        dq = min(1e-5, (1. - model.q0) / 10)
        q_left, q_right = max(model.q0, q - dq), min(1., q + dq)
        j_q = ((model.predict_losses(n, d, p, q_right)[0].mean()
                - model.predict_losses(n, d, p, q_left)[0].mean()) / (q_right - q_left)
               if q_right > q_left else 0.)
        j_d = (model.predict_losses(n, d * (1 + 1e-5), p, q)[0].mean()
               - model.predict_losses(n, d * (1 - 1e-5), p, q)[0].mean()) / (2e-5 * d)
        quality_direction = j_q - j_d * d * model.g_prime(q, form) / (a * n + h(q))
        output.update({
            "optimization_mode": mode, "solver": self.config.solver, "solver_success": True,
            "best_start": best_start, "attempts": attempts, "optimizer_config": asdict(self.config),
            "solution_claim": "best converged local candidate; no global certificate",
            "anchor_count_including_reference": len(self.anchors) if free_p else 0,
            "anchor_training_row_positions": [-1] + [int(np.argmin(np.linalg.norm(
                model.support_points - row, axis=1))) for row in self.anchors[1:]] if free_p else [],
            "anchor_minus_one_means": "training reference mean; other IDs are zero-based A4 row positions",
            "reduced_hull_not_full_training_hull": free_p,
            "active_constraints": active,
            "boundary_sensitive": any(x in active for x in ("N_lower", "N_upper", "D_lower", "D_upper")),
            "quality_budget_directional_derivative": float(quality_direction) if q_right > q_left else None,
            "quality_direction_valid_for_fixed_N_p": q_right > q_left and "budget" in active and not
                any(x in active for x in ("D_lower", "D_upper")),
            "feasible_perturbations": perturbations,
            "feasible_perturbation_tolerance": 1e-6,
            "feasible_perturbation_pass": all(x["largest_loss_improvement"] <= 1e-6 for x in perturbations),
            "stationarity_scope": "finite feasible directional probes; not a full KKT or global certificate",
            "total_model_evaluations": sum(x["model_evaluations"] for x in attempts),
            "model_evaluation_count_scope": "distinct optimizer response-kernel calls within each start; excludes verification probes, final validation and derivative diagnostics",
            "cost_shares": {key: cost[key] / cost["C_total_flops"] for key in
                            ("C_train_flops", "C_attn_flops", "C_quality_flops")},
        })
        return output

    def solve(self, budget, context, form, include_baseline=True):
        modes = ["joint_reduced_hull", "fixed_p_q_baseline"] if include_baseline else ["joint_reduced_hull"]
        return [self.solve_mode(budget, context, form, mode) for mode in modes]
