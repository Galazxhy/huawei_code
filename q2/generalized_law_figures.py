#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Output the figure set for the compact generalized scaling law (English labels).

Model under test
----------------
::

    L_k(N, D, p, Q) = E_k + A_k * N_b^(-alpha) + B_k * D_b^(-beta)
                      + S_k(N, D) * Phi_k(p, Q)

    S_k(N, D)  = (N_b / N_ref)^(-eta_k) * (D_b / D_ref)^(-zeta_k)
    Phi_k(p,Q) = slope_k * [ Ridge_k((X - mean) / scale) - center_k ]
    X          = [ p_i * Q_i (17) , clr_eps(p)_i (17) ]

Units: ``N_b``/``D_b`` in **billions**; ``L`` in nats; ``p`` on the 17-dim simplex.

Reader-facing figures (default ``--figures story``)
-------------------------------------------------
========================  ====================================================
``quality_pathway``       fixed Q, observed mean shares, and product pQ
``observed_contrast``     a real held-out mixture change and its 13 losses
``component_breakdown``  pQ and clr contributions to that same loss change
``domain_scale``          domain-specific mixture amplitudes at fitted anchors
``surface_parity``        held-out parity, Chinese labels; rendered by the same
                        builder as ``identification_parity``
``all_domain_p_q``        p and Q response across every source and loss domain
========================  ====================================================

Additional diagnostic figures (``--figures all``)
-----------------------------------------------
========================  ====================================================
``loss_vs_diversity``      loss against mixture diversity, one curve per scale
``response_parity``       predicted vs observed mixture contribution
``error_by_scale``        relative RMSE / median APE per held-out scale
``amplitude_decay``       S_k across the three observed anchors (log y)
``mixture_response``      Phi_k(p) against the arxiv share, 5 domains
``weight_heatmap``        effective Wp_(k,i), 13 evaluation x 17 mixture
``structure_diagnostics`` why one shared scalar cannot fit 13 domains
``ridge_ladder``          tuning relative RMSE against the Ridge strength
``elasticity``            dlnL/dlnN, dlnL/dlnD, dlnL/dlnQ across two laws
``epsilon_sensitivity``   epsilon sweep, error and mixture R^2 together
``domain_error_1b``       per-domain relative RMSE at the 1B anchor
``identification_parity``   held-out parity (Chinese), beside the per-scale error
                           against the same coefficients with the mixture term
                           switched off; ``surface_parity`` is identical
``domain_accuracy``        per-domain held-out relative RMSE (Chinese) and its R^2,
                           split into the overall reading and the mixture-only one
``identification_resolution``  per-domain held-out mixture R^2 and the identified
                           (eta_k, zeta_k) pair per domain
``separate_p_q``          composition and quality one-variable response cuts
``simplex_landscape``     quality-labelled composition/loss landscapes
``scaling_curves``        loss--compute curves for fixed observed mixtures
``four_factor_map``       N, D, Q, p in one aligned response matrix
========================  ====================================================

Conventions: English labels and no figure titles. The reader-facing figures use
larger canvases where 17 mixture sources or 13 evaluation domains must be read.
The pQ block is a quality-weighted *composition feature*, not an independently
identified quality effect: each Q_i is fixed for its source in these records.

Every number is recomputed from the same sources the law was fitted from, and the
held-out metrics printed by ``--verify`` must reproduce
``data_analysis/Q_1_to_2/generalized_scaling_compact.json``; a mismatch is raised
as an error rather than silently drawn.

Usage::

    MPLCONFIGDIR=$PWD/.mplcfg python q2/generalized_law_figures.py
    MPLCONFIGDIR=$PWD/.mplcfg python q2/generalized_law_figures.py \\
        --figures all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter, NullFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generalized_law as compact  # noqa: E402

COMPACT_JSON = PROJECT_ROOT / "data_analysis/Q_1_to_2/generalized_scaling_compact.json"
HANDOFF_DIR = PROJECT_ROOT / "data_analysis/Q2_to_Q3"
CLASSIC_JSON = PROJECT_ROOT / "data_analysis/scaling_law_full/scaling_law_full_results.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_analysis/generalized_scaling_law_drawings"

PANEL_FIGSIZE = (6.0, 4.0)

# One hue and one marker per observed scale, so the three scales stay separable
# even where the point clouds overlap.
SCALE_STYLE = {
    "test_1m": ("#4C78A8", "o"),
    "test_60m": ("#F58518", "s"),
    "test_1b": ("#54A24B", "^"),
}
BAR_COLOR = "#4C78A8"
BAR_COLOR_ALT = "#7BA7CC"
BAR_COLOR_MUTED = "#BFBFBF"
ACCENT = "#C44E52"

SCALE_LABEL = {
    "test_1m": "1M",
    "test_60m": "60M",
    "test_1b": "1B",
}
HELDOUT_SCALES = ("test_1m", "test_60m", "test_1b")

# Relative importance of the two tasks: 1e9 in the text below refers to the
# anchor definitions used throughout the report.
ANCHORS = (
    ("1M", 1e6, 1e9),
    ("60M", 6e7, 1e9),
    ("1B", 1e9, 2.5e10),
)


# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
def configure_fonts() -> str:
    """Use a plain sans font; the figure text is ASCII, so no CJK asset is needed."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
            "axes.unicode_minus": False,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "legend.frameon": False,
        }
    )
    return plt.rcParams["font.sans-serif"][0]


def tidy(axis: plt.Axes, *, grid_axis: str = "both") -> None:
    axis.grid(True, axis=grid_axis, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def note(axis: plt.Axes, text: str, *, loc: str = "upper left",
         size: float = 9.0) -> None:
    """Put a short caveat in a corner of the axes (never a title)."""

    positions = {
        "upper left": (0.03, 0.97, "left", "top"),
        "upper right": (0.97, 0.97, "right", "top"),
        "lower left": (0.03, 0.03, "left", "bottom"),
        "lower right": (0.97, 0.03, "right", "bottom"),
        "center right": (0.97, 0.55, "right", "center"),
    }
    x, y, ha, va = positions[loc]
    axis.text(x, y, text, transform=axis.transAxes, ha=ha, va=va, fontsize=size,
              linespacing=1.45, zorder=6,
              bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.88,
                    "boxstyle": "round,pad=0.35"})


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class LawData:
    """Everything the panels need, recomputed once from the original sources."""

    def __init__(self) -> None:
        self.blocks, self.train_names, self.loss_names, self.q = compact.load_data(
            compact.REGMIX, compact.HANDOFF)
        # Seed 2025 is compact.run()'s default, i.e. the split behind every stored
        # number in generalized_scaling_compact.json.
        self.train_blocks, self.tune_blocks, self.test_blocks = compact.partition(
            self.blocks, 2025)
        self.payload = json.loads(COMPACT_JSON.read_text(encoding="utf-8"))
        self.law = self.payload["compositional_scaling_law"]
        self.params = self.law["parameters"]
        self.mixture = json.loads(
            (HANDOFF_DIR / "mixture_response.json").read_text(encoding="utf-8"))

        self.alpha = float(self.params["alpha"])
        self.beta = float(self.params["beta"])
        self.epsilon = float(self.law["epsilon"])
        self.reference_n = float(self.law["reference_scale"]["N"]) / 1e9
        self.reference_d = float(self.law["reference_scale"]["D"]) / 1e9

        order = list(self.loss_names)
        self.ridge_intercept = np.asarray(
            [self.params["ridge_intercept"][d] for d in order])
        self.ridge_coefficients = np.asarray(
            [[self.params["ridge_coefficients"][d][f]
              for f in self.mixture["feature_names"]] for d in order])
        self.response_center = np.asarray(
            [self.params["reference_response_center"][d] for d in order])
        self.response_slope = np.asarray(
            [self.params["reference_response_slope"][d] for d in order])
        self.e_k = np.asarray([self.params["E"][d] for d in order])
        self.a_k = np.asarray([self.params["A"][d] for d in order])
        self.b_k = np.asarray([self.params["B"][d] for d in order])
        self.eta = np.asarray([self.params["eta"][d] for d in order])
        self.zeta = np.asarray([self.params["zeta"][d] for d in order])
        self.effective_pq = np.asarray(
            [[self.params["effective_weight_pQ"][d][t] for t in self.train_names]
             for d in order])

    # -- model ------------------------------------------------------------- #
    def _features(self, p: np.ndarray, quality_scale: float = 1.0) -> np.ndarray:
        log_p = np.log(p + self.epsilon)
        return np.column_stack((p * (self.q * quality_scale)[None, :],
                                log_p - log_p.mean(axis=1, keepdims=True)))

    def mix_response(self, p: np.ndarray, quality_scale: float = 1.0) -> np.ndarray:
        """``Phi_k(p)``, shape ``(n, 13)``; ``quality_scale`` rescales every Q_i."""

        p = np.atleast_2d(np.asarray(p, dtype=float))
        z = (self._features(p, quality_scale) - self.mixture["feature_mean"]) \
            / self.mixture["feature_scale"]
        raw = self.ridge_intercept[None, :] + z @ self.ridge_coefficients.T
        return (raw - self.response_center[None, :]) * self.response_slope[None, :]

    def amplitude(self, n: float, d: float) -> np.ndarray:
        n_b, d_b = n / 1e9, d / 1e9
        return (n_b / self.reference_n) ** (-self.eta) \
            * (d_b / self.reference_d) ** (-self.zeta)

    def baseline(self, n: float, d: float) -> np.ndarray:
        n_b, d_b = n / 1e9, d / 1e9
        return (self.e_k + self.a_k * n_b ** (-self.alpha)
                + self.b_k * d_b ** (-self.beta))

    def predict(self, n: float, d: float, p: np.ndarray,
                quality_scale: float = 1.0) -> np.ndarray:
        """Vectorised law evaluation.

        ``generalized_law_evaluator.predict`` only accepts a single mixture
        vector; the panels need whole blocks at once.
        """

        return self.baseline(n, d)[None, :] \
            + self.amplitude(n, d)[None, :] * self.mix_response(p, quality_scale)

    def per_scale(self, blocks: list) -> dict[str, dict[str, float]]:
        out = {}
        for block in blocks:
            predicted = self.predict(block.n, block.d, block.p)
            out[block.name] = {
                "relative_rmse": compact.relative_rmse(block.y, predicted),
                "median_ape": float(np.median(np.abs(predicted / block.y - 1))),
            }
        return out

    def per_domain(self, block) -> dict[str, dict[str, float]]:
        predicted = self.predict(block.n, block.d, block.p)
        return {
            name: {
                "relative_rmse": compact.relative_rmse(block.y[:, k], predicted[:, k]),
                "median_ape": float(np.median(np.abs(predicted[:, k] / block.y[:, k] - 1))),
            }
            for k, name in enumerate(self.loss_names)
        }


def classic_elasticities(n: float, d: float) -> dict[str, float]:
    """Analytic elasticities of ``L = E + A N^-alpha + B D^-beta`` at (n, d)."""

    payload = json.loads(CLASSIC_JSON.read_text(encoding="utf-8"))
    fit = payload["B1_main_fit"]["parameters"]
    a_term = fit["parameter_coefficient"] * n ** (-fit["parameter_exponent"])
    b_term = fit["data_coefficient"] * d ** (-fit["data_exponent"])
    total = fit["irreducible_loss"] + a_term + b_term
    return {
        "dlnL_dlnN": -fit["parameter_exponent"] * a_term / total,
        "dlnL_dlnD": -fit["data_exponent"] * b_term / total,
        "loss": float(total),
    }


def quality_law_elasticities(n: float, d: float, q: float) -> dict[str, float]:
    """Analytic elasticities of ``L = E + A N^-alpha + B (D Q)^-beta`` (B7)."""

    payload = json.loads(
        (HANDOFF_DIR / "elasticity_and_substitution.json").read_text(encoding="utf-8"))
    fit = payload["quality_law_from_B6_B7"]["B7"]
    a_term = fit["A"] * n ** (-fit["alpha"])
    b_term = fit["B"] * (d * q) ** (-fit["beta"])
    total = fit["E"] + a_term + b_term
    return {
        "dlnL_dlnN": -fit["alpha"] * a_term / total,
        "dlnL_dlnD": -fit["beta"] * b_term / total,
        "dlnL_dlnQ": -fit["beta"] * b_term / total,
        "loss": float(total),
    }


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def _entropy(p: np.ndarray) -> np.ndarray:
    """Mixture entropy ``H(p) = -sum_i p_i log p_i``, in nats. Not a model input."""

    p = np.atleast_2d(np.asarray(p, dtype=float))
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)


def panel_loss_vs_diversity(data: LawData) -> plt.Figure:
    """The generalized law as a family of loss-vs-diversity curves.

    Entropy is computed straight from the recorded shares, so it is not a model
    input: the curves are the law's *implied* relation along the diversity axis,
    obtained by averaging its prediction over the real mixtures in each bin.
    """

    # One group per observed (N, D); train_1m and test_1m share the 1M anchor.
    groups = []
    for label, name in (("1M", "test_1m"), ("60M", "test_60m"), ("1B", "test_1b")):
        members = [b for b in data.blocks if b.name in
                   {"1M": ("train_1m", "test_1m"), "60M": ("test_60m",),
                    "1B": ("test_1b",)}[label]]
        p = np.vstack([b.p for b in members])
        y = np.vstack([b.y for b in members]).mean(axis=1)
        n, d = members[0].n, members[0].d
        groups.append((label, name, n, d, p, y))

    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE, layout="constrained")
    support_edge, spreads = 0.0, []
    for label, name, n, d, p, y in groups:
        colour, marker = SCALE_STYLE[name]
        h = _entropy(p)
        support_edge = max(support_edge, float(h.max()))
        spreads.append((label, float(y.max() - y.min()), h.size))
        predicted = data.predict(n, d, p).mean(axis=1)
        axis.scatter(h, y, s=9, alpha=0.32, color=colour, marker=marker,
                     edgecolors="none", zorder=2)
        # Equal-count bins keep every segment backed by the same number of rows;
        # the sparser 1B block would otherwise produce a saw-tooth line.
        order = np.argsort(h)
        chunks = np.array_split(order, int(np.clip(h.size // 20, 4, 15)))
        bx = np.array([h[c].mean() for c in chunks])
        by = np.array([predicted[c].mean() for c in chunks])
        axis.plot(bx, by, color=colour, linewidth=2.0, marker=marker,
                  markersize=5, zorder=4, label=f"{label}  (n={h.size})")
        baseline = float(data.baseline(n, d).mean())
        axis.axhline(baseline, color=colour, linewidth=1.0, linestyle="--",
                     alpha=0.75, zorder=3)

    limit = float(np.log(len(data.train_names)))
    axis.axvspan(support_edge, limit + 0.08, color="#BBBBBB", alpha=0.30, zorder=0)
    axis.set_xlim(0.0, limit + 0.08)
    axis.set_yscale("log")
    axis.set_xlabel("Mixture entropy  $H(p)$  (nats)")
    axis.set_ylabel("Validation loss, 13-domain mean (nats)")
    ticks = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]
    axis.set_yticks(ticks)
    axis.yaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in ticks]))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, limit])
    axis.set_xticklabels(["0", "0.5", "1.0", "1.5", "2.0", "2.5",
                          f"{limit:.2f}\n(uniform)"])
    tidy(axis)
    handles, labels = axis.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#555555", linewidth=1.0, linestyle="--"))
    labels.append("mixture-free baseline\n$E_k{+}A_kN^{-\\alpha}{+}B_kD^{-\\beta}$")
    axis.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                fontsize=9, ncols=4, columnspacing=1.2, handletextpad=0.5)
    spread_text = " / ".join(f"{value:.2f}" for _, value, _ in spreads)
    note(axis,
         "dots: every recorded mixture\n"
         "line: the law, bin-averaged\n"
         f"cloud spread (nats): {spread_text}\n"
         "at 1M / 60M / 1B\n"
         f"no data beyond H = {support_edge:.2f}; the\n"
         f"uniform mixture ({limit:.2f}) is outside",
         loc="lower left", size=8)
    return figure


def panel_surface_parity(data: LawData) -> plt.Figure:
    """The delivered parity figure, rendered by :func:`_parity_figure`.

    ``generalized_law_surface_parity.png`` is the name the write-up refers to, so
    it must stay identical to ``generalized_law_identification_parity.png``: both
    names call the same builder instead of keeping two copies of the drawing code
    that could drift apart.

    The earlier standalone version plotted observed against predicted on log axes
    with no reference model.  The current figure keeps that parity check and adds
    the per-scale error against the same coefficients with the mixture term
    switched off, which is what shows the identified term is doing the work.
    """

    return _parity_figure(data, chinese=True)


def panel_response_parity(data: LawData) -> plt.Figure:
    """Does the fitted mixture response track the loss left after removing E+A+B?"""

    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE, layout="constrained")
    predicted_all, observed_all = [], []
    for name in HELDOUT_SCALES:
        block = next(b for b in data.test_blocks if b.name == name)
        colour, marker = SCALE_STYLE[name]
        predicted = data.amplitude(block.n, block.d)[None, :] * data.mix_response(block.p)
        observed = block.y - data.baseline(block.n, block.d)[None, :]
        axis.scatter(predicted.ravel(), observed.ravel(), s=11, alpha=0.55,
                     color=colour, marker=marker, edgecolors="none",
                     label=f"{SCALE_LABEL[name]} held-out")
        predicted_all.append(predicted.ravel())
        observed_all.append(observed.ravel())
    predicted = np.concatenate(predicted_all)
    observed = np.concatenate(observed_all)
    low = min(observed.min(), predicted.min()) * 1.06
    high = max(observed.max(), predicted.max()) * 0.94
    axis.plot([low, high], [low, high], color="#333333", linewidth=1.0,
              linestyle="--", zorder=1, label="y = x")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel(r"Predicted mixture term  $S_k(N,D)\,\Phi_k(p,Q)$  (nats)")
    axis.set_ylabel("Observed loss minus baseline  (nats)")
    tidy(axis)
    axis.legend(loc="upper left", markerscale=1.8, handletextpad=0.5, fontsize=10)
    within = 1.0 - np.sum((observed - predicted) ** 2) / np.sum(
        (observed - observed.mean()) ** 2)
    note(axis,
         f"baseline  $E_k+A_kN^{{-\\alpha}}+B_kD^{{-\\beta}}$\n"
         f"removed per evaluation domain\n"
         f"pooled $R^2$ = {within:.3f}",
         loc="lower right")
    return figure


def panel_error_by_scale(data: LawData) -> plt.Figure:
    metrics = data.per_scale([b for b in data.test_blocks
                             if b.name in HELDOUT_SCALES])
    names = [name for name in HELDOUT_SCALES if name in metrics]
    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE, layout="constrained")
    x = np.arange(len(names))
    width = 0.36
    rmse = [metrics[name]["relative_rmse"] * 100 for name in names]
    mape = [metrics[name]["median_ape"] * 100 for name in names]
    axis.bar(x - width / 2, rmse, width, color=BAR_COLOR, label="relative RMSE")
    axis.bar(x + width / 2, mape, width, color=BAR_COLOR_ALT, label="median APE")
    for shift, values in ((-width / 2, rmse), (width / 2, mape)):
        for position, value in zip(x + shift, values):
            axis.text(position, value + 0.12, f"{value:.2f}", ha="center",
                      va="bottom", fontsize=9)
    axis.set_xticks(x)
    axis.set_xticklabels([f"{SCALE_LABEL[name]}\nheld-out" for name in names])
    axis.set_ylabel("Held-out error (%)")
    axis.set_ylim(0, max(rmse) * 1.25)
    tidy(axis, grid_axis="y")
    axis.legend(loc="upper right", ncols=2, fontsize=11)
    return figure


def panel_amplitude_decay(data: LawData) -> plt.Figure:
    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE, layout="constrained")
    x = np.arange(len(ANCHORS))
    curves = np.vstack([data.amplitude(n, d) for _, n, d in ANCHORS])  # (3, 13)
    # Each anchor's amplitude was inferred as one scalar per evaluation domain
    # from that anchor's own training mixtures; eta_k and zeta_k are then solved
    # from the two non-reference anchors, so the curves pass through the markers.
    observed = data.law["scale_anchor_diagnostics"]["observed_amplitude"]
    obs_matrix = np.vstack([
        np.asarray([observed[key][domain] for domain in data.loss_names])
        for key in ("train_1m", "test_60m", "test_1b")])
    lo, hi = int(np.argmin(curves[-1])), int(np.argmax(curves[-1]))
    for k in range(len(data.loss_names)):
        if k in (lo, hi):
            continue
        axis.plot(x, curves[:, k], color="#BBBBBB", linewidth=1.0, zorder=1)
    for k, colour in ((hi, ACCENT), (lo, BAR_COLOR)):
        axis.plot(x, curves[:, k], color=colour, linewidth=2.0, marker="o",
                  markersize=5, zorder=3,
                  label=f"{data.loss_names[k]}   "
                        f"$\\eta$={data.eta[k]:.3f}, $\\zeta$={data.zeta[k]:.3f}")
    axis.scatter(np.repeat(x, len(data.loss_names)), obs_matrix.ravel(),
                 s=15, facecolors="none", edgecolors="#222222", linewidths=0.7,
                 zorder=4, label="amplitude inferred per anchor")
    axis.set_yscale("log")
    axis.set_xticks(x)
    axis.set_xticklabels([f"{label}\nN={label}, D="
                          f"{'25B' if label == '1B' else '1B'}" for label, _, _ in ANCHORS])
    axis.set_ylim(0.055, 1.85)
    axis.set_ylabel("Scale amplitude  $S_k(N,D)$")
    tidy(axis, grid_axis="y")
    handles, labels = axis.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#BBBBBB", linewidth=1.4))
    labels.append("remaining 11 evaluation domains")
    axis.legend(handles, labels, loc="lower left", fontsize=9, handletextpad=0.6)
    note(axis,
         "$\\eta_k,\\zeta_k$ are solved exactly from\n"
         "the 2 non-reference anchors:\n"
         "zero residual, no held-out check",
         loc="upper right")
    return figure


def panel_mixture_response(data: LawData) -> plt.Figure:
    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE, layout="constrained")
    index = data.train_names.index("arxiv")
    shares = np.linspace(0.0, 0.6, 80)
    others = [k for k in range(len(data.train_names)) if k != index]
    mixtures = np.empty((shares.size, len(data.train_names)))
    mixtures[:, index] = shares
    mixtures[:, others] = ((1.0 - shares) / len(others))[:, None]
    response = data.mix_response(mixtures)
    chosen = ["arxiv", "wikipedia_en", "github", "pile_cc", "gutenberg_pg_19"]
    colours = ["#08306B", "#2171B5", "#4292C6", "#6BAED6", "#9ECAE1"]
    ranked = sorted((data.loss_names.index(name) for name in chosen),
                    key=lambda k: response[-1, k])
    for colour, k in zip(colours, ranked):
        axis.plot(shares * 100, response[:, k], color=colour, linewidth=1.8,
                  label=data.loss_names[k])
    axis.axhline(0.0, color="#333333", linewidth=0.9, linestyle=":")
    axis.set_xlabel("Share of arxiv in the mixture (%)")
    axis.set_ylabel(r"Mixture response  $\Phi_k(p,Q)$  (nats)")
    axis.set_ylim(-3.95, 0.05)
    tidy(axis)
    axis.legend(loc="upper left", ncols=2, fontsize=10, columnspacing=1.0,
                handlelength=1.6)
    note(axis,
         "1-D slice of the fitted surface: the other 16 domains\n"
         "split the remaining share equally. 5 of the 13 evaluation\n"
         "domains shown; $\\Phi_k$ is centred on the mean of the 512\n"
         "training mixtures. The steep bend where a share reaches\n"
         "0 is the $\\epsilon$ = 1e-3 clr floor, not a data effect.",
         loc="lower right")
    return figure


def panel_weight_heatmap(data: LawData) -> plt.Figure:
    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE, layout="constrained")
    weights = data.effective_pq
    limit = float(np.abs(weights).max())
    # A few weights are ~20x the rest; a symlog norm keeps the small structure
    # visible instead of flattening the whole matrix to white.
    image = axis.imshow(weights, cmap="RdBu_r", aspect="auto",
                        norm=SymLogNorm(linthresh=5.0, vmin=-limit, vmax=limit,
                                        base=10))
    axis.set_xticks(np.arange(len(data.train_names)))
    axis.set_xticklabels(data.train_names, rotation=90, fontsize=8)
    axis.set_yticks(np.arange(len(data.loss_names)))
    axis.set_yticklabels(data.loss_names, fontsize=9)
    axis.set_xlabel("Mixture domain (17)", fontsize=12)
    axis.set_ylabel("Evaluation domain (13)", fontsize=12)
    bar = figure.colorbar(image, ax=axis, pad=0.02, fraction=0.045)
    bar.set_label(r"effective $W^{pQ}_{k,i}$", fontsize=11)
    bar.ax.tick_params(labelsize=9)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    return figure


def panel_structure_diagnostics(data: LawData) -> plt.Figure:
    """Why the response cannot be one shared positive scalar across 13 domains."""

    diagnostic = data.payload["structural_diagnostics"]
    weights = data.effective_pq
    labels = ["effective weights that are\nnegative",
              "evaluation-domain pairs that\nrespond with opposite sign",
              "1M loss-table variance in\nthe 1st principal component"]
    values = [float(np.mean(weights < 0)) * 100,
              diagnostic["training_1m_negative_domain_pair_fraction"] * 100,
              diagnostic[
                  "training_1m_first_principal_component_variance_fraction"] * 100]

    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE, layout="constrained")
    y = np.arange(len(labels))
    axis.barh(y, values, 0.55, color=BAR_COLOR)
    for position, value in zip(y, values):
        axis.text(value + 1.2, position, f"{value:.1f}%", va="center", ha="left",
                  fontsize=11)
    axis.axvline(50.0, color=ACCENT, linewidth=1.3, linestyle="--", zorder=1,
                 label="50% = no sign asymmetry")
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=10)
    axis.invert_yaxis()
    axis.set_xlabel("Share of the 1M training table (%)")
    axis.set_xlim(0, 118)
    axis.set_xticks([0, 20, 40, 60, 80, 100])
    tidy(axis, grid_axis="x")
    axis.legend(loc="lower right", fontsize=10)
    note(axis,
         "221 weights = 13 x 17 \n"
         "one shared positive scalar\n"
         "would force 0% on rows 1-2",
         loc="center right", size=8)
    return figure


def panel_ridge_ladder(data: LawData) -> plt.Figure:
    candidates = sorted(data.law["ridge_candidates"], key=lambda item: item["ridge"])
    ridge = np.asarray([item["ridge"] for item in candidates])
    score = np.asarray([item["reference_tuning_relative_rmse"] for item in candidates])
    selected = float(data.law["ridge_selected"])
    baseline = next(
        item for item in data.payload["normalization_comparison"]["raw"]["candidates"]
        if item["form"] == "no_mixture")["relative_rmse"]

    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE, layout="constrained")
    axis.axhline(baseline * 100, color=ACCENT, linewidth=1.4, linestyle="--",
                 label=f"no mixture term  ({baseline:.2%})")
    axis.plot(ridge, score * 100, color=BAR_COLOR, marker="o", markersize=6,
              linewidth=1.8, label="compositional law (tuning)")
    axis.scatter([selected], [score[ridge == selected][0] * 100], s=150,
                 facecolors="none", edgecolors="#222222", linewidths=1.2,
                 zorder=5, label=f"selected ridge = {selected:g}")
    for x_value, y_value in zip(ridge, score * 100):
        axis.text(x_value, y_value + 0.45, f"{y_value:.2f}", ha="center",
                  va="bottom", fontsize=9)
    axis.set_xscale("log")
    axis.set_xlabel("Ridge strength")
    axis.set_ylabel("Tuning relative RMSE (%)")
    axis.set_ylim(0, baseline * 100 * 1.10)
    tidy(axis, grid_axis="y")
    axis.legend(loc="center left", fontsize=10)
    return figure


def panel_elasticity(data: LawData) -> plt.Figure:
    """Elasticities of the two laws at the same anchor.

    The reference mixture is the recorded mixture at the median recorded entropy
    (see ``elasticity_and_substitution.json``). The uniform mixture 1/17 is *not*
    usable here: its entropy lies beyond every recorded mixture and the law
    under-predicts loss there by 24-45%.
    """

    n_b, d_b, q_scale = 1.0, 25.0, 1.0
    stored = json.loads(
        (HANDOFF_DIR / "elasticity_and_substitution.json").read_text(encoding="utf-8"))
    reference_mixture = np.asarray(
        [stored["reference_point"]["mixture"][name] for name in data.train_names])
    step = 1e-4

    def mean_loss(p: np.ndarray, n: float, d: float,
                  quality_scale: float = 1.0) -> float:
        """Mean over the 13 evaluation domains, matching the handoff definition."""

        return float(data.predict(n * 1e9, d * 1e9, p, quality_scale)[0].mean())

    base = mean_loss(reference_mixture, n_b, d_b)
    e_n = (mean_loss(reference_mixture, n_b * (1 + step), d_b)
           - mean_loss(reference_mixture, n_b * (1 - step), d_b)) / (2 * step) / base
    e_d = (mean_loss(reference_mixture, n_b, d_b * (1 + step))
           - mean_loss(reference_mixture, n_b, d_b * (1 - step))) / (2 * step) / base
    e_q = (mean_loss(reference_mixture, n_b, d_b, 1 + step)
           - mean_loss(reference_mixture, n_b, d_b, 1 - step)) / (2 * step) / base

    reference = next(
        item for item in stored["elasticity_from_q2_law"]["points"]
        if item["N_billion_params"] == n_b and item["D_billion_tokens"] == d_b)
    for label, value, key in (("dlnL/dlnN", e_n, "dlnL_dlnN"),
                              ("dlnL/dlnD", e_d, "dlnL_dlnD"),
                              ("dlnL/dlnQ", e_q, "dlnL_dlnQ_scale")):
        if abs(value - reference[key]) > 1e-6:
            raise SystemExit(f"{label}: recomputed {value:.8f} vs stored "
                             f"{reference[key]:.8f}")

    # How far the implied elasticities move across the recorded mixtures. For Q
    # this is the whole point: the sign is not even stable.
    recorded = np.vstack([block.p for block in data.blocks])
    span = {}
    for key, name in (("dlnL_dlnN", "n"), ("dlnL_dlnD", "d"), ("dlnL_dlnQ", "q")):
        values = []
        for mixture in recorded:
            if name == "n":
                low, high = mean_loss(mixture, n_b * (1 - step), d_b), \
                    mean_loss(mixture, n_b * (1 + step), d_b)
            elif name == "d":
                low, high = mean_loss(mixture, n_b, d_b * (1 - step)), \
                    mean_loss(mixture, n_b, d_b * (1 + step))
            else:
                low, high = mean_loss(mixture, n_b, d_b, 1 - step), \
                    mean_loss(mixture, n_b, d_b, 1 + step)
            values.append((high - low) / (2 * step) / mean_loss(mixture, n_b, d_b))
        span[key] = (float(np.min(values)), float(np.max(values)))

    classic = classic_elasticities(n_b, d_b)
    quality = quality_law_elasticities(n_b, d_b, q_scale)

    groups = ["$\\partial\\ln L/\\partial\\ln N$",
              "$\\partial\\ln L/\\partial\\ln D$",
              "$\\partial\\ln L/\\partial\\ln Q$"]
    left = [e_n, e_d, e_q]
    right = [classic["dlnL_dlnN"], classic["dlnL_dlnD"], quality["dlnL_dlnQ"]]

    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE, layout="constrained")
    x = np.arange(len(groups))
    width = 0.34
    left_bars = axis.bar(x - width / 2, left, width, color=BAR_COLOR,
                         label="Q2 generalized law")
    right_bars = axis.bar(x + width / 2, right, width, color=BAR_COLOR_ALT,
                          label="classic B1 law (N, D)  /  quality B7 law (Q)")
    # The quality elasticity of the generalized law is an extrapolation: every
    # Q_i is fixed in attachment A, so the model cannot identify a quality effect.
    left_bars[2].set_hatch("///")
    left_bars[2].set_edgecolor("#333333")
    # The two bars of a group are close together, so the value labels are put on
    # two different levels instead of colliding horizontally.
    for bars, values, offset in ((left_bars, left, -0.007), (right_bars, right, -0.031)):
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + offset,
                      f"{value:.3f}", ha="center", va="top", fontsize=10)
    axis.axhline(0.0, color="#333333", linewidth=0.9)
    axis.set_xticks(x)
    axis.set_xticklabels(groups)
    axis.set_ylabel("Log-log elasticity")
    axis.set_ylim(min(left + right) * 1.90, 0.07)
    tidy(axis, grid_axis="y")
    axis.legend(loc="lower right", fontsize=10)
    reference_entropy = float(-(
        reference_mixture * np.log(np.clip(reference_mixture, 1e-12, None))).sum())
    note(axis,
         f"at N={n_b:g}B, D={d_b:g}B, Q={q_scale:g}; reference mixture is the\n"
         f"recorded mixture at the median entropy, H = {reference_entropy:.3f} nats\n"
         "over all 1088 recorded mixtures the generalized law's\n"
         f"$\\partial\\ln L/\\partial\\ln Q$ spans {span['dlnL_dlnQ'][0]:+.3f} to "
         f"{span['dlnL_dlnQ'][1]:+.3f}$\\to$ sign-unstable,\n"
         "so Q must come from B6/B7, not from this law",
         loc="upper left", size=8)
    return figure


def panel_epsilon_sensitivity(data: LawData) -> plt.Figure:
    sweep = data.payload["epsilon_sensitivity"]
    epsilon = np.asarray([item["epsilon"] for item in sweep])
    rmse = np.asarray([item["heldout_relative_rmse"] for item in sweep]) * 100
    mixture_r2 = np.asarray(
        [item["heldout_mixture_r_squared_within_scale"] for item in sweep])

    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE, layout="constrained")
    axis.bar(np.arange(len(epsilon)), rmse, 0.55, color=BAR_COLOR,
             label="held-out relative RMSE")
    # The twin-axis line is drawn by a second Axes and therefore always sits on
    # top of the primary Axes' artists, so a label placed above the bar would be
    # struck through. Keep the values inside the bars instead.
    for index, value in enumerate(rmse):
        axis.text(index, value - 0.12, f"{value:.2f}", ha="center", va="top",
                  fontsize=9, color="white")
    axis.set_xticks(np.arange(len(epsilon)))
    axis.set_xticklabels([f"{value:g}" for value in epsilon])
    axis.set_xlabel(r"clr smoothing  $\epsilon$")
    axis.set_ylabel("Held-out relative RMSE (%)")
    axis.set_ylim(0, rmse.max() * 1.22)
    tidy(axis, grid_axis="y")

    twin = axis.twinx()
    twin.plot(np.arange(len(epsilon)), mixture_r2, color=ACCENT, marker="s",
              markersize=6, linewidth=1.8, label="held-out mixture $R^2$")
    twin.set_ylabel("Held-out mixture $R^2$", color=ACCENT)
    twin.tick_params(axis="y", colors=ACCENT)
    twin.set_ylim(min(mixture_r2) - 0.025, max(mixture_r2) + 0.035)
    twin.spines["top"].set_visible(False)
    twin.spines["right"].set_color(ACCENT)

    handles = [axis.containers[0], twin.lines[0]]
    axis.legend(handles, ["held-out relative RMSE", "held-out mixture $R^2$"],
                loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, fontsize=11)
    return figure


def panel_domain_error_1b(data: LawData) -> plt.Figure:
    block = next(b for b in data.test_blocks if b.name == "test_1b")
    per_domain = data.per_domain(block)
    overall = data.per_scale([block])["test_1b"]["relative_rmse"]
    order = sorted(data.loss_names, key=lambda name: per_domain[name]["relative_rmse"])
    values = [per_domain[name]["relative_rmse"] * 100 for name in order]

    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE, layout="constrained")
    y = np.arange(len(order))
    axis.barh(y, values, 0.68, color=BAR_COLOR, label="relative RMSE per domain")
    axis.axvline(overall * 100, color=ACCENT, linewidth=1.4, linestyle="--", zorder=1,
                 label=f"all domains pooled ({overall:.2%})")
    for position, value in zip(y, values):
        axis.text(value + 0.08, position, f"{value:.1f}", va="center", ha="left",
                  fontsize=8, zorder=6,
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
    axis.set_yticks(y)
    axis.set_yticklabels(order, fontsize=9)
    axis.set_xlabel("Relative RMSE at the 1B anchor (%)")
    axis.set_xlim(0, max(values) * 1.20)
    tidy(axis, grid_axis="x")
    axis.legend(loc="lower right", fontsize=10)
    note(axis, "13 held-out mixtures\nx 13 evaluation domains",
         loc="center right", size=8)
    return figure


def _illustrative_heldout_pair(data: LawData) -> tuple[object, int, int]:
    """Two *recorded* 1M mixtures, not an artificial path through the simplex.

    Use the 10th and 90th percentiles of predicted mean loss so that the
    contrast is visible without cherry-picking the most extreme endpoints.
    This pair is descriptive, not a performance metric or a causal intervention.
    """

    block = next(b for b in data.test_blocks if b.name == "test_1m")
    order = np.argsort(data.predict(block.n, block.d, block.p).mean(axis=1))
    low = int(order[round(0.10 * (len(order) - 1))])
    high = int(order[round(0.90 * (len(order) - 1))])
    return block, low, high


def _contrast_components(data: LawData, p_low: np.ndarray,
                         p_high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact additive decomposition of Phi(high)-Phi(low) in the fitted basis."""

    features = data._features(np.vstack((p_low, p_high)))
    standardized_delta = (features[1] - features[0]) / np.asarray(
        data.mixture["feature_scale"])
    split = len(data.train_names)
    weighted_share = (data.ridge_coefficients[:, :split]
                      @ standardized_delta[:split]) * data.response_slope
    clr = (data.ridge_coefficients[:, split:]
           @ standardized_delta[split:]) * data.response_slope
    exact_delta = data.mix_response(p_high)[0] - data.mix_response(p_low)[0]
    if not np.allclose(weighted_share + clr, exact_delta, atol=1e-9):
        raise ValueError("pQ/clr contributions do not sum to the model contrast")
    return weighted_share, clr


def panel_quality_pathway(data: LawData) -> plt.Figure:
    """Show where Q enters the law without implying a fitted quality elasticity."""

    training = next(b for b in data.train_blocks if b.name == "train_1m")
    mean_share = training.p.mean(axis=0)
    weighted_share = mean_share * data.q
    order = np.argsort(-weighted_share)
    y = np.arange(len(order))
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 6.4),
                                gridspec_kw={"width_ratios": [1.1, 1.1, 1.35]})
    figure.subplots_adjust(left=0.18, right=0.98, top=0.96, bottom=0.15,
                           wspace=0.18)
    columns = (
        (data.q[order], "Fixed quality score  $Q_i$", BAR_COLOR),
        (100 * mean_share[order], "Mean training share  $p_i$  (%)", "#F58518"),
        (100 * weighted_share[order],
         "Weighted exposure  $100p_iQ_i$", "#54A24B"),
    )
    for index, (axis, (values, label, colour)) in enumerate(zip(axes, columns)):
        axis.barh(y, values, height=0.72, color=colour, alpha=0.9)
        axis.set_xlabel(label, fontsize=10)
        axis.set_ylim(len(y) - 0.4, -0.6)
        axis.set_xlim(0, max(values) * 1.12)
        axis.tick_params(axis="x", labelsize=9)
        axis.tick_params(axis="y", length=0)
        if index == 0:
            axis.set_yticks(y)
            axis.set_yticklabels([data.train_names[i] for i in order], fontsize=9)
        else:
            axis.set_yticks(y)
            axis.set_yticklabels([])
        tidy(axis, grid_axis="x")
    figure.text(0.18, 0.035,
                "Q is fixed per source (6 distinct scores for 17 sources). "
                "The model sees pQ, not independently varied Q; "
                "a causal quality effect is not identified.",
                fontsize=9, ha="left", va="bottom")
    return figure


def panel_observed_contrast(data: LawData) -> plt.Figure:
    """A within-anchor comparison: changed shares and changed losses together."""

    block, low, high = _illustrative_heldout_pair(data)
    delta_share = 100 * (block.p[high] - block.p[low])
    predicted = data.predict(block.n, block.d, block.p)
    delta_predicted = predicted[high] - predicted[low]
    delta_observed = block.y[high] - block.y[low]
    source_order = np.argsort(-np.abs(delta_share))
    domain_order = np.argsort(-delta_predicted)
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 6.5),
                                gridspec_kw={"width_ratios": [1.05, 1.2]})
    figure.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.17,
                           wspace=0.54)
    left, right = axes
    y_left = np.arange(len(source_order))
    values = delta_share[source_order]
    left.barh(y_left, values, height=0.7,
              color=np.where(values >= 0, "#F58518", BAR_COLOR))
    left.set_yticks(y_left)
    left.set_yticklabels([data.train_names[i] for i in source_order], fontsize=9)
    left.set_ylim(len(y_left) - 0.4, -0.6)
    left.set_xlabel(r"Change in training share  $\Delta p_i$  (pp)", fontsize=11)
    left.axvline(0, color="#333333", linewidth=1.0)
    tidy(left, grid_axis="x")

    y_right = np.arange(len(domain_order))
    right.barh(y_right, delta_predicted[domain_order], height=0.65,
               color="#4C78A8", alpha=0.8, label="predicted change")
    right.scatter(delta_observed[domain_order], y_right, s=38,
                  facecolors="white", edgecolors="#222222", linewidths=1.2,
                  zorder=5, label="observed change")
    right.set_yticks(y_right)
    right.set_yticklabels([data.loss_names[i] for i in domain_order], fontsize=9)
    right.set_ylim(len(y_right) - 0.4, -0.6)
    right.set_xlabel(r"Change in validation loss  $\Delta L_k$  (nats)", fontsize=11)
    right.axvline(0, color="#333333", linewidth=1.0)
    tidy(right, grid_axis="x")
    right.legend(loc="lower right", fontsize=9)
    figure.text(0.16, 0.035,
                "Higher - lower: two recorded held-out 1M mixtures at the 90th/10th "
                "percentiles of predicted mean loss. All shares change together; "
                "this is an illustrative contrast, not a causal substitution.",
                fontsize=9, ha="left", va="bottom")
    return figure


def panel_component_breakdown(data: LawData) -> plt.Figure:
    """Exact feature-block arithmetic for the observed contrast above."""

    block, low, high = _illustrative_heldout_pair(data)
    pq, clr = _contrast_components(data, block.p[low], block.p[high])
    amplitude = data.amplitude(block.n, block.d)
    pq, clr = pq * amplitude, clr * amplitude
    total = pq + clr
    order = np.argsort(-total)
    y = np.arange(len(order))
    figure, axis = plt.subplots(figsize=(8.8, 6.3))
    figure.subplots_adjust(left=0.25, right=0.97, top=0.94, bottom=0.19)
    axis.barh(y - 0.2, pq[order], height=0.34, color="#54A24B",
              label=r"quality-weighted share block  $p_iQ_i$")
    axis.barh(y + 0.2, clr[order], height=0.34, color="#9C6ADE",
              label=r"compositional log-ratio block  $\mathrm{clr}(p)$")
    axis.scatter(total[order], y, color="#222222", marker="D", s=30, zorder=6,
                 label="total predicted change")
    axis.set_yticks(y)
    axis.set_yticklabels([data.loss_names[i] for i in order], fontsize=9)
    axis.set_ylim(len(y) - 0.5, -0.5)
    axis.axvline(0, color="#333333", linewidth=1.0)
    axis.set_xlabel(r"Contribution to loss change  $\Delta L_k$  (nats)", fontsize=11)
    tidy(axis, grid_axis="x")
    axis.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=9,
                ncols=1)
    figure.text(0.25, 0.035,
                "Same held-out pair as the preceding figure. The two blocks sum "
                "exactly to the model change; their split depends on the fitted "
                "basis and is not an independent quality effect.",
                fontsize=9, ha="left", va="bottom")
    return figure


def panel_domain_scale(data: LawData) -> plt.Figure:
    """Make the fitted, domain-specific scale attenuation easier to compare."""

    amplitudes = np.stack([data.amplitude(n, d) for _, n, d in ANCHORS], axis=1)
    order = np.argsort(amplitudes[:, -1])
    figure, axis = plt.subplots(figsize=(6.6, 6.2))
    figure.subplots_adjust(left=0.30, right=0.86, top=0.95, bottom=0.16)
    image = axis.imshow(amplitudes[order], aspect="auto", cmap="YlGnBu",
                        vmin=0, vmax=1)
    axis.set_xticks(range(len(ANCHORS)))
    axis.set_xticklabels([label for label, _, _ in ANCHORS])
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels([data.loss_names[i] for i in order], fontsize=9)
    axis.set_xlabel("Fitted (model, token) anchor", fontsize=11)
    axis.set_ylabel("Evaluation domain", fontsize=11)
    for row, i in enumerate(order):
        for column in range(len(ANCHORS)):
            value = amplitudes[i, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                      fontsize=9, color="white" if value > 0.6 else "#202020")
    bar = figure.colorbar(image, ax=axis, pad=0.03, fraction=0.05)
    bar.set_label(r"Mixture-effect multiplier  $S_k(N,D)$", fontsize=10)
    figure.text(0.30, 0.035,
                "1M: 1B tokens (S = 1); 60M: 1B tokens; 1B: 25B tokens. "
                "Values are calibrated at these anchors, not independently "
                "validated scale effects.",
                fontsize=9, ha="left", va="bottom")
    return figure


def panel_scaling_curves(data: LawData) -> plt.Figure:
    """Show the generalized law as a small family of scaling curves.

    The horizontal path connects the three *observed scale anchors* in log
    (N,D), without suggesting that arbitrary N and D combinations were tested.
    Each coloured curve holds one recorded, held-out mixture fixed.  The
    differing p and fixed source-specific Q are descriptive attributes, not
    separately identified causal effects.
    """

    one = next(b for b in data.test_blocks if b.name == "test_1m")
    sixty = next(b for b in data.test_blocks if b.name == "test_60m")
    if one.ids != sixty.ids or not np.allclose(one.p, sixty.p, atol=1e-9):
        raise ValueError("the 1M and 60M held-out mixtures must be paired")
    reference_prediction = data.predict(one.n, one.d, one.p).mean(axis=1)
    order = np.argsort(reference_prediction)
    selected = [int(order[round(q * (len(order) - 1))])
                for q in (0.1, 0.5, 0.9)]
    if len(set(selected)) != 3:
        raise ValueError("not enough distinct held-out mixtures")

    anchor_n = np.array([1e6, 6e7, 1e9])
    anchor_d = np.array([1e9, 1e9, 2.5e10])
    n_path = np.concatenate([
        np.geomspace(anchor_n[0], anchor_n[1], 90),
        np.geomspace(anchor_n[1], anchor_n[2], 90)[1:]])
    d_path = np.concatenate([
        np.geomspace(anchor_d[0], anchor_d[1], 90),
        np.geomspace(anchor_d[1], anchor_d[2], 90)[1:]])
    compute = 6 * n_path * d_path
    anchors_c = 6 * anchor_n * anchor_d

    figure, axis = plt.subplots(figsize=(10.2, 6.3))
    figure.subplots_adjust(left=0.12, right=0.97, top=0.84, bottom=0.22)
    for value in anchors_c:
        axis.axvline(value, color="#D5D9DE", linewidth=1.0, linestyle=":",
                     zorder=0)
    baseline = np.array([data.baseline(n, d).mean()
                         for n, d in zip(n_path, d_path)])
    axis.plot(compute, baseline, color="#303943", linewidth=2.2,
              linestyle=(0, (6, 4)), label=r"Reference: $E+AN^{-\alpha}+BD^{-\beta}$",
              zorder=2)

    colors = ("#2B7A9B", "#E19B34", "#925C98")
    labels = ("A", "B", "C")
    observed_sixty = sixty.y[selected].mean(axis=1)
    observed_one = one.y[selected].mean(axis=1)
    for row, (index, color, letter) in enumerate(zip(selected, colors, labels)):
        mixture = one.p[index:index + 1]
        predicted = np.array([data.predict(n, d, mixture).mean()
                              for n, d in zip(n_path, d_path)])
        share = mixture[0] / mixture.sum()
        dominant = int(np.argmax(share))
        source = data.train_names[dominant].replace("_", " ")
        legend_label = (f"{letter}: {source} {share[dominant]:.0%}, "
                        f"$Q_{{i}}$={data.q[dominant]:.2f}")
        axis.plot(compute, predicted, color=color, linewidth=3.0,
                  label=legend_label, zorder=3)
        axis.scatter(anchors_c[:2], [observed_one[row], observed_sixty[row]],
                     marker="o", s=58, facecolors="white", edgecolors=color,
                     linewidths=1.8, zorder=5)
        axis.scatter(anchors_c[-1], predicted[-1], marker="o", s=35,
                     facecolors=color, edgecolors="white", linewidths=0.8,
                     zorder=5)

    axis.set_xscale("log")
    axis.set_xlim(anchors_c[0] / 1.45, anchors_c[-1] * 1.45)
    axis.set_ylim(1.9, 6.1)
    axis.set_xlabel(r"Training compute  $C\approx 6ND$  (FLOPs)", fontsize=13)
    axis.set_ylabel("Mean validation loss (nats; 13 domains)", fontsize=13)
    axis.grid(which="major", color="#D9DDE1", alpha=0.85)
    axis.grid(which="minor", axis="x", color="#E9ECEF", alpha=0.45)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)

    anchor_labels = ("N=1M  ·  D=1B", "N=60M  ·  D=1B",
                     "N=1B  ·  D=25B")
    for j, (c, label) in enumerate(zip(anchors_c, anchor_labels)):
        align = ("left", "center", "right")[j]
        axis.annotate(label,
                      xy=(c, 1), xycoords=("data", "axes fraction"),
                      xytext=(0, 13), textcoords="offset points", ha=align,
                      va="bottom", fontsize=9.5, color="#414A52")
    axis.legend(loc="upper right", bbox_to_anchor=(0.99, 0.97), fontsize=9.2,
                frameon=True, facecolor="white", edgecolor="#CCD0D4",
                framealpha=0.98, handlelength=2.6,
                title="Fixed mixtures (dominant source: share, Q)",
                title_fontsize=9)
    figure.text(0.12, 0.08,
                "Curves: model interpolation along the three scale anchors. "
                "Open circles: held-out observations at 1M and 60M; "
                "filled circles: model-only transfer at 1B. "
                "Q is source-fixed, so the coloured gaps reflect joint p/Q composition, "
                "not an isolated quality effect.",
                fontsize=9, ha="left", va="bottom", wrap=True)
    return figure


def panel_all_domain_p_q(data: LawData) -> plt.Figure:
    """Separate p/Q effects for all 17 sources and 13 evaluation domains.

    Both columns apply the same 10% *relative* perturbation at the mean 1M
    training mixture. A share increase reallocates the simplex by reducing all
    other shares proportionally; a Q increase holds every share fixed. The Q
    panel is a formal model sensitivity, not an identified empirical effect.
    """

    training = next(b for b in data.train_blocks if b.name == "train_1m")
    reference = training.p.mean(axis=0)
    reference = reference / reference.sum()
    n, d = 1e6, 1e9
    baseline = data.predict(n, d, reference[None, :])[0]
    raised_p = np.repeat(reference[None, :], len(reference), axis=0)
    for source in range(len(reference)):
        target = 1.10 * reference[source]
        if target >= 1:
            raise ValueError("a source share is too large for the 10% perturbation")
        raised_p[source] *= (1 - target) / (1 - reference[source])
        raised_p[source, source] = target
    if not np.allclose(raised_p.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("source-share perturbations left the probability simplex")

    # Rows are evaluation domains; columns are source domains.
    p_effect = (data.predict(n, d, raised_p) - baseline[None, :]).T
    q_effect = (data.amplitude(n, d)[:, None] * data.effective_pq
                * (0.10 * reference * data.q)[None, :])
    order = np.argsort(-reference)
    values = (p_effect[:, order].T, q_effect[:, order].T)
    limit = max(float(np.max(np.abs(block))) for block in values)
    normalizer = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    figure, axes = plt.subplots(1, 2, figsize=(17.0, 8.7), sharey=True)
    figure.subplots_adjust(left=0.17, right=0.91, top=0.84, bottom=0.28,
                           wspace=0.065)
    titles = (r"(a) Increase source share $p_i$ by 10%",
              r"(b) Increase source quality $Q_i$ by 10%")
    for index, (axis, matrix, title) in enumerate(zip(axes, values, titles)):
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r",
                            norm=normalizer, interpolation="nearest")
        axis.set_title(title, fontsize=12, loc="left", pad=12)
        axis.set_xticks(range(len(data.loss_names)))
        axis.set_xticklabels([name.replace("_", " ")
                              for name in data.loss_names],
                             rotation=55, ha="right", rotation_mode="anchor",
                             fontsize=8.7)
        axis.set_yticks(range(len(order)))
        if index == 0:
            axis.set_yticklabels(
                [f"{data.train_names[source]}  (Q={data.q[source]:.2f})"
                 for source in order], fontsize=9)
            axis.set_ylabel("Training source (recorded Q)", fontsize=11)
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.set_xlabel("Evaluation domain (loss $L_k$)", fontsize=10)
        axis.tick_params(axis="both", length=0)
        axis.set_xticks(np.arange(-0.5, len(data.loss_names), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.65)
        axis.tick_params(which="minor", bottom=False, left=False)
    colorbar = figure.colorbar(image, ax=list(axes), fraction=0.025,
                               pad=0.018, aspect=26)
    colorbar.set_label(r"Predicted loss change  $\Delta L_k$  (nats)",
                       fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    figure.text(0.17, 0.94,
                "All 17 training sources × all 13 evaluation domains",
                fontsize=13, ha="left", fontweight="bold")
    figure.text(0.17, 0.895,
                "N = 1M, D = 1B tokens; reference = mean training mixture. "
                "Blue lowers loss; red raises loss. Both panels share one colour scale.",
                fontsize=10, ha="left")
    figure.text(0.17, 0.055,
                "(a) Other shares decrease proportionally; source Q scores stay fixed. "
                "(b) Shares stay fixed; each Q change is model-only because a "
                "source's Q was never independently varied in the data.",
                fontsize=9, ha="left", va="bottom", wrap=True)
    return figure


def panel_separate_p_q(data: LawData) -> plt.Figure:
    """Two one-variable cuts: composition at fixed Q, then Q at fixed p.

    The first cut exchanges shares between two sources with exactly the same
    recorded Q, so even the quality-weighted sum stays constant. The second
    cut changes one source's Q in the fitted equation only; it is explicitly
    unvalidated because Q never varies within a source in the fit records.
    """

    training = next(b for b in data.train_blocks if b.name == "train_1m")
    index_f = data.train_names.index("freelaw")
    index_p = data.train_names.index("pile_cc")
    if not np.isclose(data.q[index_f], data.q[index_p], atol=1e-10):
        raise ValueError("the controlled p contrast requires equal source Q")
    other = np.array([i for i in range(len(data.train_names))
                      if i not in (index_f, index_p)])
    reference = np.zeros(len(data.train_names))
    mean_share = training.p.mean(axis=0)
    reference[other] = 0.60 * mean_share[other] / mean_share[other].sum()
    reference[index_f], reference[index_p] = 0.10, 0.30
    n, d = 1e6, 1e9
    base_loss = float(data.predict(n, d, reference[None, :]).mean())

    pile_share = np.linspace(0.02, 0.38, 121)
    changed_p = np.repeat(reference[None, :], len(pile_share), axis=0)
    changed_p[:, index_p] = pile_share
    changed_p[:, index_f] = 0.40 - pile_share
    delta_p = data.predict(n, d, changed_p).mean(axis=1) - base_loss
    if not np.allclose(changed_p @ data.q, reference @ data.q, atol=1e-10):
        raise ValueError("the p cut inadvertently changed weighted quality")

    q_values = np.linspace(0.48, 0.71, 121)
    q_observed = float(data.q[index_p])
    # Exact derivative of the fitted linear pQ block with every p_i held fixed.
    slope = float(np.mean(data.amplitude(n, d)
                          * data.effective_pq[:, index_p] * reference[index_p]))
    delta_q = slope * (q_values - q_observed)

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), sharey=True)
    figure.subplots_adjust(left=0.12, right=0.97, top=0.81, bottom=0.24,
                           wspace=0.10)
    left, right = axes
    for axis in axes:
        axis.axhline(0, color="#4B535A", linewidth=1.0, zorder=1)
        axis.grid(axis="y", color="#D8DDE1", alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=10)
    left.plot(100 * pile_share, delta_p, color="#2C7894", linewidth=3.0)
    left.scatter([30], [0], s=58, color="#2C7894", edgecolors="white",
                 linewidths=0.8, zorder=4)
    left.axvline(30, color="#2C7894", linewidth=0.9, linestyle=":", alpha=0.7)
    left.set_title("(a) Change $p$ · fixed $Q$", loc="left", fontsize=12,
                   fontweight="bold", pad=11)
    left.set_xlabel("Pile-CC share of all tokens (%)", fontsize=11)
    left.set_ylabel(r"Change in mean loss  $\Delta L$  (nats)", fontsize=11)

    right.plot(q_values, delta_q, color="#BF7138", linewidth=2.8,
               linestyle=(0, (6, 3)))
    right.scatter([q_observed], [0], s=62, color="#BF7138",
                  edgecolors="white", linewidths=0.8, zorder=4)
    right.axvline(q_observed, color="#BF7138", linewidth=0.9,
                  linestyle=":", alpha=0.7)
    right.set_title("(b) Change $Q$ · fixed $p$", loc="left", fontsize=12,
                    fontweight="bold", pad=11)
    right.set_xlabel("Hypothetical Pile-CC quality score", fontsize=11)
    lower = min(float(delta_p.min()), float(delta_q.min()))
    upper = max(float(delta_p.max()), float(delta_q.max()))
    padding = 0.12 * (upper - lower)
    left.set_ylim(lower - padding, upper + padding)
    left.set_xlim(1, 39)
    right.set_xlim(0.47, 0.72)
    figure.text(0.12, 0.91,
                f"Fixed scale: N = 1M, D = 1B tokens; reference loss = "
                f"{base_loss:.2f} nats", fontsize=11, ha="left")
    figure.text(0.12, 0.075,
                "Left: FreeLaw + Pile-CC = 40% of tokens; both have Q = 0.61, "
                "and all other shares stay fixed. Right: only Pile-CC Q changes "
                "(observed Q = 0.61). The dashed response is unvalidated, "
                "not an identified quality effect.",
                fontsize=9, ha="left", va="bottom", wrap=True)
    return figure


def panel_simplex_landscape(data: LawData) -> plt.Figure:
    """A controlled two-dimensional slice through the 17-source fitted law.

    Three shares redistribute a fixed 32% of the token budget, while the
    other 68% remains at its mean training composition. Q is fixed per source;
    this is a model counterfactual, not a separately identified Q experiment.
    """

    names = ("github", "pile_cc", "pubmed_central")
    indices = np.array([data.train_names.index(name) for name in names])
    training = next(b for b in data.train_blocks if b.name == "train_1m")
    mean_share = training.p.mean(axis=0)
    free_share = 0.32  # approximately the observed median for these sources
    other = np.array([i for i in range(len(data.train_names)) if i not in indices])
    reference = np.zeros(len(data.train_names))
    reference[other] = (1 - free_share) * mean_share[other] / mean_share[other].sum()

    resolution = 48
    barycentric = np.array(
        [(i / resolution, j / resolution, 1 - (i + j) / resolution)
         for i in range(resolution + 1)
         for j in range(resolution + 1 - i)])
    mixtures = np.repeat(reference[None, :], len(barycentric), axis=0)
    mixtures[:, indices] = free_share * barycentric
    github_corner = reference.copy()
    github_corner[indices[0]] = free_share
    x = barycentric[:, 1] + barycentric[:, 2] / 2
    y = np.sqrt(3) * barycentric[:, 2] / 2
    triangles = mtri.Triangulation(x, y)

    anchors = (("1M parameters · 1B tokens", 1e6, 1e9),
               ("60M parameters · 1B tokens", 6e7, 1e9),
               ("1B parameters · 25B tokens", 1e9, 2.5e10))
    gains, loss_references, optima = [], [], []
    for _, n, d in anchors:
        loss = data.predict(n, d, mixtures).mean(axis=1)
        loss_ref = float(data.predict(n, d, github_corner[None, :]).mean())
        gain = loss_ref - loss
        gains.append(gain)
        loss_references.append(loss_ref)
        optima.append(int(np.argmax(gain)))
    lower = min(0.0, *(float(gain.min()) for gain in gains))
    upper = max(float(gain.max()) for gain in gains)
    normalization = plt.Normalize(vmin=lower, vmax=upper)

    figure, axes = plt.subplots(1, 3, figsize=(13.8, 5.4))
    figure.subplots_adjust(left=0.035, right=0.89, top=0.82, bottom=0.27,
                           wspace=0.03)
    image = None
    for axis, (label, _, _), gain, loss_ref, optimum in zip(
            axes, anchors, gains, loss_references, optima):
        image = axis.tripcolor(triangles, gain, shading="gouraud",
                               cmap="YlGnBu", norm=normalization)
        axis.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0],
                  color="#38434D", linewidth=1.1)
        axis.scatter(x[optimum], y[optimum], marker="*", s=170,
                     color="#EA8C24", edgecolors="#313A40", linewidths=0.6,
                     zorder=6)
        axis.set_xlim(-0.17, 1.17)
        axis.set_ylim(-0.17, 1.07)
        axis.set_aspect("equal", adjustable="box")
        axis.axis("off")
        axis.text(0.5, 1.085, label, transform=axis.transAxes,
                  ha="center", va="bottom", fontsize=10.5, fontweight="bold")
        axis.text(0.5, 1.015, f"Reference loss = {loss_ref:.2f}",
                  transform=axis.transAxes, ha="center", va="bottom",
                  fontsize=9.5, color="#46525B")
        axis.text(-0.02, -0.075, f"GitHub\nQ={data.q[indices[0]]:.2f}",
                  ha="center", va="top", fontsize=9.3)
        axis.text(1.02, -0.075, f"Pile-CC\nQ={data.q[indices[1]]:.2f}",
                  ha="center", va="top", fontsize=9.3)
        axis.text(0.5, np.sqrt(3)/2 + 0.03,
                  f"PubMed Central\nQ={data.q[indices[2]]:.2f}",
                  ha="center", va="bottom", fontsize=9.3)
        axis.text(0.5, -0.19,
                  f"Slice minimum L = {loss_ref-gain[optimum]:.2f} nats",
                  ha="center", va="top", fontsize=9.6)

    colorbar = figure.colorbar(image, ax=list(axes), fraction=0.035,
                               pad=0.005, aspect=23)
    colorbar.set_label("Loss reduction vs GitHub corner (nats)", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    figure.text(0.035, 0.06,
                "Triangle position allocates 32% of tokens among the three named "
                "sources; the other 68% is fixed. Star: minimum on this model "
                "slice. Each panel uses the same colour scale. This is a model "
                "counterfactual, not direct validation or a causal Q effect.",
                ha="left", va="bottom", fontsize=9, wrap=True)
    return figure


def panel_four_factor_map(data: LawData) -> plt.Figure:
    """Aligned N/D response matrix for three real, held-out compositions.

    The 1M and 60M cells have paired observed losses.  The same mixtures were
    not measured at 1B, so the final column is explicitly model-only.  Both
    feature bars use the same, unnormalised 0--100 scale: rescaling pQ to sum
    to 100 would erase the magnitude of the quality-weighted exposure.
    """

    one = next(b for b in data.test_blocks if b.name == "test_1m")
    sixty = next(b for b in data.test_blocks if b.name == "test_60m")
    if one.ids != sixty.ids or not np.allclose(one.p, sixty.p, atol=1e-9):
        raise ValueError("1M and 60M held-out compositions must be paired")
    p_all = one.p / one.p.sum(axis=1, keepdims=True)
    ordered = np.argsort(_entropy(p_all))
    picked = [int(ordered[round(fraction * (len(ordered) - 1))])
              for fraction in (0.1, 0.5, 0.9)]
    if len(set(picked)) != 3:
        raise ValueError("not enough distinct held-out mixtures")
    shares = p_all[picked]
    raw_shares = one.p[picked]
    top = np.argsort(-shares.sum(axis=0))[:6]
    remainder = np.array([i for i in range(shares.shape[1]) if i not in top])
    groups = [np.array([i]) for i in top] + [remainder]
    share_groups = 100 * np.stack([shares[:, group].sum(axis=1)
                                    for group in groups], axis=1)
    weighted_groups = 100 * np.stack([(shares[:, group] * data.q[group]).sum(axis=1)
                                       for group in groups], axis=1)
    if np.any(weighted_groups.sum(axis=1) > 100.01):
        raise ValueError("quality-weighted exposure exceeds the common bar scale")

    anchors = [(one.n, one.d), (sixty.n, sixty.d),
               (1e9, 2.5e10)]
    predicted = np.column_stack(
        [data.predict(n, d, raw_shares).mean(axis=1) for n, d in anchors])
    observed = np.column_stack([one.y[picked].mean(axis=1),
                                sixty.y[picked].mean(axis=1)])

    figure = plt.figure(figsize=(13.5, 7.4))
    grid = figure.add_gridspec(3, 2, left=0.12, right=0.91, top=0.76,
                               bottom=0.29, width_ratios=(1.24, 1.0),
                               wspace=0.10, hspace=0.48)
    colors = [plt.get_cmap("tab10")(i) for i in range(6)] + ["#C5C8CB"]
    row_names = ("Low diversity", "Median diversity", "High diversity")
    for row in range(3):
        axis = figure.add_subplot(grid[row, 0])
        for y, values in ((1.0, share_groups[row]),
                          (0.0, weighted_groups[row])):
            start = 0.0
            for value, color in zip(values, colors):
                axis.barh(y, value, left=start, height=0.38, color=color,
                          edgecolor="white", linewidth=0.55)
                start += value
        axis.set_xlim(0, 100)
        axis.set_ylim(-0.38, 1.38)
        axis.set_yticks([1, 0], labels=[r"$p_i$", r"$p_iQ_i$"])
        axis.set_xticks([0, 25, 50, 75, 100])
        if row != 2:
            axis.tick_params(axis="x", labelbottom=False)
        else:
            axis.set_xlabel(r"Input magnitude  $100p_i$ or $100p_iQ_i$",
                            fontsize=10)
        axis.tick_params(axis="both", labelsize=9, length=2)
        axis.grid(axis="x", alpha=0.40, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        entropy = _entropy(shares[row])[0]
        exposure = weighted_groups[row].sum() / 100
        axis.set_title(f"Mix {row + 1}: {row_names[row]}   H={entropy:.2f}   "
                       + r"$\sum_i p_iQ_i$" + f"={exposure:.2f}",
                       loc="left", fontsize=10, pad=5)

    heat = figure.add_subplot(grid[:, 1])
    image = heat.imshow(predicted, cmap="YlGnBu", aspect="auto",
                        vmin=float(predicted.min()) - 0.05,
                        vmax=float(predicted.max()) + 0.05)
    heat.set_xticks(range(3), labels=["1M\n1B tokens", "60M\n1B tokens",
                                      "1B\n25B tokens"])
    heat.xaxis.tick_top()
    heat.tick_params(axis="x", labelsize=10, length=0, pad=6)
    heat.set_yticks(range(3), labels=["Mix 1", "Mix 2", "Mix 3"])
    heat.tick_params(axis="y", labelsize=10, length=0, pad=8)
    heat.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    heat.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    heat.grid(which="minor", color="white", linewidth=3)
    heat.tick_params(which="minor", bottom=False, left=False)
    heat.set_xlabel("Model parameters / training tokens", fontsize=10,
                    labelpad=9)
    for row in range(3):
        for column in range(3):
            value = predicted[row, column]
            text_color = "white" if value > 0.5 * (predicted.min() + predicted.max()) \
                else "#152333"
            heat.text(column, row - 0.10, f"{value:.2f}", ha="center",
                      va="center", fontsize=15, fontweight="bold",
                      color=text_color)
            sublabel = (f"obs {observed[row, column]:.2f}"
                        if column < 2 else "model only")
            heat.text(column, row + 0.20, sublabel, ha="center", va="center",
                      fontsize=8.5, color=text_color)
    colorbar = figure.colorbar(image, ax=heat, fraction=0.045, pad=0.025)
    colorbar.set_label("Predicted mean loss (nats; 13 domains)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    handles = [Patch(facecolor=colors[i], label=f"{data.train_names[source]} "
                     f"(Q={data.q[source]:.2f})")
               for i, source in enumerate(top)]
    handles.append(Patch(facecolor=colors[-1], label="Other sources"))
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                  bbox_to_anchor=(0.51, 0.13), fontsize=8.8,
                  columnspacing=1.5, handlelength=1.5)
    figure.text(0.12, 0.94, "Input: composition and fixed quality",
                fontsize=12, fontweight="bold", ha="left")
    figure.text(0.56, 0.94, "Output: predicted mean loss",
                fontsize=12, fontweight="bold", ha="left")
    figure.text(0.12, 0.045,
                "Rows: held-out mixtures selected by entropy. Q is fixed by source; "
                "pQ is a feature, not an isolated quality effect. "
                "The 1B column transfers these same mixtures without matching observations; "
                "N and D also change together.",
                fontsize=8.5, ha="left", va="bottom", wrap=True)
    return figure


# --------------------------------------------------------------------------- #
# Chinese labels
# --------------------------------------------------------------------------- #
# The repository no longer ships a font (the bundled subset was removed) and this
# machine has no CJK family installed, so a Chinese panel needs a real font on
# disk.  Registration has to happen before the figure is *drawn*: matplotlib
# resolves families at draw time, so a context manager around the builder would
# not reach the legend that savefig creates lazily.
CJK_FONT_CANDIDATES = (
    Path(os.environ["DSH_CJK_FONT"]) if os.environ.get("DSH_CJK_FONT") else None,
    PROJECT_ROOT / ".fontwork" / "NotoSansCJKsc-Regular.otf",
    PROJECT_ROOT / "q2" / "assets" / "NotoSansSC-Regular-subset.otf",
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
)

CJK_HELP = (
    "中文标注需要一份 CJK 字体：本机没有安装，仓库也不再随附。取一份后重试：\n"
    "  pip download mplfonts -d .fontwork --no-deps\n"
    "  python -c \"import zipfile,glob,pathlib;"
    "z=zipfile.ZipFile(glob.glob('.fontwork/mplfonts-*.whl')[0]);"
    "pathlib.Path('.fontwork/NotoSansCJKsc-Regular.otf')"
    ".write_bytes(z.read('mplfonts/fonts/NotoSansCJKsc-Regular.otf'))\"\n"
    "或指定已有字体：DSH_CJK_FONT=/path/to/font.otf"
)


def chinese_family() -> str:
    """Register an available CJK font and return its family name.

    Fails loudly when none is found: without a CJK family matplotlib only emits a
    ``UserWarning`` and draws every Chinese glyph as a tofu box.
    """

    from matplotlib import font_manager

    for candidate in CJK_FONT_CANDIDATES:
        if candidate is not None and Path(candidate).exists():
            font_manager.fontManager.addfont(str(candidate))
            return font_manager.FontProperties(fname=str(candidate)).get_name()
    raise SystemExit("未找到可用的中文字体。\n" + CJK_HELP)


def use_chinese(figure: plt.Figure) -> plt.Figure:
    """Tag a figure so ``main`` draws its text with the CJK family."""

    figure.dsh_font_family = chinese_family()
    return figure


# Canvas and font sizes are the house standard already: the global rcParams set
# axes/legend to 14/12 and ``axes.unicode_minus`` off, so only the family differs
# between the Chinese and English panels.
PARITY_TEXT = {
    False: {
        "baseline": "law with $\\Phi_k\\equiv0$",
        "scale": "{label} held-out",
        "xlabel": "Observed loss  (val_loss)",
        "ylabel": "Predicted loss  (val_loss)",
        "note_left": ("n = {n} held-out (mixture, domain)\n"
                      "relative RMSE {rmse:.2f}%  (with $\\Phi_k\\equiv0$: "
                      "{base:.2f}%)\n"
                      "shaded band: $\\pm$5%\n"
                      "held-out folds: no fit, no selection"),
        "bar_base": "identified law with $\\Phi_k\\equiv0$",
        "bar_law": "identified law",
        "xtick": "{label} held-out",
        "ylabel_right": "Held-out relative RMSE (%)",
        "note_right": ("same coefficients, mixture term switched off:\n"
                       "the baseline is flat in $\\mathbf{p}$, so the\n"
                       "whole gap is what $S_k\\Phi_k$ identifies"),
        "gain": "{gain:.1f}x",
    },
    True: {
        "baseline": "同一系数",
        "scale": "{label} 留出集",
        "xlabel": "实测验证损失",
        "ylabel": "预测验证损失",
        "note_left": ("留出集 n = {n}（配比 × 评测域）\n"
                      "相对 RMSE {rmse:.2f}%（关闭配比项：{base:.2f}%）\n"
                      "阴影带：$\\pm$5%\n"
                      "留出折不参与拟合与选模"),
        "bar_base": "同一系数，关闭配比项",
        "bar_law": "辨识所得模型",
        "xtick": "{label} 留出集",
        "ylabel_right": "留出集相对 RMSE（%）",
        "note_right": ("同一套系数，仅关闭配比项：\n"
                       "基线在配比 $\\mathbf{p}$ 上恒定，\n"
                       "整段落差即 $S_k\\Phi_k$ 的辨识贡献"),
        "gain": "{gain:.1f}×",
    },
}


# --------------------------------------------------------------------------- #
# identification-effect figures
# --------------------------------------------------------------------------- #
# Two 6x4 panels side by side, matching the paper convention for composites.
COMPOSITE_FIGSIZE = (12.0, 4.0)


def _heldout_pairs(data: LawData) -> list:
    order = {name: index for index, name in enumerate(HELDOUT_SCALES)}
    return sorted((block for block in data.test_blocks if block.name in HELDOUT_SCALES),
                  key=lambda block: order[block.name])


def _mixture_r2_per_domain(pairs: list) -> np.ndarray:
    """Per-evaluation-domain mixture R², centring inside each (domain, scale) cell.

    ``compact.mixture_r_squared_within_scale`` pools the two sums over scales and
    returns the aggregate; keeping them per column answers the different question
    "which evaluation domain did the identified response actually resolve?".
    """

    numerator = denominator = None
    for block, predicted in pairs:
        observed = block.y - block.y.mean(axis=0, keepdims=True)
        fitted = predicted - predicted.mean(axis=0, keepdims=True)
        num = np.square(observed - fitted).sum(axis=0)
        den = np.square(observed).sum(axis=0)
        numerator = num if numerator is None else numerator + num
        denominator = den if denominator is None else denominator + den
    return 1.0 - numerator / denominator


def _flat_baseline(data: LawData, block) -> np.ndarray:
    """The no-mixture baseline repeated for every mixture at that scale."""

    return np.tile(data.baseline(block.n, block.d), (len(block.ids), 1))


def _parity_figure(data: LawData, *, chinese: bool) -> plt.Figure:
    """Shared builder for the held-out parity figure.

    Left: every held-out mixture against its prediction, with the same
    coefficients but the mixture term switched off drawn as grey crosses.  That
    baseline carries no mixture information, so its points collapse onto 13 flat
    levels per scale, while the identified law tracks the diagonal.

    Right: the same two models compared scale by scale, which is where the
    identification gain can be read off without pooling folds.

    ``chinese`` selects the label set; the two delivered names pass the same
    value so their PNGs cannot drift apart.
    """

    text = PARITY_TEXT[chinese]
    pairs = [(block, data.predict(block.n, block.d, block.p))
             for block in _heldout_pairs(data)]
    figure, (left) = plt.subplots(1, 1, figsize=COMPOSITE_FIGSIZE,
                                         layout="constrained")

    observed = np.concatenate([block.y.ravel() for block, _ in pairs])
    baseline = np.concatenate([_flat_baseline(data, block).ravel()
                               for block, _ in pairs])
    predicted = np.concatenate([value.ravel() for _, value in pairs])
    lo, hi = float(observed.min()), float(observed.max())
    span = hi - lo
    grid = np.linspace(lo - 0.03 * span, hi + 0.03 * span, 64)
    left.fill_between(grid, grid * 0.95, grid * 1.05, color="#E8EEF6", zorder=0,
                      linewidth=0)
    left.plot(grid, grid, color="#444444", linewidth=1.1, zorder=3)
    left.scatter(observed, baseline, s=13, marker="+", color="#9A9A9A",
                 linewidths=0.8, zorder=1, label=text["baseline"])
    for block, value in pairs:
        colour, marker = SCALE_STYLE[block.name]
        left.scatter(block.y.ravel(), value.ravel(), s=17, marker=marker,
                     facecolors="none", edgecolors=colour, linewidths=0.9,
                     zorder=4,
                     label=text["scale"].format(label=SCALE_LABEL[block.name]))
    left.set_xlim(grid[0], grid[-1])
    left.set_ylim(grid[0], grid[-1])
    left.set_aspect("equal", adjustable="box")
    left.set_xlabel(text["xlabel"])
    left.set_ylabel(text["ylabel"])
    tidy(left)
    # The note owns the upper-left corner, so the legend goes to the free corner
    # below the diagonal; the +-5% band is explained in the note instead of as a
    # separate label that would collide with either.  Frameless legends let the
    # marker cloud show through the labels, so both panels carry a background box.
    left.legend(loc="lower right", fontsize=12, handletextpad=0.5,
                labelspacing=0.35, borderpad=0.4, frameon=True, framealpha=0.92,
                edgecolor="#CCCCCC")
    # note(left,
    #      text["note_left"].format(
    #          n=len(observed),
    #          rmse=compact.relative_rmse(observed, predicted) * 100,
    #          base=compact.relative_rmse(observed, baseline) * 100),
    #      loc="upper left", size=11)

    labels = [SCALE_LABEL[block.name] for block, _ in pairs]
    base_error = [compact.relative_rmse(block.y, _flat_baseline(data, block)) * 100
                  for block, _ in pairs]
    law_error = [compact.relative_rmse(block.y, value) * 100 for block, value in pairs]
    x = np.arange(len(labels))
    width = 0.36
    # right.bar(x - width / 2, base_error, width, color=BAR_COLOR_MUTED,
    #           label=text["bar_base"])
    # right.bar(x + width / 2, law_error, width, color=BAR_COLOR,
    #           label=text["bar_law"])
    # for position, base, law in zip(x, base_error, law_error):
    #     right.text(position - width / 2, base + 0.25, f"{base:.2f}", ha="center",
    #                va="bottom", fontsize=11, color="#555555")
    #     right.text(position + width / 2, law + 0.25, f"{law:.2f}", ha="center",
    #                va="bottom", fontsize=11)
    #     # The gain goes inside the bar: the top-left corner belongs to the legend.
    #     right.text(position + width / 2, law / 2, text["gain"].format(gain=base / law),
    #                ha="center", va="center", fontsize=12, color="white",
    #                fontweight="bold")
    # right.set_xticks(x)
    # right.set_xticklabels([text["xtick"].format(label=label) for label in labels])
    # right.set_ylabel(text["ylabel_right"])
    # right.set_ylim(0, max(base_error) * 1.42)
    # tidy(right, grid_axis="y")
    # right.legend(loc="upper left", fontsize=12, borderpad=0.4, frameon=True,
    #              framealpha=0.92, edgecolor="#CCCCCC")
    # note(right, text["note_right"], loc="upper right", size=11)
    return use_chinese(figure) if chinese else figure


def panel_identification_parity(data: LawData) -> plt.Figure:
    """Held-out parity of the identified law, beside the per-scale error it buys."""

    return _parity_figure(data, chinese=True)


def _domain_accuracy(data: LawData) -> tuple[list[str], np.ndarray, np.ndarray,
                                              np.ndarray]:
    """Per-evaluation-domain held-out error and two R² readings.

    ``relative_rmse`` and ``r_squared_overall`` pool the three held-out scales, so
    their total sum of squares carries the between-scale loss difference and a
    model that only tracks the scale trend already scores high.
    ``r_squared_mixture`` centres inside every (domain, scale) cell and therefore
    measures the mixture response alone; the gap between the two is the scale
    trend.
    """

    pairs = [(block, data.predict(block.n, block.d, block.p))
             for block in _heldout_pairs(data)]
    observed = np.vstack([block.y for block, _ in pairs])
    predicted = np.vstack([value for _, value in pairs])
    relative, overall = [], []
    for k in range(observed.shape[1]):
        actual, fitted = observed[:, k], predicted[:, k]
        relative.append(float(np.sqrt(np.mean(np.square(fitted / actual - 1))) * 100))
        overall.append(float(1.0 - np.square(fitted - actual).sum()
                             / np.square(actual - actual.mean()).sum()))
    return (list(data.loss_names), np.asarray(relative), np.asarray(overall),
            _mixture_r2_per_domain(pairs))


def panel_domain_accuracy(data: LawData) -> plt.Figure:
    """Per-domain held-out relative RMSE and R² in a single panel.

    Bars read the left axis (relative RMSE per evaluation domain, all three
    held-out scales pooled); the two R² readings are dots on the right axis.
    Carries no annotation box: the axis labels and the legend are the only text
    besides the numbers themselves.
    """

    names, relative, overall, mixture = _domain_accuracy(data)
    order = np.argsort(-relative)          # worst error first
    labels = [names[index] for index in order]
    relative, overall, mixture = relative[order], overall[order], mixture[order]
    x = np.arange(len(labels))

    figure, left = plt.subplots(figsize=PANEL_FIGSIZE, layout="constrained")
    right = left.twinx()
    right.patch.set_visible(False)

    left.bar(x, relative, 0.68, color=BAR_COLOR_ALT, zorder=2,
             label="相对 RMSE")
    # No per-bar numbers: with 13 domains the R² dots land on top of them (and the
    # first bar's label runs off the axis), so the values are read off the ticks.

    # The two R² readings sit ~0.07 apart, so they are dots rather than bars: a
    # bar read from zero would flatten exactly that difference.
    right.vlines(x, mixture, overall, color="#DDDDDD", linewidth=2.2, zorder=3)
    right.scatter(x, overall, s=42, color=BAR_COLOR_MUTED, zorder=4,
                  label="总体 $R^2$")
    right.scatter(x, mixture, s=42, color=BAR_COLOR, zorder=4,
                  label="配比效应 $R^2$")

    left.set_xticks(x)
    left.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    left.set_ylabel("留出集相对 RMSE（%）")
    right.set_ylabel("留出集 $R^2$")
    left.set_ylim(0, float(relative.max()) * 1.22)
    # The right axis starts at 0.80, not 0: the dots carry the value by position,
    # not by length, and its tick labels state the range.  On a 0-1 axis every dot
    # would be pressed against the top edge and the two readings would overlap.
    right.set_ylim(0.80, 1.0)
    right.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    left.grid(True, axis="y", color="#D9D9D9", linewidth=0.6)
    left.set_axisbelow(True)
    # One shared frame: the twin axis draws the right spine, the primary the left.
    for axis in (left, right):
        axis.spines["top"].set_visible(False)
    left.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)
    right.grid(False)
    handles = left.get_legend_handles_labels()
    other = right.get_legend_handles_labels()
    left.legend(handles[0] + other[0], handles[1] + other[1], loc="lower center",
                bbox_to_anchor=(0.5, 1.02), ncols=3, fontsize=10, frameon=False)
    return use_chinese(figure)


def panel_identification_resolution(data: LawData) -> plt.Figure:
    """What the identification resolved: per-domain mixture R² and both exponents.

    Left: per-evaluation-domain mixture R² on the held-out folds — how much of
    each domain's within-scale loss variation the identified response explains.
    Pooling these cells gives the headline 0.9006.

    Right: the identified pair (eta_k, zeta_k) per domain.  Both exponents are
    solved from the three scale anchors alone, so this panel shows *what was
    resolved*, not how well it generalises; the dotted line is eta = zeta.
    """

    pairs = [(block, data.predict(block.n, block.d, block.p))
             for block in _heldout_pairs(data)]
    r_squared = _mixture_r2_per_domain(pairs)
    order = np.argsort(-r_squared)
    figure, (left, right) = plt.subplots(1, 2, figsize=COMPOSITE_FIGSIZE,
                                         layout="constrained")

    x = np.arange(len(order))
    values = r_squared[order]
    weak = values < 0.75
    left.bar(x[~weak], values[~weak], 0.68, color=BAR_COLOR)
    left.bar(x[weak], values[weak], 0.68, color=ACCENT)
    for position, value in zip(x, values):
        left.text(position, value + 0.015, f"{value:.2f}", ha="center", va="bottom",
                  fontsize=8.0, rotation=90, color="#333333")
    left.set_xticks(x)
    left.set_xticklabels([data.loss_names[index] for index in order], rotation=45,
                         ha="right", fontsize=9.5)
    # Every domain lands in 0.86-0.95, so a mean reference line would only cross
    # the bars; the contrast that carries information is the flat baseline, which
    # scores exactly 0 here, and it goes in the note.
    left.set_ylim(0, 1.30)
    left.set_ylabel("Held-out mixture $R^2$ per domain")
    tidy(left, grid_axis="y")
    flat = _mixture_r2_per_domain([(block, _flat_baseline(data, block))
                                   for block, _ in pairs])
    note(left,
         f"pooled over all cells  "
         f"{data.law['aggregate_heldout']['mixture_r_squared_within_scale']:.4f}\n"
         f"range across domains  {values.min():.2f} - {values.max():.2f}\n"
         f"with $\\Phi_k\\equiv0$:  {flat.mean():.2f} (all 13 domains)",
         loc="upper right", size=9.5)

    eta, zeta = data.eta, data.zeta
    top = float(max(eta.max(), zeta.max())) * 1.12
    right.fill_between([0, top], [0, top], [top, top], color="#F4F6F9", zorder=0,
                       linewidth=0)
    right.plot([0, top], [0, top], color="#666666", linewidth=1.1,
               linestyle=":", zorder=2)
    right.scatter(eta, zeta, s=46, facecolors="none", edgecolors=BAR_COLOR,
                  linewidths=1.4, zorder=4)
    # The highest point sits at the top of the frame, so its label goes below it;
    # that in turn frees the upper-left corner for the separation counts.
    marked = {int(np.argmax(eta)), int(np.argmax(zeta)),
              int(np.argmin(zeta - eta))}
    for index in marked:
        below = index == int(np.argmax(zeta))
        # Mark the single domain that falls on the other side of the diagonal
        # where it actually sits, instead of in a corner label that would land
        # on top of a marker.
        exception = index == int(np.argmin(zeta - eta))
        if exception:
            # One line, to the right: the diagonal runs up-left of this point and
            # a two-line block would reach down into the axis.
            # Short enough to stay left of the note box; the coordinates are in
            # the parameter tables, and the name is what identifies the point.
            text = f"{data.loss_names[index]}   $\\zeta_k<\\eta_k$"
            offset, align = (9, -3), "left"
        elif below:             # the tallest point sits at the top of the frame
            text = f"{data.loss_names[index]}\n({eta[index]:.3f}, {zeta[index]:.3f})"
            offset, align = (9, -19), "left"
        else:
            text = f"{data.loss_names[index]}\n({eta[index]:.3f}, {zeta[index]:.3f})"
            offset, align = (8, 7), "left"
        right.annotate(text, xy=(eta[index], zeta[index]),
                       xytext=offset, textcoords="offset points",
                       ha=align, va="top" if offset[1] < 0 else "baseline",
                       fontsize=9,
                       color=ACCENT if exception else "#333333", linespacing=1.3)
    right.set_xlim(0, top)
    right.set_ylim(0, top)
    right.set_xlabel("$\\eta_k$  (parameter-count decay)")
    right.set_ylabel("$\\zeta_k$  (token-count decay)")
    tidy(right)
    note(right,
         "solved from the 3 anchors only:\n"
         "zero residual, no held-out check\n"
         "$\\zeta_k>\\eta_k$ in 12/13 domains\n"
         "PCXI: $\\eta\\neq\\zeta$ for 9/13",
         loc="lower right", size=9.5)
    return figure


BUILDERS = {
    "all_domain_p_q": panel_all_domain_p_q,
    "separate_p_q": panel_separate_p_q,
    "scaling_curves": panel_scaling_curves,
    "simplex_landscape": panel_simplex_landscape,
    "quality_pathway": panel_quality_pathway,
    "observed_contrast": panel_observed_contrast,
    "component_breakdown": panel_component_breakdown,
    "domain_scale": panel_domain_scale,
    "four_factor_map": panel_four_factor_map,
    "loss_vs_diversity": panel_loss_vs_diversity,
    "surface_parity": panel_surface_parity,
    "response_parity": panel_response_parity,
    "error_by_scale": panel_error_by_scale,
    "amplitude_decay": panel_amplitude_decay,
    "mixture_response": panel_mixture_response,
    "weight_heatmap": panel_weight_heatmap,
    "structure_diagnostics": panel_structure_diagnostics,
    "ridge_ladder": panel_ridge_ladder,
    "elasticity": panel_elasticity,
    "epsilon_sensitivity": panel_epsilon_sensitivity,
    "domain_error_1b": panel_domain_error_1b,
    "identification_parity": panel_identification_parity,
    "domain_accuracy": panel_domain_accuracy,
    "identification_resolution": panel_identification_resolution,
}
STORY_FIGURES = ("all_domain_p_q", "quality_pathway", "observed_contrast",
                 "component_breakdown", "domain_scale", "surface_parity")


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def verify_against_json(data: LawData) -> None:
    """The panels must reproduce the stored held-out metrics; fail loudly if not."""

    stored = data.law["heldout_test"]
    recomputed = data.per_scale(list(data.test_blocks))
    problems = []
    for name, values in recomputed.items():
        for key in ("relative_rmse", "median_ape"):
            expected, actual = stored[name][key], values[key]
            if abs(expected - actual) > 1e-6:
                problems.append(f"{name}.{key}: json {expected:.8f} vs {actual:.8f}")
    print("held-out reproduction check")
    for name in HELDOUT_SCALES:
        print(f"  {name:<9} relative RMSE {recomputed[name]['relative_rmse']:.4%}"
              f"   median APE {recomputed[name]['median_ape']:.4%}"
              f"   (n={stored[name]['n_mixtures']})")
    if problems:
        raise SystemExit("held-out metrics do not reproduce:\n  " + "\n  ".join(problems))
    print("  -> matches generalized_scaling_compact.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figures", default="story",
                        help="'story' (default), 'all', or comma-separated panel names")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", default="png",
                        help="comma separated; the figure set is published as png")
    parser.add_argument("--skip-verify", action="store_true")
    arguments = parser.parse_args()

    font = configure_fonts()
    data = LawData()
    if not arguments.skip_verify:
        verify_against_json(data)

    names = list(BUILDERS) if arguments.figures == "all" else \
        list(STORY_FIGURES) if arguments.figures == "story" else \
        [item.strip() for item in arguments.figures.split(",") if item.strip()]
    unknown = [name for name in names if name not in BUILDERS]
    if unknown:
        raise SystemExit(f"unknown panel(s): {unknown}; available: {list(BUILDERS)}")
    extensions = [item.strip().lstrip(".") for item in arguments.formats.split(",")
                  if item.strip()]

    # 解析成绝对路径：--output-dir 允许传相对路径，而下面打印时要用它相对仓库根取短名。
    arguments.output_dir = arguments.output_dir.expanduser().resolve()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nfont: {font}\noutput: {arguments.output_dir}")
    for name in names:
        figure = BUILDERS[name](data)
        # A Chinese panel is tagged with its family by the builder; draw time is
        # here, not there, so the family has to be applied around savefig.
        family = getattr(figure, "dsh_font_family", None)
        for extension in extensions:
            path = arguments.output_dir / f"generalized_law_{name}.{extension}"
            # A missing CJK glyph is only a UserWarning and the result is a box
            # where text should be, so a Chinese panel renders to a scratch file
            # first: checking after writing ``path`` would leave the delivered
            # figure already overwritten with tofu when the check fails.
            scratch = path.with_name(f"{path.stem}.tmp{path.suffix}") if family else path
            with plt.rc_context({"font.sans-serif": [family]} if family else {}):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    figure.savefig(
                        scratch, dpi=arguments.dpi if extension == "png" else None)
            if family:
                missing = [str(item.message) for item in caught
                           if "missing from font" in str(item.message)]
                if missing:
                    scratch.unlink(missing_ok=True)
                    raise SystemExit(
                        f"{name}: 字体 {family} 未覆盖全部用字，中文会画成方框，"
                        f"未写出 {path.name}：\n  " + "\n  ".join(missing[:4]))
                os.replace(scratch, path)
            print(f"  wrote {path.relative_to(PROJECT_ROOT)}")
        plt.close(figure)


if __name__ == "__main__":
    main()
