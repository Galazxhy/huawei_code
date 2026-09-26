"""Fit Scheffe mixture models (first-order, multi-task sparse, ILR-Ridge)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, MultiTaskElasticNet, Ridge
from sklearn.model_selection import KFold


OUT = Path(__file__).resolve().parent / "results"
SEED = 20260923


def macro_rmse(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=0))))


def macro_r2(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    ss_res = np.sum((Y_true - Y_pred) ** 2, axis=0)
    ss_tot = np.sum((Y_true - Y_true.mean(axis=0)) ** 2, axis=0)
    return float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-9)))


def clr_transform(P: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    P_pos = np.clip(P, eps, None)
    P_pos = P_pos / P_pos.sum(axis=1, keepdims=True)
    logp = np.log(P_pos)
    return logp - logp.mean(axis=1, keepdims=True)


def main() -> None:
    X1 = pd.read_csv(OUT / "X1_train_1m.csv").to_numpy(dtype=float)
    X2 = pd.read_csv(OUT / "X2_train_1m.csv").to_numpy(dtype=float)
    Y = pd.read_csv(OUT / "Z_train.csv").to_numpy(dtype=float)
    X1_test = pd.read_csv(OUT / "X1_test_1m.csv").to_numpy(dtype=float)
    X2_test = pd.read_csv(OUT / "X2_test_1m.csv").to_numpy(dtype=float)
    Y_test = pd.read_csv(OUT / "Z_test_1m.csv").to_numpy(dtype=float)

    # First-order Scheffe baseline.
    lin = LinearRegression(fit_intercept=False)
    lin.fit(X1, Y)
    lin_pred_test = lin.predict(X1_test)

    # Multi-task sparse Scheffe with internal CV.
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    alphas = np.logspace(-4.5, -1.0, 12)
    l1_ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    best = None
    best_score = np.inf
    cv_rows = []
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            fold_errors = []
            for tr, va in kf.split(X2):
                model = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000,
                                            tol=1e-5, fit_intercept=False, random_state=SEED)
                model.fit(X2[tr], Y[tr])
                fold_errors.append(macro_rmse(Y[va], model.predict(X2[va])))
            score = float(np.mean(fold_errors))
            cv_rows.append({"alpha": alpha, "l1_ratio": l1_ratio, "cv_macro_rmse": score})
            if score < best_score:
                best_score = score
                best = (alpha, l1_ratio)
    cv_table = pd.DataFrame(cv_rows)
    cv_table.to_csv(OUT / "scheffe_multitask_cv.csv", index=False, encoding="utf-8-sig")
    alpha, l1_ratio = best
    sparse = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000,
                                 tol=1e-5, fit_intercept=False, random_state=SEED)
    sparse.fit(X2, Y)
    sparse_pred_test = sparse.predict(X2_test)

    # ILR-Ridge baseline with internal CV.
    Xc_train = clr_transform(pd.read_csv(OUT / "P_train_1m.csv").to_numpy(dtype=float))
    Xc_test = clr_transform(pd.read_csv(OUT / "P_test_1m.csv").to_numpy(dtype=float))
    ridge_alphas = np.logspace(-3, 2, 20)
    best_ridge = None
    best_ridge_score = np.inf
    for a in ridge_alphas:
        fold_errors = []
        for tr, va in kf.split(Xc_train):
            ridge = Ridge(alpha=a, random_state=SEED)
            ridge.fit(Xc_train[tr], Y[tr])
            fold_errors.append(macro_rmse(Y[va], ridge.predict(Xc_train[va])))
        score = float(np.mean(fold_errors))
        if score < best_ridge_score:
            best_ridge_score = score
            best_ridge = a
    ridge = Ridge(alpha=best_ridge, random_state=SEED)
    ridge.fit(Xc_train, Y)
    ridge_pred_test = ridge.predict(Xc_test)

    models = {
        "first_order_scheffe": lin_pred_test,
        "multitask_sparse_scheffe": sparse_pred_test,
        "ilr_ridge": ridge_pred_test,
    }
    rows = []
    for name, pred in models.items():
        rows.append({
            "model": name,
            "test_macro_rmse": macro_rmse(Y_test, pred),
            "test_macro_r2": macro_r2(Y_test, pred),
            "test_mean_abs_error": float(np.mean(np.abs(Y_test - pred))),
        })
    test_metrics = pd.DataFrame(rows)
    test_metrics.to_csv(OUT / "scheffe_model_test_metrics.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(sparse_pred_test, columns=pd.read_csv(OUT / "loss_domain_names.csv")["loss_domain"]).to_csv(
        OUT / "pred_test_1m_multitask.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(lin_pred_test, columns=pd.read_csv(OUT / "loss_domain_names.csv")["loss_domain"]).to_csv(
        OUT / "pred_test_1m_first_order.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ridge_pred_test, columns=pd.read_csv(OUT / "loss_domain_names.csv")["loss_domain"]).to_csv(
        OUT / "pred_test_1m_ilr.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sparse.coef_.T, index=pd.read_csv(OUT / "scheffe_feature_names.csv")["feature"],
                 columns=pd.read_csv(OUT / "loss_domain_names.csv")["loss_domain"]).to_csv(
        OUT / "B_multitask_sparse_scheffe.csv", encoding="utf-8-sig")
    pd.DataFrame(lin.coef_.T, index=pd.read_csv(OUT / "train_domain_names.csv")["domain"],
                 columns=pd.read_csv(OUT / "loss_domain_names.csv")["loss_domain"]).to_csv(
        OUT / "B_first_order_scheffe.csv", encoding="utf-8-sig")

    summary = {
        "best_multitask": {"alpha": float(alpha), "l1_ratio": float(l1_ratio), "cv_macro_rmse": best_score},
        "best_ridge": {"alpha": float(best_ridge), "cv_macro_rmse": best_ridge_score},
        "test_metrics": rows,
        "feature_count": X2.shape[1],
    }
    (OUT / "scheffe_fit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(test_metrics.to_string(index=False))
    print("best multitask:", best, "cv_rmse", best_score)
    print("best ridge:", best_ridge, "cv_rmse", best_ridge_score)


if __name__ == "__main__":
    main()
