#!/usr/bin/env python3
"""
Revised, publication-grade PSNN experiment for the UCI Raisin dataset
(Kecimen = 0, Besni = 1).

What changed relative to the original manuscript code
-----------------------------------------------------
1. NO DATA LEAKAGE: Min-Max scaling is fitted on each training fold only.
2. Repeated stratified 5-fold CV (default 10 repeats = 50 fold-runs) with
   identical splits for EVERY model (PSNN and all baselines).
3. Baselines re-run by us under the same protocol (LR, quadratic LR, SVM-RBF,
   MLP, Random Forest) -> a fair comparison instead of copying literature numbers.
4. PSNN improvements: binary cross-entropy, mini-batch Adam, learnable output
   scale/bias, bias-1 initialisation (Shin & Ghosh style), L2, early stopping.
5. K (number of summing units) ablation: K = 2, 3, 4.
6. Statistics: mean +/- std, accuracy / P / R / F1 / AUC / MCC, Nadeau-Bengio
   corrected resampled t-test with Holm correction.
7. Efficiency is MEASURED (training time, inference latency, #parameters).
8. Threshold analysis: shows that the precision/recall asymmetry is an operating
   point (adjustable), not an intrinsic property, and gives a cost-sensitive example.

Usage
-----
  pip install numpy pandas scipy scikit-learn matplotlib openpyxl ucimlrepo
  python psnn_raisin_revised.py                       # downloads UCI id=850
  python psnn_raisin_revised.py --data Raisin_Dataset.xlsx
  python psnn_raisin_revised.py --synthetic --repeats 2   # quick smoke test
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve, accuracy_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Pi-Sigma Neural Network (NumPy, analytic gradients)
# --------------------------------------------------------------------------
class PiSigmaNN(BaseEstimator, ClassifierMixin):
    """
    h_j = w_j . x + b_j            (K linear summing units)
    P   = prod_j h_j               (product unit, order K)
    p   = sigmoid(a * P + c)       (output unit; a, c learnable)
    Loss: binary cross-entropy + (l2/2)||W||^2, optimiser: Adam, mini-batches.
    """

    def __init__(self, K=2, lr=0.01, epochs=400, batch_size=32, l2=1e-4,
                 patience=40, val_frac=0.15, init_std=0.1, seed=0):
        self.K = K
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.patience = patience
        self.val_frac = val_frac
        self.init_std = init_std
        self.seed = seed

    # ---- core maths ----
    def _forward(self, X, params):
        W, b, a, c = params
        H = X @ W.T + b                       # (n, K)
        P = np.prod(H, axis=1)                # (n,)
        z = a * P + c
        return H, P, expit(z)

    def _grads(self, X, y, params):
        W, b, a, c = params
        n = X.shape[0]
        H, P, p = self._forward(X, params)
        dz = (p - y) / n                      # dL/dz for mean BCE
        G = np.empty_like(H)                  # dL/dH
        for j in range(self.K):
            others = np.prod(np.delete(H, j, axis=1), axis=1)  # empty prod = 1
            G[:, j] = dz * a * others
        dW = G.T @ X + self.l2 * W
        db = G.sum(axis=0)
        da = np.sum(dz * P)
        dc = np.sum(dz)
        return [dW, db, np.array(da), np.array(dc)]

    @staticmethod
    def _bce(p, y):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    # ---- sklearn API ----
    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        rng = np.random.RandomState(self.seed)
        n, d = X.shape
        self.classes_ = np.array([0, 1])

        # inner validation split for early stopping (from the TRAINING fold only)
        idx_tr, idx_va = train_test_split(
            np.arange(n), test_size=self.val_frac, stratify=y,
            random_state=self.seed)
        Xtr, ytr, Xva, yva = X[idx_tr], y[idx_tr], X[idx_va], y[idx_va]

        W = rng.normal(0, self.init_std, (self.K, d))
        b = np.ones(self.K)                   # h_j ~ 1 at start  => P ~ 1
        a = np.array(1.0)
        c = np.array(-1.0)                    # z ~ 0 at start    => p ~ 0.5
        params = [W, b, a, c]
        m = [np.zeros_like(p_) for p_ in params]
        v = [np.zeros_like(p_) for p_ in params]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0

        best = (np.inf, [p_.copy() for p_ in params])
        bad = 0
        for _ in range(self.epochs):
            perm = rng.permutation(len(Xtr))
            for s in range(0, len(Xtr), self.batch_size):
                bi = perm[s:s + self.batch_size]
                grads = self._grads(Xtr[bi], ytr[bi], params)
                t += 1
                for k in range(4):
                    m[k] = b1 * m[k] + (1 - b1) * grads[k]
                    v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
                    mh = m[k] / (1 - b1 ** t)
                    vh = v[k] / (1 - b2 ** t)
                    params[k] = params[k] - self.lr * mh / (np.sqrt(vh) + eps)
            _, _, pva = self._forward(Xva, params)
            vl = self._bce(pva, yva)
            if vl < best[0] - 1e-5:
                best = (vl, [p_.copy() for p_ in params])
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.params_ = best[1]
        self.n_params_ = int(sum(np.size(p_) for p_ in self.params_))
        return self

    def predict_proba(self, X):
        _, _, p = self._forward(np.asarray(X, float), self.params_)
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def gradient_check(seed=0):
    """Finite-difference check of the analytic gradients (sanity test)."""
    rng = np.random.RandomState(seed)
    X, y = rng.rand(20, 5), (rng.rand(20) > 0.5).astype(float)
    for K in (1, 2, 3):
        net = PiSigmaNN(K=K, l2=0.0)
        net.K = K
        params = [rng.randn(K, 5) * 0.3, rng.randn(K) * 0.3 + 1,
                  np.array(0.8), np.array(-0.5)]
        g = net._grads(X, y, params)
        worst = 0.0
        for k in range(4):
            flat = params[k].reshape(-1)
            for i in range(flat.size):
                old = flat[i]
                flat[i] = old + 1e-6
                lp = net._bce(net._forward(X, params)[2], y)
                flat[i] = old - 1e-6
                lm = net._bce(net._forward(X, params)[2], y)
                flat[i] = old
                num = (lp - lm) / 2e-6
                worst = max(worst, abs(num - g[k].reshape(-1)[i]))
        assert worst < 1e-6, f"gradient check failed for K={K}: {worst}"
    print("[ok] gradient check passed")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_raisin(path=None, synthetic=False):
    if synthetic:
        from sklearn.datasets import make_classification
        X, y = make_classification(900, 7, n_informative=5, n_redundant=2,
                                   flip_y=0.1, class_sep=0.9, random_state=1)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(7)]), pd.Series(y)
    if path:
        df = (pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls"))
              else pd.read_csv(path))
    else:
        from ucimlrepo import fetch_ucirepo
        r = fetch_ucirepo(id=850)  # Raisin dataset (verify id on the UCI page)
        df = pd.concat([r.data.features, r.data.targets], axis=1)
    target = [c for c in df.columns if c.lower() == "class"][0]
    y = (df[target].astype(str).str.strip().str.lower() == "besni").astype(int)
    X = df.drop(columns=[target]).astype(float)
    assert len(df) == 900, "unexpected dataset size"
    return X, y


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def make_models(seed=0):
    psnn = lambda K: make_pipeline(MinMaxScaler(), PiSigmaNN(K=K, seed=seed))
    return {
        "PSNN (K=2)": psnn(2),
        "PSNN (K=3)": psnn(3),
        "PSNN (K=4)": psnn(4),
        "LogReg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000)),
        "LogReg + quadratic": make_pipeline(
            StandardScaler(), PolynomialFeatures(2, include_bias=False),
            StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
        "SVM (RBF)": make_pipeline(StandardScaler(),
                                   SVC(probability=True, random_state=seed)),
        "MLP (Adam, early stop)": make_pipeline(StandardScaler(), MLPClassifier(
            (16,), max_iter=1000, early_stopping=True, random_state=seed)),
        "MLP (L-BFGS)": make_pipeline(StandardScaler(), MLPClassifier(
            (16,), solver="lbfgs", alpha=1e-2, max_iter=2000,
            random_state=seed)),
        "Random Forest": RandomForestClassifier(300, random_state=seed),
    }


def n_params(model):
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "n_params_"):
        return est.n_params_
    if hasattr(est, "coef_"):
        return int(est.coef_.size + est.intercept_.size)
    if hasattr(est, "coefs_"):
        return int(sum(c.size for c in est.coefs_) +
                   sum(i.size for i in est.intercepts_))
    if hasattr(est, "support_vectors_"):
        return int(est.support_vectors_.size)
    if hasattr(est, "estimators_"):
        return int(sum(t.tree_.node_count for t in est.estimators_))
    return -1


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(X, y, repeats, folds, seed, models):
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats,
                                   random_state=seed)
    rows, cms = [], {k: np.zeros((2, 2)) for k in models}
    oof = {k: np.zeros(len(y)) for k in models}   # repeat-0 out-of-fold probs
    Xv, yv = X.values, y.values
    n_test = None
    for i, (tr, te) in enumerate(rskf.split(Xv, yv)):
        n_test = len(te)
        n_train = len(tr)
        for name, proto in models.items():
            mdl = clone(proto)
            t0 = time.perf_counter()
            mdl.fit(Xv[tr], yv[tr])
            fit_t = time.perf_counter() - t0
            t0 = time.perf_counter()
            proba = mdl.predict_proba(Xv[te])[:, 1]
            lat = (time.perf_counter() - t0) / len(te) * 1e3   # ms / sample
            pred = (proba >= 0.5).astype(int)
            p, r, f, _ = precision_recall_fscore_support(
                yv[te], pred, labels=[0, 1], zero_division=0)
            rows.append(dict(
                model=name, fold_run=i, repeat=i // folds,
                accuracy=accuracy_score(yv[te], pred),
                prec_kecimen=p[0], prec_besni=p[1],
                rec_kecimen=r[0], rec_besni=r[1],
                f1_kecimen=f[0], f1_besni=f[1],
                macro_f1=f.mean(), mcc=matthews_corrcoef(yv[te], pred),
                auc=roc_auc_score(yv[te], proba),
                fit_s=fit_t, ms_per_sample=lat, n_params=n_params(mdl)))
            cms[name] += confusion_matrix(yv[te], pred, labels=[0, 1])
            if i // folds == 0:
                oof[name][te] = proba
        if (i + 1) % folds == 0:
            print(f"  repeat {(i + 1) // folds}/{repeats} done")
    return pd.DataFrame(rows), cms, oof, n_train, n_test


def nadeau_bengio(d, n_train, n_test):
    """Corrected resampled t-test on per-fold differences d."""
    d = np.asarray(d)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return t, 2 * stats.t.sf(abs(t), k - 1)


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def stat_tests(df, ref, n_train, n_test, metric="accuracy"):
    piv = df.pivot(index="fold_run", columns="model", values=metric)
    out = []
    for m in piv.columns:
        if m == ref:
            continue
        d = piv[ref] - piv[m]
        t, p = nadeau_bengio(d, n_train, n_test)
        out.append(dict(reference=ref, versus=m, metric=metric,
                        mean_diff=d.mean(), t=t, p_raw=p))
    res = pd.DataFrame(out)
    res["p_holm"] = holm(res["p_raw"].values)
    return res


def threshold_analysis(y, proba, cost_besni_as_kecimen=1.0,
                       cost_kecimen_as_besni=1.0):
    rows = []
    for th in np.round(np.arange(0.05, 0.96, 0.05), 2):
        pred = (proba >= th).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y, pred, labels=[0, 1], zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append(dict(threshold=th, accuracy=(tp + tn) / len(y),
                         prec_kecimen=p[0], rec_kecimen=r[0],
                         prec_besni=p[1], rec_besni=r[1],
                         expected_cost=(fn * cost_besni_as_kecimen +
                                        fp * cost_kecimen_as_besni) / len(y)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def make_figures(df, oof, y, thr_df, best_psnn, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ROC curves (repeat-0 out-of-fold predictions)
    plt.figure(figsize=(6, 5))
    for name, pr in oof.items():
        fpr, tpr, _ = roc_curve(y, pr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, pr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.legend(fontsize=7); plt.title("ROC (out-of-fold)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_roc.png"), dpi=200)
    plt.close()

    # accuracy distribution
    order = df.groupby("model")["accuracy"].mean().sort_values().index
    plt.figure(figsize=(7, 4))
    plt.boxplot([df[df.model == m]["accuracy"] for m in order])
    plt.xticks(range(1, len(order) + 1), list(order), rotation=35, ha="right"); plt.ylabel("Test accuracy")
    plt.title("Accuracy across repeated 5-fold CV")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_accuracy_box.png"), dpi=200)
    plt.close()

    # threshold trade-off for the best PSNN
    plt.figure(figsize=(6, 4))
    for col in ["prec_kecimen", "rec_kecimen", "prec_besni", "rec_besni"]:
        plt.plot(thr_df.threshold, thr_df[col], label=col)
    plt.axvline(0.5, color="k", ls=":", lw=0.8)
    plt.xlabel("Decision threshold on P(Besni)"); plt.ylabel("Score")
    plt.title(f"{best_psnn}: operating-point trade-off"); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_threshold.png"), dpi=200)
    plt.close()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="local Raisin xlsx/csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_raisin")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gradient_check()
    X, y = load_raisin(args.data, args.synthetic)
    print(f"Data: {X.shape}, class balance: {y.value_counts().to_dict()}")

    models = make_models(args.seed)
    df, cms, oof, n_train, n_test = evaluate(
        X, y, args.repeats, args.folds, args.seed, models)
    df.to_csv(os.path.join(args.out, "per_fold_results.csv"), index=False)

    cols = ["accuracy", "prec_kecimen", "prec_besni", "rec_kecimen", "rec_besni",
            "f1_kecimen", "f1_besni", "macro_f1", "mcc", "auc",
            "fit_s", "ms_per_sample", "n_params"]
    g = df.groupby("model")[cols]
    summary = g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)
    summary["n_params"] = df.groupby("model")["n_params"].mean().round(0)
    summary.to_csv(os.path.join(args.out, "summary_mean_std.csv"))
    print("\n=== Mean ± std over", len(df) // len(models), "fold-runs ===")
    print(summary[["accuracy", "macro_f1", "auc", "mcc"]].to_string())

    # best PSNN variant is chosen by mean CV accuracy and reported as such
    psnn_names = [m for m in models if m.startswith("PSNN")]
    best = df[df.model.isin(psnn_names)].groupby("model")["accuracy"].mean().idxmax()
    tests = pd.concat([stat_tests(df, best, n_train, n_test, m)
                       for m in ("accuracy", "macro_f1", "auc")])
    tests.to_csv(os.path.join(args.out, "stat_tests.csv"), index=False)
    print(f"\n=== Corrected resampled t-tests vs {best} (accuracy) ===")
    print(tests[tests.metric == "accuracy"].round(4).to_string(index=False))

    # confusion matrices (mean per fold-run)
    n_runs = len(df) // len(models)
    cm_out = {k: (v / n_runs).round(3).tolist() for k, v in cms.items()}
    with open(os.path.join(args.out, "mean_confusion_matrices.json"), "w") as f:
        json.dump(cm_out, f, indent=2)

    # operating-point analysis (illustrative: uses repeat-0 OOF probabilities)
    thr = threshold_analysis(y.values, oof[best])
    thr.to_csv(os.path.join(args.out, "threshold_analysis.csv"), index=False)
    best_th = thr.loc[thr.expected_cost.idxmin(), "threshold"]
    print(f"\nCost-minimising threshold (equal costs) for {best}: {best_th}")

    make_figures(df, oof, y.values, thr, best, args.out)
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(vars(args) | {"best_psnn": best}, f, indent=2)
    print(f"\nAll outputs written to ./{args.out}/")


if __name__ == "__main__":
    main()
