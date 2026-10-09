"""
Group 30 Cyber Security Case Study
AI-Driven Anomaly Detection for MOVEit Transfer Attack (2023)

Educational proof-of-concept using synthetic sessions. This is NOT a production
intrusion detection system and does not process real MOVEit victim logs.
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve
)

SEED = 42
FEATURES = [
    "request_count", "distinct_endpoints", "error_ratio", "post_ratio",
    "mean_request_size", "payload_special_char_ratio", "path_entropy",
    "inter_request_mean", "inter_request_std", "rare_endpoint_hits",
    "mean_response_size", "download_burst_size", "upload_count",
    "custom_header_flag", "off_hours_flag"
]

def generate_sessions(n, attack=False, stealth_fraction=0.30, seed=SEED):
    """Create transparent synthetic session features inspired by the report."""
    rng = np.random.default_rng(seed)
    # Benign traffic: varied, skewed counts/sizes and bounded ratios.
    x = np.zeros((n, len(FEATURES)), dtype=float)
    x[:, 0] = rng.poisson(18, n) + 1
    x[:, 1] = np.clip(rng.poisson(5, n) + 1, 1, 20)
    x[:, 2] = rng.beta(1.4, 18, n)
    x[:, 3] = rng.beta(2.0, 5.0, n)
    x[:, 4] = rng.lognormal(6.2, 0.7, n)
    x[:, 5] = rng.beta(1.2, 25, n)
    x[:, 6] = rng.uniform(0.1, 3.0, n)
    x[:, 7] = rng.lognormal(1.5, 0.65, n)
    x[:, 8] = rng.lognormal(1.0, 0.6, n)
    x[:, 9] = rng.poisson(0.25, n)
    x[:, 10] = rng.lognormal(7.0, 0.8, n)
    x[:, 11] = rng.lognormal(8.0, 0.8, n)
    x[:, 12] = rng.poisson(0.4, n)
    x[:, 13] = rng.binomial(1, 0.025, n)
    x[:, 14] = rng.binomial(1, 0.18, n)

    # Difficult but benign cases: backup/sync clients and noisy scanners.
    backup = rng.random(n) < 0.04
    x[backup, 0] *= 4
    x[backup, 10] *= 2
    x[backup, 11] *= 2.5
    x[backup, 12] += rng.poisson(2, backup.sum())
    scanner = rng.random(n) < 0.02
    x[scanner, 2] = np.clip(x[scanner, 2] + rng.uniform(0.15, 0.45, scanner.sum()), 0, 1)
    x[scanner, 9] += rng.poisson(3, scanner.sum())

    if attack:
        x[:, 0] += rng.poisson(25, n)
        x[:, 1] = np.clip(x[:, 1] + rng.poisson(5, n), 1, 30)
        x[:, 2] = np.clip(x[:, 2] + rng.uniform(0.10, 0.45, n), 0, 1)
        x[:, 3] = np.clip(x[:, 3] + rng.uniform(0.15, 0.55, n), 0, 1)
        x[:, 4] *= rng.uniform(1.1, 2.0, n)
        x[:, 5] = np.clip(x[:, 5] + rng.uniform(0.12, 0.5, n), 0, 1)
        x[:, 6] += rng.uniform(0.3, 2.5, n)
        x[:, 7] *= rng.uniform(0.5, 1.4, n)
        x[:, 8] *= rng.uniform(0.7, 1.7, n)
        x[:, 9] += rng.poisson(4, n)
        x[:, 10] *= rng.uniform(2.0, 6.0, n)
        x[:, 11] *= rng.uniform(2.0, 8.0, n)
        x[:, 12] += rng.poisson(2.5, n)
        x[:, 13] = rng.binomial(1, 0.72, n)
        x[:, 14] = rng.binomial(1, 0.52, n)
        stealth = rng.random(n) < stealth_fraction
        # Low-and-slow cases deliberately resemble benign traffic.
        for j, lo, hi in [
            (0, 0.35, 0.65), (2, 0.15, 0.35), (3, 0.2, 0.4),
            (5, 0.15, 0.35), (9, 0.2, 0.5), (10, 0.3, 0.6), (11, 0.3, 0.6)
        ]:
            x[stealth, j] = x[stealth, j] * rng.uniform(lo, hi, stealth.sum())
        x[stealth, 13] = rng.binomial(1, 0.35, stealth.sum())
    return pd.DataFrame(x, columns=FEATURES)

def transform_features(df, scaler=None, fit=False):
    values = df[FEATURES].to_numpy(dtype=float).copy()
    # Counts/sizes are right-skewed; log1p transform selected columns.
    log_cols = [0, 1, 4, 7, 8, 9, 10, 11, 12]
    values[:, log_cols] = np.log1p(np.maximum(values[:, log_cols], 0))
    if fit:
        scaler = StandardScaler()
        values = scaler.fit_transform(values)
        return values, scaler
    return scaler.transform(values)

def percentile_scores(scores, reference):
    """Convert scores to empirical percentiles using benign validation scores."""
    ref = np.sort(np.asarray(reference))
    return np.searchsorted(ref, scores, side="right") / max(len(ref), 1)

def metrics_for(y_true, scores, preds):
    return {
        "ROC_AUC": float(roc_auc_score(y_true, scores)),
        "PR_AUC": float(average_precision_score(y_true, scores)),
        "Precision": float(precision_score(y_true, preds, zero_division=0)),
        "Recall": float(recall_score(y_true, preds, zero_division=0)),
        "F1": float(f1_score(y_true, preds, zero_division=0)),
        "FPR": float(np.sum((preds == 1) & (y_true == 0)) / max(np.sum(y_true == 0), 1))
    }

def run(output_dir="outputs", quick=False):
    warnings.filterwarnings("ignore")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_train, n_val, n_benign_test, n_attack = (4000, 1000, 2000, 150) if quick else (20000, 4000, 8000, 400)

    print("Generating reproducible synthetic sessions...")
    train_df = generate_sessions(n_train, attack=False, seed=42)
    val_df = generate_sessions(n_val, attack=False, seed=43)
    benign_test = generate_sessions(n_benign_test, attack=False, seed=44)
    attack_test = generate_sessions(n_attack, attack=True, seed=45)
    test_df = pd.concat([benign_test, attack_test], ignore_index=True)
    y_test = np.r_[np.zeros(n_benign_test, dtype=int), np.ones(n_attack, dtype=int)]

    X_train, scaler = transform_features(train_df, fit=True)
    X_val = transform_features(val_df, scaler=scaler)
    X_test = transform_features(test_df, scaler=scaler)

    print("Training Isolation Forest...")
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=1, n_jobs=-1)
    iso.fit(X_train)
    s_if_val, s_if_test = -iso.score_samples(X_val), -iso.score_samples(X_test)

    print("Training autoencoder (MLP reconstruction model)...")
    ae = MLPRegressor(hidden_layer_sizes=(12, 6, 12), activation="relu",
                      solver="adam", max_iter=160 if quick else 250,
                      early_stopping=True, random_state=1, learning_rate_init=0.001)
    ae.fit(X_train, X_train)
    s_ae_val = np.mean((ae.predict(X_val) - X_val) ** 2, axis=1)
    s_ae_test = np.mean((ae.predict(X_test) - X_test) ** 2, axis=1)

    print("Scoring Local Outlier Factor...")
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True, contamination="auto", n_jobs=-1)
    lof.fit(X_train)
    s_lof_val, s_lof_test = -lof.score_samples(X_val), -lof.score_samples(X_test)

    print("Training One-Class SVM baseline (using a reproducible training subset)...")
    svm_n = min(6000, len(X_train))
    svm = OneClassSVM(kernel="rbf", nu=0.02, gamma="scale")
    svm.fit(X_train[:svm_n])
    s_svm_val, s_svm_test = -svm.score_samples(X_val), -svm.score_samples(X_test)

    raw = {
        "Isolation Forest": (s_if_val, s_if_test),
        "Autoencoder": (s_ae_val, s_ae_test),
        "LOF": (s_lof_val, s_lof_test),
        "One-Class SVM": (s_svm_val, s_svm_test),
    }
    percentile_test = {}
    percentile_val = {}
    for name, (sv, st) in raw.items():
        percentile_val[name] = percentile_scores(sv, sv)
        percentile_test[name] = percentile_scores(st, sv)

    # Threshold each individual baseline using benign validation 99th percentile.
    results = []
    score_by_model = {}
    for name, (sv, st) in raw.items():
        threshold = float(np.quantile(sv, 0.99))
        pred = (st > threshold).astype(int)
        results.append({"Model": name, **metrics_for(y_test, st, pred)})
        score_by_model[name] = st

    hybrid_val = np.mean(np.column_stack([
        percentile_val["Isolation Forest"], percentile_val["Autoencoder"],
        percentile_val["LOF"]]), axis=1)
    hybrid_test = np.mean(np.column_stack([
        percentile_test["Isolation Forest"], percentile_test["Autoencoder"],
        percentile_test["LOF"]]), axis=1)
    threshold = float(np.quantile(hybrid_val, 0.99))
    hybrid_pred = (hybrid_test > threshold).astype(int)
    results.append({"Model": "Hybrid (IF + AE + LOF)", **metrics_for(y_test, hybrid_test, hybrid_pred)})
    score_by_model["Hybrid (IF + AE + LOF)"] = hybrid_test

    table = pd.DataFrame(results)
    table.to_csv(out / "model_comparison.csv", index=False)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset": {"benign_train": n_train, "benign_validation": n_val,
                        "benign_test": n_benign_test, "attack_test": n_attack,
                        "features": len(FEATURES), "synthetic": True},
            "hybrid_threshold": threshold,
            "results": results
        }, f, indent=2)

    # Save scored test sessions for demonstration and further inspection.
    scored = test_df.copy()
    scored["true_label"] = y_test
    scored["hybrid_score"] = hybrid_test
    scored["predicted_anomaly"] = hybrid_pred
    scored.to_csv(out / "scored_test_sessions.csv", index=False)

    # ROC curves.
    plt.figure(figsize=(8, 6))
    for name, scores in score_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Synthetic MOVEit-like Sessions")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "roc_curves.png", dpi=160)
    plt.close()

    # Feature shift chart, using raw generated features.
    shift = (attack_test[FEATURES].mean() - benign_test[FEATURES].mean())
    shift = shift.sort_values()
    plt.figure(figsize=(9, 6))
    shift.plot(kind="barh")
    plt.xlabel("Mean feature difference (attack − benign)")
    plt.title("Synthetic Feature Shifts")
    plt.tight_layout()
    plt.savefig(out / "feature_shift.png", dpi=160)
    plt.close()

    # Score distribution.
    plt.figure(figsize=(8, 5))
    plt.hist(hybrid_test[y_test == 0], bins=45, alpha=0.7, label="Benign sessions")
    plt.hist(hybrid_test[y_test == 1], bins=30, alpha=0.7, label="Attack sessions")
    plt.axvline(threshold, linestyle="--", label=f"Threshold={threshold:.3f}")
    plt.xlabel("Hybrid anomaly score")
    plt.ylabel("Session count")
    plt.title("Hybrid Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "hybrid_score_distribution.png", dpi=160)
    plt.close()

    print("\n=== MODEL COMPARISON (actual run) ===")
    print(table.round(4).to_string(index=False))
    print(f"\nHybrid alert threshold: {threshold:.4f}")
    print(f"Outputs saved in: {out.resolve()}")
    print("Note: these are synthetic proof-of-concept results, not real-world detection claims.")
    return table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic MOVEit anomaly detection proof of concept")
    parser.add_argument("--output", default="outputs", help="Directory for result files")
    parser.add_argument("--quick", action="store_true", help="Run a smaller, faster demonstration")
    args = parser.parse_args()
    run(args.output, quick=args.quick)
