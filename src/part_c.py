"""Part C (Tasks 11–16): logistic regression. Owner: Ziad."""

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from assignment2_core import (
    Standardizer,
    accuracy,
    add_bias,
    batch_gd,
    class_metrics,
    grad_check,
    logistic_grad,
    logistic_loss,
    roc_curve_from_scratch,
    sigmoid,
)
from part_a import save_table
from plot_utils import savefig


def verify_logistic_gradient_and_stability(dataset, out):
    """Task 11: Derivation, stable implementation, and gradient checking."""
    Xtr, _, _ = dataset["X"]
    ytr, _, _ = dataset["y"]
    rng = np.random.default_rng(42)

    # 1. Gradient check with lambda = 0.1
    point = rng.normal(0, 0.1, Xtr.shape[1])
    error = grad_check(
        lambda w: logistic_loss(w, Xtr, ytr, 0.1),
        lambda w: logistic_grad(w, Xtr, ytr, 0.1),
        point,
    )
    assert error < 1e-06, f"Logistic gradient check failed: {error}"

    # 2. Section 3.3 worked example: z = +-800 numerical stability demonstration
    z_values = [800.0, -800.0]
    stability_rows = []
    for z in z_values:
        # Naive implementation (causes overflow or 0*log(0) -> NaN / inf)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            naive_sig = 1.0 / (1.0 + np.exp(-z))
            naive_loss_y1 = -np.log(naive_sig)
            naive_loss_y0 = -np.log(1.0 - naive_sig)

        # Stable implementation
        sig_stable = float(sigmoid(np.array([z]))[0])
        loss_y1_stable = float(np.logaddexp(0, z) - 1.0 * z)
        loss_y0_stable = float(np.logaddexp(0, z) - 0.0 * z)

        stability_rows.append(
            [
                z,
                naive_sig,
                sig_stable,
                naive_loss_y1,
                loss_y1_stable,
                naive_loss_y0,
                loss_y0_stable,
            ]
        )

    save_table(
        pd.DataFrame(
            stability_rows,
            columns=[
                "z",
                "naive_sigmoid",
                "stable_sigmoid",
                "naive_loss_y1",
                "stable_loss_y1",
                "naive_loss_y0",
                "stable_loss_y0",
            ],
        ),
        "c11_numerical_stability",
        out,
    )
    save_table(
        pd.DataFrame([[error]], columns=["grad_check_relative_error"]),
        "c11_gradient_check",
        out,
    )
    return error


def train_and_tune_logistic_regression(dataset, out):
    """Task 12: Training, regularization sweep, and learning rate analysis."""
    Xtr, Xva, Xte = dataset["X"]
    ytr, yva, yte = dataset["y"]

    # Lambda sweep
    lambdas = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
    records = []
    histories = {}
    for lam in lambdas:
        w, loss_hist = batch_gd(
            lambda q: (
                logistic_loss(q, Xtr, ytr, lam),
                logistic_grad(q, Xtr, ytr, lam),
            ),
            np.zeros(Xtr.shape[1]),
            3000,
            0.5,
        )
        # Validation loss must report data term only (lam=0.0) so models are comparable
        val_loss = logistic_loss(w, Xva, yva, lam=0.0)
        tr_acc = accuracy(ytr, sigmoid(Xtr @ w) >= 0.5)
        va_acc = accuracy(yva, sigmoid(Xva @ w) >= 0.5)
        w_norm = float(np.linalg.norm(w[1:]))
        records.append([lam, tr_acc, va_acc, val_loss, w_norm, w])
        histories[lam] = (w, loss_hist)

    reg_table = pd.DataFrame(
        [r[:5] for r in records],
        columns=["lambda", "train_accuracy", "validation_accuracy", "validation_loss", "w_norm"],
    )
    save_table(reg_table, "c12_regularization", out)

    # Model selection based on unregularized validation loss
    best_record = min(records, key=lambda r: r[3])
    best_lam, _, best_va_acc, best_va_loss, _, best_w = best_record
    test_acc_best = accuracy(yte, sigmoid(Xte @ best_w) >= 0.5)

    # Sanity check: lambda = 1.0 is clearly over-regularized
    lam1_loss = [r[3] for r in records if r[0] == 1.0][0]
    assert lam1_loss > best_va_loss * 2.0, "lambda=1.0 should be clearly over-regularized"

    # Track epoch curves for the chosen lambda*
    w_star = np.zeros(Xtr.shape[1])
    tr_loss_trace, va_loss_trace = [], []
    tr_acc_trace, va_acc_trace = [], []
    for _ in range(3000):
        tr_loss_trace.append(logistic_loss(w_star, Xtr, ytr, 0.0))
        va_loss_trace.append(logistic_loss(w_star, Xva, yva, 0.0))
        tr_acc_trace.append(accuracy(ytr, sigmoid(Xtr @ w_star) >= 0.5))
        va_acc_trace.append(accuracy(yva, sigmoid(Xva @ w_star) >= 0.5))
        w_star -= 0.5 * logistic_grad(w_star, Xtr, ytr, best_lam)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(tr_loss_trace, label="Train loss (data term)")
    axes[0].plot(va_loss_trace, label="Validation loss")
    axes[0].set(title=f"Task 12: Loss curve (lambda*={best_lam})", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tr_acc_trace, label="Train accuracy")
    axes[1].plot(va_acc_trace, label="Validation accuracy")
    axes[1].set(title=f"Task 12: Accuracy curve (lambda*={best_lam})", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    savefig(out, "c12_curves")

    # Learning rate sweep at lambda = 0.01 for 1000 epochs
    rate_rows = []
    for eta in [0.01, 0.1, 1.0, 5.0, 20.0]:
        w_eta, h_eta = batch_gd(
            lambda q: (
                logistic_loss(q, Xtr, ytr, 0.01),
                logistic_grad(q, Xtr, ytr, 0.01),
            ),
            np.zeros(Xtr.shape[1]),
            1000,
            eta,
        )
        rate_rows.append([eta, h_eta[-1], accuracy(ytr, sigmoid(Xtr @ w_eta) >= 0.5)])

    save_table(
        pd.DataFrame(rate_rows, columns=["learning_rate", "final_train_loss", "train_accuracy"]),
        "c12_learning_rates",
        out,
    )
    # Sanity checks for eta
    assert rate_rows[0][1] > rate_rows[1][1], "eta=0.01 should be under-trained after 1000 epochs"
    assert rate_rows[-1][1] > 1.0, "eta=20.0 should fail to converge"

    # Also retrieve or train the lambda = 0.01 model specifically for Task 15
    w_001 = [r[5] for r in records if r[0] == 0.01][0]
    return {"best_lam": best_lam, "test_accuracy": test_acc_best, "w_best": best_w, "w_001": w_001}


def analyze_linear_separability(dataset, out):
    """Task 13: Linear separability experiment with unregularized logistic regression."""
    Xtr, _, _ = dataset["X"]
    ytr, _, _ = dataset["y"]

    w0 = np.zeros(Xtr.shape[1])
    losses = []
    norms = []
    for _ in range(20000):
        losses.append(logistic_loss(w0, Xtr, ytr, 0.0))
        norms.append(float(np.linalg.norm(w0[1:])))
        w0 -= 0.5 * logistic_grad(w0, Xtr, ytr, 0.0)

    epochs = np.arange(1, 20001)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, losses, color="tab:blue")
    axes[0].set_xscale("log")
    axes[0].set(title="Task 13: Training Loss (lambda=0)", xlabel="Epoch (log scale)", ylabel="Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, norms, color="tab:orange")
    axes[1].set_xscale("log")
    axes[1].set(title="Task 13: Weight Norm ||w1:|| (lambda=0)", xlabel="Epoch (log scale)", ylabel="||w1:||")
    axes[1].grid(True, alpha=0.3)
    savefig(out, "c13_separability")

    save_table(
        pd.DataFrame(
            [[losses[0], losses[-1], norms[0], norms[-1]]],
            columns=["initial_loss", "final_loss_epoch20000", "initial_w_norm", "final_w_norm_epoch20000"],
        ),
        "c13_separability_summary",
        out,
    )
    assert norms[-1] > norms[0] * 20.0, "Weights must grow substantially without regularization"


def benchmark_sklearn_logistic_regression(dataset, out):
    """Task 14: Off-the-shelf comparisons with scikit-learn's LogisticRegression."""
    Xtr, Xva, Xte = dataset["X"]
    ytr, yva, yte = dataset["y"]
    N_train = len(ytr)

    # 1. Verification at lambda = 0.01: match weights within 1e-4 and 100% test agreement
    w_ours, _ = batch_gd(
        lambda q: (logistic_loss(q, Xtr, ytr, 0.01), logistic_grad(q, Xtr, ytr, 0.01)),
        np.zeros(Xtr.shape[1]),
        5000,
        0.5,
    )
    C_001 = 1.0 / (0.01 * N_train)
    lr_001 = LogisticRegression(C=C_001, tol=1e-12, max_iter=100000, fit_intercept=True).fit(
        Xtr[:, 1:], ytr
    )
    w_skl = np.r_[lr_001.intercept_, lr_001.coef_.ravel()]
    rel_error = float(np.linalg.norm(w_ours - w_skl) / np.linalg.norm(w_skl))
    pred_ours = (sigmoid(Xte @ w_ours) >= 0.5).astype(int)
    pred_skl = lr_001.predict(Xte[:, 1:])
    agreement = float(np.mean(pred_ours == pred_skl))

    assert rel_error < 1e-04, f"Weight agreement error too high: {rel_error}"
    assert agreement == 1.0, f"Predictions did not agree 100%: {agreement}"

    save_table(
        pd.DataFrame(
            [[C_001, rel_error, agreement]],
            columns=["C_equivalent", "weight_relative_error", "test_prediction_agreement"],
        ),
        "c14_verification",
        out,
    )

    # 2. Sweep C in {0.001, 0.01, 0.1, 1, 10, 100, 1000}
    c_rows = []
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = LogisticRegression(C=C, tol=1e-12, max_iter=100000).fit(Xtr[:, 1:], ytr)
        c_rows.append(
            [
                C,
                accuracy(ytr, model.predict(Xtr[:, 1:])),
                accuracy(yva, model.predict(Xva[:, 1:])),
                accuracy(yte, model.predict(Xte[:, 1:])),
            ]
        )
    save_table(
        pd.DataFrame(c_rows, columns=["C", "train_accuracy", "validation_accuracy", "test_accuracy"]),
        "c14_c_sweep",
        out,
    )
    # Sanity checks
    assert c_rows[0][1] < 0.95, "C=0.001 must be clearly under-fit"
    assert c_rows[-1][1] == 1.0, "C>=100 must achieve 1.0 train accuracy"

    # 3. Solver comparison at C = 1.0: lbfgs, liblinear, saga
    solver_rows = []
    for solver in ["lbfgs", "liblinear", "saga"]:
        t0 = time.perf_counter()
        clf = LogisticRegression(C=1.0, solver=solver, tol=1e-12, max_iter=100000).fit(Xtr[:, 1:], ytr)
        fit_time = time.perf_counter() - t0
        solver_rows.append(
            [
                solver,
                int(clf.n_iter_[0]),
                fit_time,
                float(clf.intercept_[0]),
                float(np.linalg.norm(clf.coef_)),
                accuracy(yte, clf.predict(Xte[:, 1:])),
            ]
        )
    save_table(
        pd.DataFrame(
            solver_rows,
            columns=["solver", "iterations", "fit_seconds", "intercept", "coef_norm", "test_accuracy"],
        ),
        "c14_solvers",
        out,
    )


def evaluate_roc_and_tune_threshold(dataset, w_001, out):
    """Task 15: Evaluation beyond accuracy using the lambda = 0.01 model."""
    _, Xva, Xte = dataset["X"]
    _, yva, yte = dataset["y"]

    # 1. At threshold 0.5 on test set
    test_scores = sigmoid(Xte @ w_001)
    cm_05, prec_05, rec_05, f1_05 = class_metrics(yte, test_scores >= 0.5, 2)

    # 2. Validation ROC curve from scratch and AUC verification
    val_scores = sigmoid(Xva @ w_001)
    fpr, tpr, thresholds, auc_scratch = roc_curve_from_scratch(yva, val_scores)
    auc_sklearn = float(roc_auc_score(yva, val_scores))
    auc_diff = abs(auc_scratch - auc_sklearn)
    assert auc_diff < 1e-06, f"AUC difference vs scikit-learn exceeds 1e-6: {auc_diff}"

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_scratch:.4f})", color="tab:blue", lw=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random baseline")
    plt.title("Task 15: Validation ROC Curve (lambda = 0.01)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out, "c15_roc")

    # 3. Threshold selection on validation set: largest threshold where recall == 1.0
    candidates = thresholds[tpr == 1.0]
    tau = float(candidates.max()) if len(candidates) else 0.5

    # 4. Evaluate at tau on test set
    cm_tau, prec_tau, rec_tau, f1_tau = class_metrics(yte, test_scores >= tau, 2)

    comparison_rows = [
        ["0.5 (default)", 0.5, cm_05[1, 1], cm_05[0, 1], cm_05[1, 0], cm_05[0, 0], prec_05[1], rec_05[1], f1_05[1]],
        ["validated tau", tau, cm_tau[1, 1], cm_tau[0, 1], cm_tau[1, 0], cm_tau[0, 0], prec_tau[1], rec_tau[1], f1_tau[1]],
    ]
    save_table(
        pd.DataFrame(
            comparison_rows,
            columns=["threshold_type", "threshold", "TP", "FP", "FN", "TN", "precision_malignant", "recall_malignant", "f1_malignant"],
        ),
        "c15_threshold_comparison",
        out,
    )
    save_table(
        pd.DataFrame(
            [[auc_scratch, auc_sklearn, auc_diff]],
            columns=["scratch_AUC", "sklearn_AUC", "absolute_difference"],
        ),
        "c15_auc_check",
        out,
    )


def plot_raw_coordinates_decision_boundary(dataset, out):
    """Task 16: Decision boundary in original coordinates (mean radius and mean texture)."""
    raw_tr, _, _ = dataset["raw"]
    ytr, _, _ = dataset["y"]

    # Select features 0 and 1: mean radius and mean texture
    raw_tr_2 = raw_tr[:, :2]

    # Standardize 2 features with training statistics
    scaler2 = Standardizer().fit(raw_tr_2)
    std_tr_2 = add_bias(scaler2.transform(raw_tr_2))

    # Train logistic regression (lambda = 0, eta = 0.5, 5000 epochs)
    w_std, _ = batch_gd(
        lambda q: (logistic_loss(q, std_tr_2, ytr, 0.0), logistic_grad(q, std_tr_2, ytr, 0.0)),
        np.zeros(3),
        5000,
        0.5,
    )

    mu = scaler2.mean_
    sigma = scaler2.scale_

    # Convert weights to original coordinates:
    # w_j^orig = w_j / sigma_j,  w_0^orig = w_0 - sum_j w_j * mu_j / sigma_j
    w_orig = np.empty_like(w_std)
    w_orig[1:] = w_std[1:] / sigma
    w_orig[0] = w_std[0] - float(np.sum(w_std[1:] * mu / sigma))

    # Verify scores agree to 1e-9
    scores_std = std_tr_2 @ w_std
    scores_orig = add_bias(raw_tr_2) @ w_orig
    max_score_diff = float(np.max(np.abs(scores_std - scores_orig)))
    assert max_score_diff < 1e-09, f"Task 16 score difference exceeds 1e-9: {max_score_diff}"

    save_table(
        pd.DataFrame(
            [[max_score_diff, w_std[0], w_std[1], w_std[2], w_orig[0], w_orig[1], w_orig[2]]],
            columns=[
                "max_score_diff",
                "w0_standardized",
                "w1_radius_std",
                "w2_texture_std",
                "w0_original",
                "w1_radius_orig",
                "w2_texture_orig",
            ],
        ),
        "c16_boundary_weights",
        out,
    )

    # Plot decision line over raw data points:
    # w0 + w1*x1 + w2*x2 = 0  =>  x2 = -(w0 + w1*x1) / w2
    plt.figure(figsize=(7, 5))
    benign_mask = ytr == 0
    malig_mask = ytr == 1

    plt.scatter(
        raw_tr_2[benign_mask, 0],
        raw_tr_2[benign_mask, 1],
        color="tab:blue",
        label="Benign (0)",
        alpha=0.6,
        edgecolors="none",
    )
    plt.scatter(
        raw_tr_2[malig_mask, 0],
        raw_tr_2[malig_mask, 1],
        color="tab:red",
        label="Malignant (1)",
        alpha=0.6,
        edgecolors="none",
    )

    x1_grid = np.linspace(raw_tr_2[:, 0].min() - 1, raw_tr_2[:, 0].max() + 1, 200)
    x2_boundary = -(w_orig[0] + w_orig[1] * x1_grid) / w_orig[2]

    plt.plot(x1_grid, x2_boundary, "k-", lw=2, label="Decision Boundary")
    plt.xlim(raw_tr_2[:, 0].min() - 0.5, raw_tr_2[:, 0].max() + 0.5)
    plt.ylim(raw_tr_2[:, 1].min() - 0.5, raw_tr_2[:, 1].max() + 0.5)
    plt.title("Task 16: Decision Line in Raw Coordinates")
    plt.xlabel("Mean Radius (original units)")
    plt.ylabel("Mean Texture (original units)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out, "c16_boundary")


# Aliases for task-based callers
task_11 = verify_logistic_gradient_and_stability
task_12 = train_and_tune_logistic_regression
task_13 = analyze_linear_separability
task_14 = benchmark_sklearn_logistic_regression
task_15 = evaluate_roc_and_tune_threshold
task_16 = plot_raw_coordinates_decision_boundary


def library_comparisons(P, out):
    """Off-the-shelf wrapper called by assignment runner."""
    benchmark_sklearn_logistic_regression(P["breast_cancer"], out)


def part_c(d, out):
    """Run all Tasks 11–16 of Part C."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== Verifying Logistic Gradient and Stability (Task 11) ===")
    verify_logistic_gradient_and_stability(d, out)

    print("\n=== Training & Tuning Logistic Regression (Task 12) ===")
    t12_res = train_and_tune_logistic_regression(d, out)

    print("\n=== Analyzing Linear Separability (Task 13) ===")
    analyze_linear_separability(d, out)

    print("\n=== Benchmarking with Scikit-Learn LogisticRegression (Task 14) ===")
    benchmark_sklearn_logistic_regression(d, out)

    print("\n=== Evaluating ROC and Tuning Classification Threshold (Task 15) ===")
    evaluate_roc_and_tune_threshold(d, t12_res["w_001"], out)

    print("\n=== Plotting Raw Coordinates Decision Boundary (Task 16) ===")
    plot_raw_coordinates_decision_boundary(d, out)

    print(f"\nPart C passed completely. Tables and figures saved to {out}.")
