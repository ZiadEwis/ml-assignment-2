"""Part C: Logistic Regression implementation and analysis. Owner: Ziad."""

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


def check_logistic_gradient_and_stability(dataset, out):
    """Verify analytical gradient against finite differences and test numerical stability."""
    X_train, _, _ = dataset["X"]
    y_train, _, _ = dataset["y"]
    rng = np.random.default_rng(42)

    # Gradient check at lambda = 0.1
    w_initial = rng.normal(0, 0.1, X_train.shape[1])
    grad_error = grad_check(
        lambda w: logistic_loss(w, X_train, y_train, 0.1),
        lambda w: logistic_grad(w, X_train, y_train, 0.1),
        w_initial,
    )
    assert grad_error < 1e-06, f"Gradient check failed: {grad_error}"

    # Numerical stability check for large z values (+-800)
    stability_data = []
    for z in [800.0, -800.0]:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            naive_sigmoid = 1.0 / (1.0 + np.exp(-z))
            naive_loss_y1 = -np.log(naive_sigmoid)
            naive_loss_y0 = -np.log(1.0 - naive_sigmoid)

        sig_val = float(sigmoid(np.array([z]))[0])
        loss_y1 = float(np.logaddexp(0, z) - 1.0 * z)
        loss_y0 = float(np.logaddexp(0, z) - 0.0 * z)

        stability_data.append(
            [
                z,
                naive_sigmoid,
                sig_val,
                naive_loss_y1,
                loss_y1,
                naive_loss_y0,
                loss_y0,
            ]
        )

    save_table(
        pd.DataFrame(
            stability_data,
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
        pd.DataFrame([[grad_error]], columns=["grad_check_relative_error"]),
        "c11_gradient_check",
        out,
    )
    return grad_error


def train_logistic_regression(dataset, out):
    """Train logistic regression across regularization strengths and learning rates."""
    X_train, X_val, X_test = dataset["X"]
    y_train, y_val, y_test = dataset["y"]

    # Sweep lambda values
    lambdas = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
    sweep_results = []
    models_by_lambda = {}

    for lam in lambdas:
        w_fit, _ = batch_gd(
            lambda w: (
                logistic_loss(w, X_train, y_train, lam),
                logistic_grad(w, X_train, y_train, lam),
            ),
            np.zeros(X_train.shape[1]),
            3000,
            0.5,
        )
        val_loss = logistic_loss(w_fit, X_val, y_val, lam=0.0)
        train_acc = accuracy(y_train, sigmoid(X_train @ w_fit) >= 0.5)
        val_acc = accuracy(y_val, sigmoid(X_val @ w_fit) >= 0.5)
        weight_norm = float(np.linalg.norm(w_fit[1:]))

        sweep_results.append([lam, train_acc, val_acc, val_loss, weight_norm, w_fit])
        models_by_lambda[lam] = w_fit

    reg_summary = pd.DataFrame(
        [row[:5] for row in sweep_results],
        columns=["lambda", "train_accuracy", "validation_accuracy", "validation_loss", "w_norm"],
    )
    save_table(reg_summary, "c12_regularization", out)

    # Pick the model with the lowest unregularized validation loss
    best_entry = min(sweep_results, key=lambda row: row[3])
    best_lambda, _, _, best_val_loss, _, best_weights = best_entry
    test_acc_best = accuracy(y_test, sigmoid(X_test @ best_weights) >= 0.5)

    assert models_by_lambda[1.0] is not None
    lam1_val_loss = [row[3] for row in sweep_results if row[0] == 1.0][0]
    assert lam1_val_loss > best_val_loss * 2.0, "lambda=1.0 should show strong over-regularization"

    # Track loss and accuracy curves over epochs for best_lambda
    w_tracking = np.zeros(X_train.shape[1])
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for _ in range(3000):
        train_losses.append(logistic_loss(w_tracking, X_train, y_train, 0.0))
        val_losses.append(logistic_loss(w_tracking, X_val, y_val, 0.0))
        train_accuracies.append(accuracy(y_train, sigmoid(X_train @ w_tracking) >= 0.5))
        val_accuracies.append(accuracy(y_val, sigmoid(X_val @ w_tracking) >= 0.5))
        w_tracking -= 0.5 * logistic_grad(w_tracking, X_train, y_train, best_lambda)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(train_losses, label="Train Loss (data term)")
    axes[0].plot(val_losses, label="Validation Loss")
    axes[0].set(title=f"Loss vs Epoch (lambda = {best_lambda})", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accuracies, label="Train Accuracy")
    axes[1].plot(val_accuracies, label="Validation Accuracy")
    axes[1].set(title=f"Accuracy vs Epoch (lambda = {best_lambda})", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    savefig(out, "c12_curves")

    # Evaluate learning rate stability at lambda = 0.01
    rate_results = []
    for lr in [0.01, 0.1, 1.0, 5.0, 20.0]:
        w_lr, loss_history = batch_gd(
            lambda w: (
                logistic_loss(w, X_train, y_train, 0.01),
                logistic_grad(w, X_train, y_train, 0.01),
            ),
            np.zeros(X_train.shape[1]),
            1000,
            lr,
        )
        rate_results.append([lr, loss_history[-1], accuracy(y_train, sigmoid(X_train @ w_lr) >= 0.5)])

    save_table(
        pd.DataFrame(rate_results, columns=["learning_rate", "final_train_loss", "train_accuracy"]),
        "c12_learning_rates",
        out,
    )
    assert rate_results[0][1] > rate_results[1][1], "eta=0.01 should be under-trained after 1000 epochs"
    assert rate_results[-1][1] > 1.0, "eta=20.0 should diverge or fail to converge"

    w_lam_001 = models_by_lambda[0.01]
    return {
        "best_lambda": best_lambda,
        "test_accuracy": test_acc_best,
        "best_weights": best_weights,
        "weights_lambda_001": w_lam_001,
    }


def analyze_linear_separability(dataset, out):
    """Test unregularized logistic regression over 20,000 epochs to observe weight norm divergence."""
    X_train, _, _ = dataset["X"]
    y_train, _, _ = dataset["y"]

    w = np.zeros(X_train.shape[1])
    epoch_losses = []
    weight_norms = []

    for _ in range(20000):
        epoch_losses.append(logistic_loss(w, X_train, y_train, 0.0))
        weight_norms.append(float(np.linalg.norm(w[1:])))
        w -= 0.5 * logistic_grad(w, X_train, y_train, 0.0)

    epochs = np.arange(1, 20001)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, epoch_losses, color="tab:blue")
    axes[0].set_xscale("log")
    axes[0].set(title="Training Loss over 20k Epochs (lambda=0)", xlabel="Epoch (log scale)", ylabel="Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, weight_norms, color="tab:orange")
    axes[1].set_xscale("log")
    axes[1].set(title="Weight Norm ||w_1:|| over 20k Epochs (lambda=0)", xlabel="Epoch (log scale)", ylabel="||w_1:||")
    axes[1].grid(True, alpha=0.3)
    savefig(out, "c13_separability")

    save_table(
        pd.DataFrame(
            [[epoch_losses[0], epoch_losses[-1], weight_norms[0], weight_norms[-1]]],
            columns=["initial_loss", "final_loss_epoch20000", "initial_w_norm", "final_w_norm_epoch20000"],
        ),
        "c13_separability_summary",
        out,
    )
    assert weight_norms[-1] > weight_norms[0] * 20.0, "Weights should grow unbounded on separable data"


def compare_sklearn_logistic(dataset, out):
    """Benchmark scratch implementation against scikit-learn LogisticRegression."""
    X_train, X_val, X_test = dataset["X"]
    y_train, y_val, y_test = dataset["y"]
    n_train = len(y_train)

    # Verify matching weights at lambda = 0.01 (C = 1 / (lambda * N))
    w_scratch, _ = batch_gd(
        lambda w: (logistic_loss(w, X_train, y_train, 0.01), logistic_grad(w, X_train, y_train, 0.01)),
        np.zeros(X_train.shape[1]),
        5000,
        0.5,
    )
    c_equivalent = 1.0 / (0.01 * n_train)
    sklearn_model = LogisticRegression(C=c_equivalent, tol=1e-12, max_iter=100000, fit_intercept=True).fit(
        X_train[:, 1:], y_train
    )
    w_sklearn = np.r_[sklearn_model.intercept_, sklearn_model.coef_.ravel()]
    relative_diff = float(np.linalg.norm(w_scratch - w_sklearn) / np.linalg.norm(w_sklearn))

    preds_scratch = (sigmoid(X_test @ w_scratch) >= 0.5).astype(int)
    preds_sklearn = sklearn_model.predict(X_test[:, 1:])
    pred_agreement = float(np.mean(preds_scratch == preds_sklearn))

    assert relative_diff < 1e-04, f"Weight difference too large: {relative_diff}"
    assert pred_agreement == 1.0, f"Predictions do not match: {pred_agreement}"

    save_table(
        pd.DataFrame(
            [[c_equivalent, relative_diff, pred_agreement]],
            columns=["C_equivalent", "weight_relative_error", "test_prediction_agreement"],
        ),
        "c14_verification",
        out,
    )

    # Sweep C values in sklearn
    c_results = []
    for c_val in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        clf = LogisticRegression(C=c_val, tol=1e-12, max_iter=100000).fit(X_train[:, 1:], y_train)
        c_results.append(
            [
                c_val,
                accuracy(y_train, clf.predict(X_train[:, 1:])),
                accuracy(y_val, clf.predict(X_val[:, 1:])),
                accuracy(y_test, clf.predict(X_test[:, 1:])),
            ]
        )
    save_table(
        pd.DataFrame(c_results, columns=["C", "train_accuracy", "validation_accuracy", "test_accuracy"]),
        "c14_c_sweep",
        out,
    )
    assert c_results[0][1] < 0.95, "C=0.001 should be under-fit"
    assert c_results[-1][1] == 1.0, "C>=100 should fit training set fully"

    # Solver comparison at C = 1.0
    solver_data = []
    for solver in ["lbfgs", "liblinear", "saga"]:
        start_time = time.perf_counter()
        clf_solver = LogisticRegression(C=1.0, solver=solver, tol=1e-12, max_iter=100000).fit(
            X_train[:, 1:], y_train
        )
        elapsed = time.perf_counter() - start_time
        solver_data.append(
            [
                solver,
                int(clf_solver.n_iter_[0]),
                elapsed,
                float(clf_solver.intercept_[0]),
                float(np.linalg.norm(clf_solver.coef_)),
                accuracy(y_test, clf_solver.predict(X_test[:, 1:])),
            ]
        )
    save_table(
        pd.DataFrame(
            solver_data,
            columns=["solver", "iterations", "fit_seconds", "intercept", "coef_norm", "test_accuracy"],
        ),
        "c14_solvers",
        out,
    )


def plot_roc_and_select_threshold(dataset, weights_lambda_001, out):
    """Compute ROC from scratch, verify AUC, and tune classification threshold for sensitivity."""
    _, X_val, X_test = dataset["X"]
    _, y_val, y_test = dataset["y"]

    # Metrics at default threshold 0.5 on test set
    test_probs = sigmoid(X_test @ weights_lambda_001)
    cm_default, prec_default, rec_default, f1_default = class_metrics(y_test, test_probs >= 0.5, 2)

    # ROC curve on validation set
    val_probs = sigmoid(X_val @ weights_lambda_001)
    fpr, tpr, thresholds, auc_val = roc_curve_from_scratch(y_val, val_probs)
    auc_reference = float(roc_auc_score(y_val, val_probs))
    auc_diff = abs(auc_val - auc_reference)
    assert auc_diff < 1e-06, f"AUC differs from sklearn: {auc_diff}"

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.4f})", color="tab:blue", lw=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Baseline")
    plt.title("Validation ROC Curve (lambda = 0.01)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out, "c15_roc")

    # Select threshold where validation recall is 1.0
    perfect_recall_thresholds = thresholds[tpr == 1.0]
    tau = float(perfect_recall_thresholds.max()) if len(perfect_recall_thresholds) else 0.5

    # Test metrics at tuned threshold tau
    cm_tau, prec_tau, rec_tau, f1_tau = class_metrics(y_test, test_probs >= tau, 2)

    comparison = [
        [
            "0.5 (default)",
            0.5,
            cm_default[1, 1],
            cm_default[0, 1],
            cm_default[1, 0],
            cm_default[0, 0],
            prec_default[1],
            rec_default[1],
            f1_default[1],
        ],
        [
            "validated tau",
            tau,
            cm_tau[1, 1],
            cm_tau[0, 1],
            cm_tau[1, 0],
            cm_tau[0, 0],
            prec_tau[1],
            rec_tau[1],
            f1_tau[1],
        ],
    ]
    save_table(
        pd.DataFrame(
            comparison,
            columns=[
                "threshold_type",
                "threshold",
                "TP",
                "FP",
                "FN",
                "TN",
                "precision_malignant",
                "recall_malignant",
                "f1_malignant",
            ],
        ),
        "c15_threshold_comparison",
        out,
    )
    save_table(
        pd.DataFrame([[auc_val, auc_reference, auc_diff]], columns=["scratch_AUC", "sklearn_AUC", "diff"]),
        "c15_auc_check",
        out,
    )


def plot_decision_boundary(dataset, out):
    """Train logistic regression on 2 features and draw decision boundary in original coordinates."""
    raw_train, _, _ = dataset["raw"]
    y_train, _, _ = dataset["y"]

    # Select mean radius (0) and mean texture (1)
    raw_2features = raw_train[:, :2]
    scaler = Standardizer().fit(raw_2features)
    std_2features = add_bias(scaler.transform(raw_2features))

    # Train model on standardized features
    w_std, _ = batch_gd(
        lambda w: (logistic_loss(w, std_2features, y_train, 0.0), logistic_grad(w, std_2features, y_train, 0.0)),
        np.zeros(3),
        5000,
        0.5,
    )

    # Convert weights to original coordinates
    mean = scaler.mean_
    scale = scaler.scale_
    w_original = np.empty_like(w_std)
    w_original[1:] = w_std[1:] / scale
    w_original[0] = w_std[0] - float(np.sum(w_std[1:] * mean / scale))

    # Validate score agreement
    scores_standardized = std_2features @ w_std
    scores_original = add_bias(raw_2features) @ w_original
    max_diff = float(np.max(np.abs(scores_standardized - scores_original)))
    assert max_diff < 1e-09, f"Score conversion difference too large: {max_diff}"

    save_table(
        pd.DataFrame(
            [[max_diff, w_std[0], w_std[1], w_std[2], w_original[0], w_original[1], w_original[2]]],
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

    # Plot decision boundary over raw data points
    plt.figure(figsize=(7, 5))
    benign_mask = y_train == 0
    malignant_mask = y_train == 1

    plt.scatter(
        raw_2features[benign_mask, 0],
        raw_2features[benign_mask, 1],
        color="tab:blue",
        label="Benign (0)",
        alpha=0.6,
    )
    plt.scatter(
        raw_2features[malignant_mask, 0],
        raw_2features[malignant_mask, 1],
        color="tab:red",
        label="Malignant (1)",
        alpha=0.6,
    )

    x1_range = np.linspace(raw_2features[:, 0].min() - 1, raw_2features[:, 0].max() + 1, 200)
    x2_boundary = -(w_original[0] + w_original[1] * x1_range) / w_original[2]

    plt.plot(x1_range, x2_boundary, "k-", lw=2, label="Decision Boundary")
    plt.xlim(raw_2features[:, 0].min() - 0.5, raw_2features[:, 0].max() + 0.5)
    plt.ylim(raw_2features[:, 1].min() - 0.5, raw_2features[:, 1].max() + 0.5)
    plt.title("Decision Boundary in Original Feature Units")
    plt.xlabel("Mean Radius (raw units)")
    plt.ylabel("Mean Texture (raw units)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out, "c16_boundary")


def library_comparisons(prepared_data, out):
    """Off-the-shelf comparison entry point for runner."""
    compare_sklearn_logistic(prepared_data["breast_cancer"], out)


def part_c(dataset, out):
    """Execute all logistic regression experiments for Part C."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    print("\nRunning gradient check and numerical stability...")
    check_logistic_gradient_and_stability(dataset, out)

    print("\nTraining logistic regression and tuning regularization...")
    logistic_model = train_logistic_regression(dataset, out)

    print("\nTesting linear separability with unregularized model...")
    analyze_linear_separability(dataset, out)

    print("\nComparing with scikit-learn LogisticRegression...")
    compare_sklearn_logistic(dataset, out)

    print("\nEvaluating ROC curve and threshold tuning...")
    plot_roc_and_select_threshold(dataset, logistic_model["weights_lambda_001"], out)

    print("\nPlotting decision boundary on raw coordinates...")
    plot_decision_boundary(dataset, out)

    print(f"\nPart C completed successfully. Figures and tables saved in {out}.")
