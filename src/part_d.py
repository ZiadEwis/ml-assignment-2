"""Part D (Tasks 17–22): softmax classification. Owner: Ziad."""

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from assignment2_core import (
    Standardizer,
    accuracy,
    add_bias,
    class_metrics,
    grad_check,
    minibatch_gd,
    sigmoid,
    softmax,
    softmax_grad,
    softmax_loss,
)
from part_a import save_table
from plot_utils import savefig


def verify_softmax_gradient_and_stability(dataset, out):
    """Task 17: Stable softmax, cross-entropy via log-sum-exp, and gradient check."""
    Xtr, _, _ = dataset["X"]
    ytr, _, _ = dataset["y"]
    K = 10

    # 1. Verification at W = 0: loss equals ln(10) ~ 2.302585
    W_zero = np.zeros((Xtr.shape[1], K))
    loss_at_zero = softmax_loss(W_zero, Xtr, ytr, lam=0.0)
    expected_zero = float(np.log(K))
    diff_zero = abs(loss_at_zero - expected_zero)
    assert diff_zero < 1e-12, f"Loss at W=0 ({loss_at_zero}) does not match ln(10) ({expected_zero})"

    # 2. Section 3.3 worked example: [1000, 1001, 1002] numerical stability
    s_test = np.array([[1000.0, 1001.0, 1002.0]])
    p_stable = softmax(s_test)[0]
    expected_p = np.array([0.09003057, 0.24472847, 0.66524096])
    assert np.allclose(p_stable, expected_p, atol=1e-4), f"Softmax stability check failed: {p_stable}"

    # 3. Gradient check on 100 randomly chosen entries of W at lambda = 0.1
    rng = np.random.default_rng(42)
    W0 = rng.normal(0, 0.1, (Xtr.shape[1], K))
    entries = rng.choice(W0.size, size=100, replace=False)
    grad_err = grad_check(
        lambda q: softmax_loss(q.reshape(W0.shape), Xtr, ytr, 0.1),
        lambda q: softmax_grad(q.reshape(W0.shape), Xtr, ytr, 0.1).ravel(),
        W0.ravel(),
        entries=entries,
    )
    assert grad_err < 1e-06, f"Softmax gradient check failed: {grad_err}"

    save_table(
        pd.DataFrame(
            [[loss_at_zero, expected_zero, diff_zero, grad_err]],
            columns=["W_zero_loss", "ln_10_expected", "zero_loss_diff", "grad_check_relative_error"],
        ),
        "d17_checks",
        out,
    )
    save_table(
        pd.DataFrame(
            [list(expected_p), list(p_stable)],
            index=["expected", "computed_stable"],
            columns=["s1=1000", "s2=1001", "s3=1002"],
        ).reset_index(),
        "d17_numerical_stability",
        out,
    )


def train_and_tune_softmax_classifier(dataset, out):
    """Task 18: Mini-batch gradient descent (B=64, 100 epochs), learning rate & lambda sweeps."""
    Xtr, Xva, Xte = dataset["X"]
    ytr, yva, yte = dataset["y"]
    N = len(ytr)
    K = 10
    W0 = np.zeros((Xtr.shape[1], K))

    # (a) With lambda = 0, run eta in {0.01, 0.1, 0.5, 1.0}
    eta_candidates = [0.01, 0.1, 0.5, 1.0]
    eta_results = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for eta in eta_candidates:
        rng = np.random.default_rng(42)
        tr_loss_hist, va_loss_hist = [], []
        tr_acc_hist, va_acc_hist = [], []
        W_curr = W0.copy()

        for epoch in range(100):
            order = rng.permutation(N)
            for start in range(0, N, 64):
                ix = order[start : start + 64]
                grad = softmax_grad(W_curr, Xtr[ix], ytr[ix], lam=0.0)
                W_curr -= eta * grad

            tr_loss_hist.append(softmax_loss(W_curr, Xtr, ytr, 0.0))
            va_loss_hist.append(softmax_loss(W_curr, Xva, yva, 0.0))
            tr_acc_hist.append(accuracy(ytr, (Xtr @ W_curr).argmax(1)))
            va_acc_hist.append(accuracy(yva, (Xva @ W_curr).argmax(1)))

        axes[0].plot(va_loss_hist, label=f"eta={eta}")
        axes[1].plot(va_acc_hist, label=f"eta={eta}")

        eta_results.append(
            [
                eta,
                tr_loss_hist[-1],
                va_loss_hist[-1],
                tr_acc_hist[-1],
                va_acc_hist[-1],
                W_curr,
            ]
        )

    axes[0].set(title="Task 18(a): Validation Loss vs Epoch", xlabel="Epoch", ylabel="Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set(title="Task 18(a): Validation Accuracy vs Epoch", xlabel="Epoch", ylabel="Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    savefig(out, "d18_learning_rates_curves")

    save_table(
        pd.DataFrame(
            [r[:5] for r in eta_results],
            columns=["learning_rate", "final_train_loss", "final_validation_loss", "train_accuracy", "validation_accuracy"],
        ),
        "d18_learning_rates",
        out,
    )

    # Choose eta* by lowest final validation loss
    best_eta_row = min(eta_results, key=lambda r: r[2])
    best_eta = best_eta_row[0]

    # Sanity check: eta = 0.01 is clearly under-trained
    assert eta_results[0][1] > eta_results[1][1], "eta=0.01 should be under-trained after 100 epochs"

    # (b) With eta*, sweep lambda in {0, 1e-4, 1e-3, 1e-2, 1e-1}
    lambdas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    lam_results = []
    for lam in lambdas:
        rng = np.random.default_rng(42)
        f = lambda W, ix: (
            softmax_loss(W, Xtr if ix is None else Xtr[ix], ytr if ix is None else ytr[ix], lam),
            softmax_grad(W, Xtr if ix is None else Xtr[ix], ytr if ix is None else ytr[ix], lam),
        )
        W_lam, _, _ = minibatch_gd(f, W0, N, 64, 100, best_eta, rng)
        tr_acc = accuracy(ytr, (Xtr @ W_lam).argmax(1))
        va_acc = accuracy(yva, (Xva @ W_lam).argmax(1))
        # Validation loss must be data term only for fair comparisons
        va_loss = softmax_loss(W_lam, Xva, yva, lam=0.0)
        lam_results.append([lam, tr_acc, va_acc, va_loss, W_lam])

    save_table(
        pd.DataFrame(
            [r[:4] for r in lam_results],
            columns=["lambda", "train_accuracy", "validation_accuracy", "validation_loss"],
        ),
        "d18_regularization",
        out,
    )

    # Choose lambda* by validation accuracy (ties -> larger lambda)
    best_lam_row = max(lam_results, key=lambda r: (r[2], r[0]))
    best_lam = best_lam_row[0]
    best_W = best_lam_row[4]
    test_acc = accuracy(yte, (Xte @ best_W).argmax(1))

    # Sanity checks
    assert 0.95 <= best_lam_row[2] <= 0.98, f"Best validation accuracy {best_lam_row[2]} should be between 0.95 and 0.98"
    assert lam_results[-1][2] < 0.80, "lambda=0.1 should be clearly too strong"

    save_table(
        pd.DataFrame(
            [[best_eta, best_lam, best_lam_row[2], test_acc]],
            columns=["selected_eta", "selected_lambda", "best_validation_accuracy", "test_accuracy"],
        ),
        "d18_selected_model",
        out,
    )

    return {"best_eta": best_eta, "best_lam": best_lam, "best_W": best_W, "test_acc": test_acc}


def benchmark_sklearn_softmax_classifier(dataset, best_lam, best_W, out):
    """Task 19: Off-the-shelf comparison with scikit-learn's LogisticRegression."""
    Xtr, Xva, Xte = dataset["X"]
    ytr, yva, yte = dataset["y"]
    N_train = len(ytr)

    # 1. Sweep C in {0.1, 1, 10, 100}
    c_rows = []
    for C in [0.1, 1.0, 10.0, 100.0]:
        model = LogisticRegression(C=C, max_iter=10000).fit(Xtr[:, 1:], ytr)
        va_acc = accuracy(yva, model.predict(Xva[:, 1:]))
        te_acc = accuracy(yte, model.predict(Xte[:, 1:]))
        c_rows.append([C, va_acc, te_acc])

    save_table(
        pd.DataFrame(c_rows, columns=["C", "validation_accuracy", "test_accuracy"]),
        "d19_c_sweep",
        out,
    )

    # 2. Match at lambda*: test prediction agreement >= 97%
    C_star = 1.0 / (best_lam * N_train) if best_lam > 0 else 1000.0
    lr_star = LogisticRegression(C=C_star, max_iter=10000).fit(Xtr[:, 1:], ytr)
    pred_ours = (Xte @ best_W).argmax(1)
    pred_skl = lr_star.predict(Xte[:, 1:])
    agreement = float(np.mean(pred_ours == pred_skl))

    assert agreement >= 0.97, f"Agreement at lambda* ({agreement:.4f}) is below 97%"

    save_table(
        pd.DataFrame(
            [[best_lam, C_star, accuracy(yte, pred_ours), accuracy(yte, pred_skl), agreement]],
            columns=["selected_lambda", "matching_C", "our_test_acc", "sklearn_test_acc", "prediction_agreement"],
        ),
        "d19_verification",
        out,
    )


def analyze_digit_classification_errors_and_weights(dataset, best_W, out):
    """Task 20: Error analysis, confusion matrix, and 8x8 weight template visualization."""
    _, _, Xte = dataset["X"]
    _, _, yte = dataset["y"]
    pred = (Xte @ best_W).argmax(1)

    # 1. Confusion matrix and class metrics
    cm, precision, recall, f1 = class_metrics(yte, pred, 10)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.title("Task 20: Digits Test Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    savefig(out, "d20_confusion")

    metrics_df = pd.DataFrame(
        {
            "class": np.arange(10),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    save_table(metrics_df, "d20_class_metrics", out)

    # 2. Identify top 3 most confused pairs
    cm_offdiag = cm.copy()
    np.fill_diagonal(cm_offdiag, 0)
    confused_pairs = []
    for i in range(10):
        for j in range(10):
            if cm_offdiag[i, j] > 0:
                confused_pairs.append([int(i), int(j), int(cm_offdiag[i, j])])
    confused_pairs.sort(key=lambda row: row[2], reverse=True)
    save_table(
        pd.DataFrame(confused_pairs[:3], columns=["true_class", "predicted_class", "error_count"]),
        "d20_top_confused_pairs",
        out,
    )

    # 3. Show 6 misclassified test digits as 8x8 images
    wrong_indices = np.flatnonzero(pred != yte)[:6]
    fig, axs = plt.subplots(2, 3, figsize=(8, 5.5))
    for ax, idx in zip(axs.ravel(), wrong_indices):
        # Pixel values in Digits were divided by 16, Xte has bias at index 0
        img = (Xte[idx, 1:] * 16.0).reshape(8, 8)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"True: {yte[idx]} / Pred: {pred[idx]}")
        ax.axis("off")
    savefig(out, "d20_misclassified")

    # 4. Reshape each class's 64 weights into 8x8 image and show the ten side by side
    fig, axs = plt.subplots(2, 5, figsize=(11, 4.5))
    for k, ax in enumerate(axs.ravel()):
        weight_img = best_W[1:, k].reshape(8, 8)
        vmax = np.max(np.abs(weight_img))
        ax.imshow(weight_img, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Digit {k}")
        ax.axis("off")
    savefig(out, "d20_weights")


def verify_binary_equivalence_and_initializations(digits_dataset, bc_dataset, out):
    """Task 21: Proof and numerical check that K=2 softmax equals binary logistic regression, plus initialization study."""
    # 1. Numerical verification of 2-class equivalence using Breast Cancer test features
    _, _, Xte_bc = bc_dataset["X"]
    rng = np.random.default_rng(42)
    W_two_class = rng.normal(0, 1.0, (Xte_bc.shape[1], 2))

    # Softmax probability of class 1:
    p_softmax = softmax(Xte_bc @ W_two_class)[:, 1]

    # Binary logistic regression probability using parameter difference vector w = w1 - w0:
    w_diff = W_two_class[:, 1] - W_two_class[:, 0]
    p_logistic = sigmoid(Xte_bc @ w_diff)

    max_prob_diff = float(np.max(np.abs(p_softmax - p_logistic)))
    assert max_prob_diff < 1e-12, f"Equivalence check failed: difference {max_prob_diff} >= 1e-12"

    save_table(
        pd.DataFrame([[max_prob_diff]], columns=["max_absolute_probability_difference"]),
        "d21_equivalence_check",
        out,
    )

    # 2. Initialization analysis: Zeros vs Random normal
    Xtr, Xva, Xte = digits_dataset["X"]
    ytr, yva, yte = digits_dataset["y"]
    N = len(ytr)
    K = 10

    init_rows = []
    for name, init_w in [
        ("zeros", np.zeros((Xtr.shape[1], K))),
        ("random_normal_0.01", rng.normal(0, 0.01, (Xtr.shape[1], K))),
        ("random_normal_0.10", rng.normal(0, 0.10, (Xtr.shape[1], K))),
    ]:
        f = lambda W, ix: (
            softmax_loss(W, Xtr if ix is None else Xtr[ix], ytr if ix is None else ytr[ix], 0.001),
            softmax_grad(W, Xtr if ix is None else Xtr[ix], ytr if ix is None else ytr[ix], 0.001),
        )
        W_fit, h, _ = minibatch_gd(f, init_w, N, 64, 100, 1.0, np.random.default_rng(42))
        init_rows.append(
            [
                name,
                float(h[-1]),
                accuracy(yva, (Xva @ W_fit).argmax(1)),
                accuracy(yte, (Xte @ W_fit).argmax(1)),
            ]
        )

    save_table(
        pd.DataFrame(init_rows, columns=["initialization", "final_train_loss", "validation_accuracy", "test_accuracy"]),
        "d21_initialization",
        out,
    )


def visualize_iris_decision_boundaries(P, out, rng=None):
    """Task 22: Softmax decision regions on Iris using standardized petal length and petal width."""
    iris = P["iris"]
    raw_tr, _, raw_te = iris["raw"]
    ytr, _, yte = iris["y"]

    # Select features 2 (petal length) and 3 (petal width)
    raw_tr_2 = raw_tr[:, [2, 3]]
    raw_te_2 = raw_te[:, [2, 3]]

    scaler = Standardizer().fit(raw_tr_2)
    Xtr_2 = add_bias(scaler.transform(raw_tr_2))
    Xte_2 = add_bias(scaler.transform(raw_te_2))
    N_train = len(ytr)
    K = 3

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    comparison_records = []

    # Meshgrid for contourf
    x_min, x_max = Xtr_2[:, 1].min() - 0.8, Xtr_2[:, 1].max() + 0.8
    y_min, y_max = Xtr_2[:, 2].min() - 0.8, Xtr_2[:, 2].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid_X = add_bias(np.column_stack([xx.ravel(), yy.ravel()]))

    for row_idx, lam in enumerate([0.0, 1.0]):
        # Train scratch softmax (full batch, eta=0.5, 2000 epochs)
        W0 = np.zeros((3, K))
        f = lambda W, ix: (
            softmax_loss(W, Xtr_2 if ix is None else Xtr_2[ix], ytr if ix is None else ytr[ix], lam),
            softmax_grad(W, Xtr_2 if ix is None else Xtr_2[ix], ytr if ix is None else ytr[ix], lam),
        )
        W_fit, _, _ = minibatch_gd(f, W0, N_train, N_train, 2000, 0.5, np.random.default_rng(42))
        scratch_pred_grid = (grid_X @ W_fit).argmax(1).reshape(xx.shape)
        scratch_test_acc = accuracy(yte, (Xte_2 @ W_fit).argmax(1))

        # Train scikit-learn LogisticRegression at matching C
        C_val = 1e5 if lam == 0 else 1.0 / (lam * N_train)
        lr = LogisticRegression(C=C_val, tol=1e-12, max_iter=10000).fit(Xtr_2[:, 1:], ytr)
        skl_pred_grid = lr.predict(grid_X[:, 1:]).reshape(xx.shape)
        skl_test_acc = accuracy(yte, lr.predict(Xte_2[:, 1:]))

        comparison_records.append([lam, C_val, scratch_test_acc, skl_test_acc])

        # Plot scratch regions
        ax_scratch = axes[row_idx, 0]
        ax_scratch.contourf(xx, yy, scratch_pred_grid, alpha=0.3, cmap="viridis")
        scatter = ax_scratch.scatter(Xtr_2[:, 1], Xtr_2[:, 2], c=ytr, cmap="viridis", edgecolors="k")
        ax_scratch.set(
            title=f"NumPy Softmax (lambda={lam}) - Test Acc: {scratch_test_acc:.3f}",
            xlabel="Petal Length (standardized)",
            ylabel="Petal Width (standardized)",
        )

        # Plot sklearn regions
        ax_skl = axes[row_idx, 1]
        ax_skl.contourf(xx, yy, skl_pred_grid, alpha=0.3, cmap="viridis")
        ax_skl.scatter(Xtr_2[:, 1], Xtr_2[:, 2], c=ytr, cmap="viridis", edgecolors="k")
        ax_skl.set(
            title=f"scikit-learn (C={C_val:.4g}) - Test Acc: {skl_test_acc:.3f}",
            xlabel="Petal Length (standardized)",
            ylabel="Petal Width (standardized)",
        )

    savefig(out, "d22_iris_decision_regions")
    save_table(
        pd.DataFrame(
            comparison_records,
            columns=["lambda", "matching_C", "scratch_test_accuracy", "sklearn_test_accuracy"],
        ),
        "d22_iris_comparisons",
        out,
    )


# Aliases for task-based callers
task_17 = verify_softmax_gradient_and_stability
task_18 = train_and_tune_softmax_classifier
task_19 = benchmark_sklearn_softmax_classifier
task_20 = analyze_digit_classification_errors_and_weights
task_21 = verify_binary_equivalence_and_initializations
task_22 = visualize_iris_decision_boundaries


def library_comparisons(P, out):
    """Task 19 off-the-shelf wrapper called by assignment runner."""
    dg = P["digits"]
    Xtr, Xva, Xte = dg["X"]
    ytr, yva, yte = dg["y"]
    c_rows = []
    for C in [0.1, 1.0, 10.0, 100.0]:
        model = LogisticRegression(C=C, max_iter=10000).fit(Xtr[:, 1:], ytr)
        c_rows.append([C, accuracy(yva, model.predict(Xva[:, 1:])), accuracy(yte, model.predict(Xte[:, 1:]))])
    save_table(
        pd.DataFrame(c_rows, columns=["C", "validation_accuracy", "test_accuracy"]),
        "d19_library_comparisons",
        out,
    )


def part_d(d, out):
    """Run all Digits tasks in Part D (Tasks 17, 18, 19, 20, 21)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== Verifying Softmax Gradient and Numerical Stability (Task 17) ===")
    verify_softmax_gradient_and_stability(d, out)

    print("\n=== Training & Tuning Softmax Classifier (Task 18) ===")
    t18_res = train_and_tune_softmax_classifier(d, out)

    print("\n=== Benchmarking with Scikit-Learn Softmax (Task 19) ===")
    benchmark_sklearn_softmax_classifier(d, t18_res["best_lam"], t18_res["best_W"], out)

    print("\n=== Analyzing Digits Classification Errors & Weight Templates (Task 20) ===")
    analyze_digit_classification_errors_and_weights(d, t18_res["best_W"], out)

    # For Task 21 we also pass breast cancer data for 2-class equivalence check
    from part_a import load_all, prepare
    all_prepared = prepare(load_all())

    print("\n=== Verifying Binary Equivalence and Weight Initializations (Task 21) ===")
    verify_binary_equivalence_and_initializations(d, all_prepared["breast_cancer"], out)

    print(f"\nPart D passed completely. Tables and figures saved to {out}.")
