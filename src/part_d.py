"""Part D: Softmax classification and multi-class experiments. Owner: Ziad."""

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


def check_softmax_gradient_and_stability(dataset, out):
    """Verify analytical gradient, numerical stability, and baseline loss at W=0."""
    X_train, _, _ = dataset["X"]
    y_train, _, _ = dataset["y"]
    num_classes = 10

    # Theoretical loss at W = 0 equals ln(K) = ln(10) ~ 2.302585
    W_zero = np.zeros((X_train.shape[1], num_classes))
    loss_at_zero = softmax_loss(W_zero, X_train, y_train, lam=0.0)
    expected_zero = float(np.log(num_classes))
    diff_zero = abs(loss_at_zero - expected_zero)
    assert diff_zero < 1e-12, f"Loss at W=0 ({loss_at_zero}) does not match ln(10) ({expected_zero})"

    # Numerical stability check for large input scores [1000, 1001, 1002]
    scores_test = np.array([[1000.0, 1001.0, 1002.0]])
    probs_computed = softmax(scores_test)[0]
    expected_probs = np.array([0.09003057, 0.24472847, 0.66524096])
    assert np.allclose(probs_computed, expected_probs, atol=1e-4), f"Softmax stability check failed: {probs_computed}"

    # Gradient check on 100 randomly chosen weights at lambda = 0.1
    rng = np.random.default_rng(42)
    W_init = rng.normal(0, 0.1, (X_train.shape[1], num_classes))
    sample_indices = rng.choice(W_init.size, size=100, replace=False)
    grad_error = grad_check(
        lambda w: softmax_loss(w.reshape(W_init.shape), X_train, y_train, 0.1),
        lambda w: softmax_grad(w.reshape(W_init.shape), X_train, y_train, 0.1).ravel(),
        W_init.ravel(),
        entries=sample_indices,
    )
    assert grad_error < 1e-06, f"Softmax gradient check failed: {grad_error}"

    save_table(
        pd.DataFrame(
            [[loss_at_zero, expected_zero, diff_zero, grad_error]],
            columns=["W_zero_loss", "ln_10_expected", "zero_loss_diff", "grad_check_relative_error"],
        ),
        "d17_checks",
        out,
    )
    save_table(
        pd.DataFrame(
            [list(expected_probs), list(probs_computed)],
            index=["expected", "computed_stable"],
            columns=["s1=1000", "s2=1001", "s3=1002"],
        ).reset_index(),
        "d17_numerical_stability",
        out,
    )


def train_softmax_classifier(dataset, out):
    """Train softmax classifier using mini-batch GD with hyperparameter tuning."""
    X_train, X_val, X_test = dataset["X"]
    y_train, y_val, y_test = dataset["y"]
    num_samples = len(y_train)
    num_classes = 10
    W_zeros = np.zeros((X_train.shape[1], num_classes))

    # Evaluate learning rate candidates with lambda = 0
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    lr_sweep_records = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for eta in learning_rates:
        rng = np.random.default_rng(42)
        train_loss_hist, val_loss_hist = [], []
        train_acc_hist, val_acc_hist = [], []
        W_current = W_zeros.copy()

        for epoch in range(100):
            shuffled_order = rng.permutation(num_samples)
            for start_idx in range(0, num_samples, 64):
                batch_indices = shuffled_order[start_idx : start_idx + 64]
                grad = softmax_grad(W_current, X_train[batch_indices], y_train[batch_indices], lam=0.0)
                W_current -= eta * grad

            train_loss_hist.append(softmax_loss(W_current, X_train, y_train, 0.0))
            val_loss_hist.append(softmax_loss(W_current, X_val, y_val, 0.0))
            train_acc_hist.append(accuracy(y_train, (X_train @ W_current).argmax(1)))
            val_acc_hist.append(accuracy(y_val, (X_val @ W_current).argmax(1)))

        axes[0].plot(val_loss_hist, label=f"eta={eta}")
        axes[1].plot(val_acc_hist, label=f"eta={eta}")

        lr_sweep_records.append(
            [
                eta,
                train_loss_hist[-1],
                val_loss_hist[-1],
                train_acc_hist[-1],
                val_acc_hist[-1],
                W_current,
            ]
        )

    axes[0].set(title="Validation Loss vs Epoch", xlabel="Epoch", ylabel="Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set(title="Validation Accuracy vs Epoch", xlabel="Epoch", ylabel="Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    savefig(out, "d18_learning_rates_curves")

    save_table(
        pd.DataFrame(
            [r[:5] for r in lr_sweep_records],
            columns=["learning_rate", "final_train_loss", "final_validation_loss", "train_accuracy", "validation_accuracy"],
        ),
        "d18_learning_rates",
        out,
    )

    # Pick best learning rate by lowest validation loss
    best_lr_row = min(lr_sweep_records, key=lambda r: r[2])
    best_eta = best_lr_row[0]

    # Verify that eta=0.01 underfits compared to eta=0.1
    assert lr_sweep_records[0][1] > lr_sweep_records[1][1], "eta=0.01 should be under-trained after 100 epochs"

    # Regularization sweep with chosen eta
    lambdas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    reg_sweep_records = []
    for lam in lambdas:
        rng = np.random.default_rng(42)
        loss_grad_fn = lambda W, batch_idx: (
            softmax_loss(W, X_train if batch_idx is None else X_train[batch_idx], y_train if batch_idx is None else y_train[batch_idx], lam),
            softmax_grad(W, X_train if batch_idx is None else X_train[batch_idx], y_train if batch_idx is None else y_train[batch_idx], lam),
        )
        W_trained, _, _ = minibatch_gd(loss_grad_fn, W_zeros, num_samples, 64, 100, best_eta, rng)
        tr_acc = accuracy(y_train, (X_train @ W_trained).argmax(1))
        va_acc = accuracy(y_val, (X_val @ W_trained).argmax(1))
        # Validation loss is evaluated on unregularized data loss for fair comparison
        va_loss = softmax_loss(W_trained, X_val, y_val, lam=0.0)
        reg_sweep_records.append([lam, tr_acc, va_acc, va_loss, W_trained])

    save_table(
        pd.DataFrame(
            [r[:4] for r in reg_sweep_records],
            columns=["lambda", "train_accuracy", "validation_accuracy", "validation_loss"],
        ),
        "d18_regularization",
        out,
    )

    # Pick best lambda by validation accuracy (favor larger lambda on tie)
    best_reg_row = max(reg_sweep_records, key=lambda r: (r[2], r[0]))
    best_lambda = best_reg_row[0]
    best_weights = best_reg_row[4]
    test_acc = accuracy(y_test, (X_test @ best_weights).argmax(1))

    # Sanity checks
    assert 0.95 <= best_reg_row[2] <= 0.98, f"Best validation accuracy {best_reg_row[2]} should be between 0.95 and 0.98"
    assert reg_sweep_records[-1][2] < 0.80, "lambda=0.1 should be clearly over-regularized"

    save_table(
        pd.DataFrame(
            [[best_eta, best_lambda, best_reg_row[2], test_acc]],
            columns=["selected_eta", "selected_lambda", "best_validation_accuracy", "test_accuracy"],
        ),
        "d18_selected_model",
        out,
    )

    return {
        "best_eta": best_eta,
        "best_lambda": best_lambda,
        "best_weights": best_weights,
        "test_accuracy": test_acc,
    }


def compare_sklearn_softmax(dataset, best_lambda, best_weights, out):
    """Benchmark custom softmax against scikit-learn LogisticRegression."""
    X_train, X_val, X_test = dataset["X"]
    y_train, y_val, y_test = dataset["y"]
    num_train = len(y_train)

    # Sweep scikit-learn regularization parameter C
    c_records = []
    for C in [0.1, 1.0, 10.0, 100.0]:
        model = LogisticRegression(C=C, max_iter=10000).fit(X_train[:, 1:], y_train)
        val_acc = accuracy(y_val, model.predict(X_val[:, 1:]))
        test_acc = accuracy(y_test, model.predict(X_test[:, 1:]))
        c_records.append([C, val_acc, test_acc])

    save_table(
        pd.DataFrame(c_records, columns=["C", "validation_accuracy", "test_accuracy"]),
        "d19_c_sweep",
        out,
    )

    # Match C with our best lambda (C = 1 / (N * lambda))
    C_equivalent = 1.0 / (best_lambda * num_train) if best_lambda > 0 else 1000.0
    sklearn_model = LogisticRegression(C=C_equivalent, max_iter=10000).fit(X_train[:, 1:], y_train)
    our_preds = (X_test @ best_weights).argmax(1)
    sklearn_preds = sklearn_model.predict(X_test[:, 1:])
    agreement = float(np.mean(our_preds == sklearn_preds))

    assert agreement >= 0.97, f"Prediction agreement at matching regularization ({agreement:.4f}) is below 97%"

    save_table(
        pd.DataFrame(
            [[best_lambda, C_equivalent, accuracy(y_test, our_preds), accuracy(y_test, sklearn_preds), agreement]],
            columns=["selected_lambda", "matching_C", "our_test_acc", "sklearn_test_acc", "prediction_agreement"],
        ),
        "d19_verification",
        out,
    )


def analyze_digits_performance(dataset, best_weights, out):
    """Analyze confusion matrix, classification metrics, misclassified samples, and weight templates."""
    _, _, X_test = dataset["X"]
    _, _, y_test = dataset["y"]
    test_predictions = (X_test @ best_weights).argmax(1)

    # Confusion matrix and per-class metrics
    conf_mat, precisions, recalls, f1_scores = class_metrics(y_test, test_predictions, 10)
    plt.figure(figsize=(7, 6))
    plt.imshow(conf_mat, cmap="Blues")
    plt.colorbar()
    plt.title("Digits Test Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    savefig(out, "d20_confusion")

    metrics_summary = pd.DataFrame(
        {
            "class": np.arange(10),
            "precision": precisions,
            "recall": recalls,
            "f1": f1_scores,
        }
    )
    save_table(metrics_summary, "d20_class_metrics", out)

    # Top confused digit pairs
    conf_mat_errors = conf_mat.copy()
    np.fill_diagonal(conf_mat_errors, 0)
    confused_pairs = []
    for i in range(10):
        for j in range(10):
            if conf_mat_errors[i, j] > 0:
                confused_pairs.append([int(i), int(j), int(conf_mat_errors[i, j])])
    confused_pairs.sort(key=lambda row: row[2], reverse=True)
    save_table(
        pd.DataFrame(confused_pairs[:3], columns=["true_class", "predicted_class", "error_count"]),
        "d20_top_confused_pairs",
        out,
    )

    # Visualize 6 misclassified test digits as 8x8 images
    error_indices = np.flatnonzero(test_predictions != y_test)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.5))
    for ax, idx in zip(axes.ravel(), error_indices):
        digit_img = (X_test[idx, 1:] * 16.0).reshape(8, 8)
        ax.imshow(digit_img, cmap="gray")
        ax.set_title(f"True: {y_test[idx]} / Pred: {test_predictions[idx]}")
        ax.axis("off")
    savefig(out, "d20_misclassified")

    # Visualize learned 8x8 weight templates for each digit class
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.5))
    for digit, ax in enumerate(axes.ravel()):
        weight_template = best_weights[1:, digit].reshape(8, 8)
        max_abs_val = np.max(np.abs(weight_template))
        ax.imshow(weight_template, cmap="coolwarm", vmin=-max_abs_val, vmax=max_abs_val)
        ax.set_title(f"Digit {digit}")
        ax.axis("off")
    savefig(out, "d20_weights")


def test_binary_equivalence_and_inits(digits_dataset, bc_dataset, out):
    """Verify equivalence between 2-class softmax and logistic regression, and analyze weight initializations."""
    # Equivalence check on Breast Cancer test features
    _, _, X_test_bc = bc_dataset["X"]
    rng = np.random.default_rng(42)
    W_two_classes = rng.normal(0, 1.0, (X_test_bc.shape[1], 2))

    # Softmax class 1 probability
    p_softmax = softmax(X_test_bc @ W_two_classes)[:, 1]

    # Binary logistic regression probability using parameter difference (w = w1 - w0)
    w_difference = W_two_classes[:, 1] - W_two_classes[:, 0]
    p_logistic = sigmoid(X_test_bc @ w_difference)

    max_prob_diff = float(np.max(np.abs(p_softmax - p_logistic)))
    assert max_prob_diff < 1e-12, f"Equivalence check failed: difference {max_prob_diff} >= 1e-12"

    save_table(
        pd.DataFrame([[max_prob_diff]], columns=["max_absolute_probability_difference"]),
        "d21_equivalence_check",
        out,
    )

    # Compare initialization strategies on Digits
    X_train, X_val, X_test = digits_dataset["X"]
    y_train, y_val, y_test = digits_dataset["y"]
    num_samples = len(y_train)
    num_classes = 10

    init_records = []
    strategies = [
        ("zeros", np.zeros((X_train.shape[1], num_classes))),
        ("random_normal_0.01", rng.normal(0, 0.01, (X_train.shape[1], num_classes))),
        ("random_normal_0.10", rng.normal(0, 0.10, (X_train.shape[1], num_classes))),
    ]

    for name, initial_weights in strategies:
        loss_grad_fn = lambda W, batch_idx: (
            softmax_loss(W, X_train if batch_idx is None else X_train[batch_idx], y_train if batch_idx is None else y_train[batch_idx], 0.001),
            softmax_grad(W, X_train if batch_idx is None else X_train[batch_idx], y_train if batch_idx is None else y_train[batch_idx], 0.001),
        )
        W_fitted, loss_history, _ = minibatch_gd(loss_grad_fn, initial_weights, num_samples, 64, 100, 1.0, np.random.default_rng(42))
        init_records.append(
            [
                name,
                float(loss_history[-1]),
                accuracy(y_val, (X_val @ W_fitted).argmax(1)),
                accuracy(y_test, (X_test @ W_fitted).argmax(1)),
            ]
        )

    save_table(
        pd.DataFrame(init_records, columns=["initialization", "final_train_loss", "validation_accuracy", "test_accuracy"]),
        "d21_initialization",
        out,
    )


def plot_iris_decision_boundaries(prepared_data, out, rng=None):
    """Visualize 2D decision regions for Softmax classifier on Iris dataset."""
    iris_data = prepared_data["iris"]
    raw_train, _, raw_test = iris_data["raw"]
    y_train, _, y_test = iris_data["y"]

    # Select features 2 (petal length) and 3 (petal width)
    raw_train_2d = raw_train[:, [2, 3]]
    raw_test_2d = raw_test[:, [2, 3]]

    scaler = Standardizer().fit(raw_train_2d)
    X_train_2d = add_bias(scaler.transform(raw_train_2d))
    X_test_2d = add_bias(scaler.transform(raw_test_2d))
    num_train = len(y_train)
    num_classes = 3

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    comparison_records = []

    # Create coordinate grid for decision regions
    x1_min, x1_max = X_train_2d[:, 1].min() - 0.8, X_train_2d[:, 1].max() + 0.8
    x2_min, x2_max = X_train_2d[:, 2].min() - 0.8, X_train_2d[:, 2].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 250), np.linspace(x2_min, x2_max, 250))
    grid_points = add_bias(np.column_stack([xx.ravel(), yy.ravel()]))

    for row_idx, lam in enumerate([0.0, 1.0]):
        # Train custom softmax with full-batch gradient descent
        W_init = np.zeros((3, num_classes))
        loss_grad_fn = lambda W, batch_idx: (
            softmax_loss(W, X_train_2d if batch_idx is None else X_train_2d[batch_idx], y_train if batch_idx is None else y_train[batch_idx], lam),
            softmax_grad(W, X_train_2d if batch_idx is None else X_train_2d[batch_idx], y_train if batch_idx is None else y_train[batch_idx], lam),
        )
        W_fit, _, _ = minibatch_gd(loss_grad_fn, W_init, num_train, num_train, 2000, 0.5, np.random.default_rng(42))
        scratch_grid_preds = (grid_points @ W_fit).argmax(1).reshape(xx.shape)
        scratch_acc = accuracy(y_test, (X_test_2d @ W_fit).argmax(1))

        # Train scikit-learn model with equivalent C
        C_equiv = 1e5 if lam == 0 else 1.0 / (lam * num_train)
        sklearn_model = LogisticRegression(C=C_equiv, tol=1e-12, max_iter=10000).fit(X_train_2d[:, 1:], y_train)
        sklearn_grid_preds = sklearn_model.predict(grid_points[:, 1:]).reshape(xx.shape)
        sklearn_acc = accuracy(y_test, sklearn_model.predict(X_test_2d[:, 1:]))

        comparison_records.append([lam, C_equiv, scratch_acc, sklearn_acc])

        # Plot custom model decision regions
        ax_scratch = axes[row_idx, 0]
        ax_scratch.contourf(xx, yy, scratch_grid_preds, alpha=0.3, cmap="viridis")
        ax_scratch.scatter(X_train_2d[:, 1], X_train_2d[:, 2], c=y_train, cmap="viridis", edgecolors="k")
        ax_scratch.set(
            title=f"Custom Softmax (lambda={lam}) - Test Acc: {scratch_acc:.3f}",
            xlabel="Petal Length (standardized)",
            ylabel="Petal Width (standardized)",
        )

        # Plot scikit-learn decision regions
        ax_sklearn = axes[row_idx, 1]
        ax_sklearn.contourf(xx, yy, sklearn_grid_preds, alpha=0.3, cmap="viridis")
        ax_sklearn.scatter(X_train_2d[:, 1], X_train_2d[:, 2], c=y_train, cmap="viridis", edgecolors="k")
        ax_sklearn.set(
            title=f"scikit-learn (C={C_equiv:.4g}) - Test Acc: {sklearn_acc:.3f}",
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


def library_comparisons(prepared_data, out):
    """Off-the-shelf LogisticRegression comparisons on Digits dataset."""
    digits = prepared_data["digits"]
    X_train, X_val, X_test = digits["X"]
    y_train, y_val, y_test = digits["y"]

    records = []
    for C in [0.1, 1.0, 10.0, 100.0]:
        model = LogisticRegression(C=C, max_iter=10000).fit(X_train[:, 1:], y_train)
        records.append([C, accuracy(y_val, model.predict(X_val[:, 1:])), accuracy(y_test, model.predict(X_test[:, 1:]))])

    save_table(
        pd.DataFrame(records, columns=["C", "validation_accuracy", "test_accuracy"]),
        "d19_library_comparisons",
        out,
    )


def part_d(dataset, out):
    """Execute all Softmax classification experiments for Part D."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    print("\nChecking softmax gradient and numerical stability...")
    check_softmax_gradient_and_stability(dataset, out)

    print("\nTraining softmax classifier and tuning hyperparameters...")
    softmax_model = train_softmax_classifier(dataset, out)

    print("\nComparing custom softmax with scikit-learn...")
    compare_sklearn_softmax(dataset, softmax_model["best_lambda"], softmax_model["best_weights"], out)

    print("\nAnalyzing errors, confusion matrix, and learned weight templates...")
    analyze_digits_performance(dataset, softmax_model["best_weights"], out)

    from part_a import load_all, prepare
    all_prepared = prepare(load_all())

    print("\nChecking binary equivalence and weight initialization effects...")
    test_binary_equivalence_and_inits(dataset, all_prepared["breast_cancer"], out)

    print(f"\nPart D completed successfully. Figures and tables saved in {out}.")
