"""Part B, Tasks 5–10. Owner: Mostafa."""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from assignment2_core import (
    Standardizer,
    add_bias,
    grad_check,
    hessian_spectrum,
    linear_grad,
    linear_loss,
    normal_equation,
    polynomial_degree2,
    r2,
    rmse,
)
from part_a import record_table
from plot_utils import savefig


def relative_distance(weights, reference):
    return float(np.linalg.norm(weights - reference) / np.linalg.norm(reference))


def gd_trace(X, y, epochs, rate, reference, lam=0.0):
    """Return full-training data loss and distance after each GD epoch."""
    weights = np.zeros(X.shape[1])
    losses = np.full(epochs + 1, np.inf)
    distances = np.full(epochs + 1, np.inf)
    losses[0] = linear_loss(weights, X, y)
    distances[0] = relative_distance(weights, reference)
    for epoch in range(1, epochs + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            weights -= rate * linear_grad(weights, X, y, lam)
            loss = linear_loss(weights, X, y)
            distance = relative_distance(weights, reference)
        if not np.isfinite(loss) or not np.isfinite(distance):
            # Numerical overflow means the remaining history is divergent.
            break
        losses[epoch] = loss
        distances[epoch] = distance
    return weights, losses, distances


def mini_batch_trace(X, y, batch_size, epochs, rate):
    """Measure full-training loss after updates for fair batch comparisons."""
    weights = np.zeros(X.shape[1])
    rng = np.random.default_rng(42)
    epoch_losses = [linear_loss(weights, X, y)]
    update_losses = [epoch_losses[0]]
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            indices = order[start : start + batch_size]
            weights -= rate * linear_grad(weights, X[indices], y[indices])
            update_losses.append(linear_loss(weights, X, y))
        epoch_losses.append(update_losses[-1])
    return weights, np.asarray(epoch_losses), np.asarray(update_losses)


def figure_note(out, name, interpretation):
    savefig(out, name)
    return f"![{name}]({name}.png)\n\n{interpretation}\n\n"


def task_5(dataset, out):
    from sklearn.linear_model import LinearRegression

    train, validation, test = dataset["X"]
    y_train, y_validation, y_test = dataset["y"]
    weights = normal_equation(train, y_train)
    model = LinearRegression().fit(train[:, 1:], y_train)
    difference = np.max(np.abs(weights - np.r_[model.intercept_, model.coef_]))
    condition, _, _ = hessian_spectrum(train)
    test_r2 = r2(y_test, test @ weights)
    assert difference < 1e-6
    assert 0.4 <= test_r2 <= 0.5 and 100 < condition < 1000
    report = r"""## Task 5 — Normal equations

\(J(w)=\frac{1}{2N}(Xw-y)^T(Xw-y)\).
Differentiating gives \(\nabla J=X^T(Xw-y)/N\).
Setting this to zero gives \(X^TXw=X^Ty\).
We solve this system using np.linalg.solve rather than forming an inverse.

"""
    names = ["bias"] + list(dataset["names"])
    bmi, s1, s2 = [weights[names.index(name)] for name in ("bmi", "s1", "s2")]
    report += record_table(
        pd.DataFrame({"feature": names, "weight": weights}),
        "b5_weights",
        out,
        f"The standardized bmi coefficient is {bmi:.3f}, giving a positive association when other inputs are fixed. "
        f"The s1 and s2 coefficients are {s1:.3f} and {s2:.3f}, with opposite signs despite their related measurements. "
        "Correlated features can exchange explanatory weight, so individual coefficients should not be interpreted as independent causal effects.",
    )
    rows = [
        [name, rmse(y, X @ weights)]
        for name, X, y in zip(
            ["train", "validation", "test"], dataset["X"], dataset["y"]
        )
    ]
    report += record_table(
        pd.DataFrame(rows, columns=["split", "RMSE"]),
        "b5_rmse",
        out,
        "RMSE is measured in original target units because the target was not standardized.",
    )
    report += record_table(
        pd.DataFrame(
            [[test_r2, condition, difference]],
            columns=["test_R2", "condition_XTX", "max_weight_difference_sklearn"],
        ),
        "b5_checks",
        out,
        "R² and conditioning match the sanity ranges and the exact solvers agree within tolerance.",
    )
    return weights, report


def learning_rates(X, y, reference, rates, out, prefix):
    rows, histories = [], []
    plt.figure()
    for rate in rates:
        _, loss, distance = gd_trace(X, y, 2000, rate, reference)
        reached = np.flatnonzero(distance[1:] < 1e-3) + 1
        first = int(reached[0]) if len(reached) else "not reached"
        status = "diverged" if not np.isfinite(loss[-1]) else "finite"
        rows.append([rate, loss[-1], distance[-1], first, status])
        histories.append((loss, distance))
        visible = np.where(np.isfinite(loss) & (loss <= 1e12), loss, np.nan)
        plt.semilogy(np.arange(len(loss)), visible, label=f"eta={rate:g}")
    plt.title(prefix + ": gradient descent")
    plt.xlabel("Epoch")
    plt.ylabel("Full-training data loss")
    plt.legend()
    report = figure_note(
        out,
        prefix,
        "The figure displays losses up to 1e12 for readability; the table retains the final loss. "
        "After numerical overflow the remaining history is marked as infinity.",
    )
    report += record_table(
        pd.DataFrame(
            rows,
            columns=[
                "learning_rate",
                "final_data_loss",
                "relative_distance",
                "first_epoch_below_1e-3",
                "status",
            ],
        ),
        prefix + "_table",
        out,
        "The relative distance measures convergence to the exact solution in the same feature coordinates.",
    )
    return histories, report


def tasks_6_and_7(dataset, weights, out):
    train = dataset["X"][0]
    y = dataset["y"][0]
    report = "## Task 6 — Batch gradient descent\n\n"
    histories, note = learning_rates(
        train, y, weights, [0.001, 0.01, 0.1, 0.5, 1.0], out, "b6_rates"
    )
    report += note
    assert sum(not np.isfinite(loss[-1]) for loss, _ in histories) == 1
    assert sum(np.any(distance[1:] < 1e-3) for _, distance in histories) == 1
    condition, largest, _ = hessian_spectrum(train)
    threshold = 2 / largest
    rows = []
    plt.figure()
    for factor in [0.95, 1.05]:
        _, loss, distance = gd_trace(train, y, 2000, factor * threshold, weights)
        rows.append([factor, factor * threshold, loss[-1], distance[-1]])
        plt.semilogy(loss, label=f"{factor:g} times stability limit")
    assert rows[0][-1] < 1e-3 and rows[1][2] > rows[0][2] * 100
    plt.title("Task 6: stability threshold")
    plt.xlabel("Epoch")
    plt.ylabel("Full-training data loss")
    plt.legend()
    report += figure_note(
        out,
        "b6_stability",
        "The rate below the stability limit approaches the exact solution, while the rate above it amplifies the error.",
    )
    report += record_table(
        pd.DataFrame(
            rows,
            columns=[
                "limit_factor",
                "learning_rate",
                "final_data_loss",
                "relative_distance",
            ],
        ),
        "b6_stability_table",
        out,
        f"The largest Hessian eigenvalue is {largest:.8g}; the learning-rate limit is {threshold:.8g}.",
    )
    report += r"""Writing \(e_t=w_t-w^*\) and \(H=X^TX/N\), we obtain
\(e_{t+1}=(I-\eta H)e_t\).
Each eigencomponent is multiplied by \(1-\eta\lambda_i\).
Convergence requires \(|1-\eta\lambda_i|<1\), so \(0<\eta<2/\lambda_{\max}\).

## Task 7 — Why scaling matters

"""
    raw = add_bias(dataset["raw"][0])
    raw_weights = normal_equation(raw, y)
    raw_condition, raw_largest, _ = hessian_spectrum(raw)
    raw_histories, note = learning_rates(
        raw, y, raw_weights, [1e-8, 1e-6, 1e-5, 1e-4], out, "b7_raw_rates"
    )
    report += note
    assert not np.isfinite(raw_histories[-1][0][-1])
    assert 1e7 < raw_condition < 1e8
    assert all(
        loss[-1] > linear_loss(weights, train, y) * 1.01
        for loss, _ in raw_histories[:-1]
    )
    report += record_table(
        pd.DataFrame(
            [
                ["standardized", condition, largest, threshold],
                ["raw", raw_condition, raw_largest, 2 / raw_largest],
            ],
            columns=[
                "features",
                "condition_XTX",
                "largest_eigenvalue",
                "stability_limit",
            ],
        ),
        "b7_conditioning",
        out,
        "The condition number measures the spread of curvature across directions. "
        "A large maximum eigenvalue forces a small stable learning rate. "
        "When the minimum eigenvalue is much smaller, that rate makes progress along flat directions slow. "
        "Standardization reduces the observed curvature imbalance and allows useful progress within 2000 epochs.",
    )
    return report


def task_8(dataset, out):
    train = dataset["X"][0]
    y = dataset["y"][0]
    rows, traces = [], {}
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    for size in [1, 16, 64, len(y)]:
        _, epochs, updates = mini_batch_trace(train, y, size, 100, 0.01)
        traces[size] = updates
        rows.append([size, len(updates) - 1, epochs[-1]])
        axes[0].plot(epochs, label=f"B={size}")
        axes[1].plot(updates, label=f"B={size}")
    axes[0].set(
        title="Task 8: progress per epoch",
        xlabel="Epoch",
        ylabel="Full-training data loss",
    )
    axes[1].set(
        title="Task 8: progress per update",
        xlabel="Parameter updates",
        ylabel="Full-training data loss",
    )
    for axis in axes:
        axis.set_yscale("log")
        axis.legend()
    best_epoch = min(rows, key=lambda row: row[-1])[0]
    budget = min(len(trace) - 1 for trace in traces.values())
    best_update = min(traces, key=lambda size: traces[size][budget])
    report = "## Task 8 — SGD and mini-batches\n\n"
    report += figure_note(
        out,
        "b8_batches",
        f"At 100 epochs the best final loss comes from B={best_epoch}; at a shared budget of {budget} updates, B={best_update} is best. "
        "One-sample gradients are noisy because each update reflects one patient rather than the full training distribution.",
    )
    report += record_table(
        pd.DataFrame(rows, columns=["batch_size", "updates", "final_data_loss"]),
        "b8_batches_table",
        out,
        "Every epoch reshuffles rows and retains the last partial batch; full-training loss is measured for all batch sizes.",
    )
    rows = []
    plt.figure()
    for rate in [0.001, 0.01, 0.1]:
        _, epochs, _ = mini_batch_trace(train, y, 1, 100, rate)
        rows.append([rate, epochs[-1], np.std(epochs[-20:])])
        plt.plot(epochs, label=f"eta={rate:g}")
    plt.yscale("log")
    plt.title("Task 8: SGD learning rates")
    plt.xlabel("Epoch")
    plt.ylabel("Full-training data loss")
    plt.legend()
    report += figure_note(
        out,
        "b8_sgd_rates",
        "Larger steps give individual samples more influence and cause stronger fluctuations near the optimum; smaller steps reduce these fluctuations but may require more updates.",
    )
    report += record_table(
        pd.DataFrame(
            rows, columns=["learning_rate", "final_data_loss", "last_20_epoch_loss_std"]
        ),
        "b8_sgd_noise",
        out,
        "The standard deviation measures observed loss fluctuations and any remaining drift over the last 20 epochs.",
    )
    return report


def task_9(dataset, out):
    from sklearn.linear_model import Ridge

    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([2.0, 3.0, 5.0])
    results = [
        normal_equation(X, y),
        -0.1 * linear_grad(np.zeros(2), X, y),
        -0.1 * linear_grad(np.zeros(2), X[1:2], y[1:2]),
        normal_equation(X, y, 1.0),
    ]
    expected = np.array([[1 / 3, 1.5], [1 / 3, 23 / 30], [0.3, 0.6], [32 / 15, 0.6]])
    assert np.allclose(results, expected, atol=1e-12, rtol=0)
    report = r"""## Task 9 — Ridge

\(J_\lambda(w)=\frac{1}{2N}\|Xw-y\|^2+\frac{\lambda}{2}w^TI_0w\).
Its gradient is \(X^T(Xw-y)/N+\lambda I_0w\).
Setting it to zero gives \((X^TX/N+\lambda I_0)w=X^Ty/N\).
We solve this system directly, with the first entry of \(I_0\) set to zero.

For the worked example, \(X^TX=\begin{bmatrix}3&6\\6&14\end{bmatrix}\)
and \(X^Ty=[10,23]^T\).
The equations \(3w_0+6w_1=10\), \(6w_0+14w_1=23\) give \(w=[1/3,3/2]^T\).
At zero weights the batch gradient is \([-10/3,-23/3]^T\), so one step of size 0.1 gives \([1/3,23/30]^T\).
For the second sample alone the gradient is \([-3,-6]^T\), giving \([0.3,0.6]^T\).
With lambda equal to one, the ridge equations become \(w_0+2w_1=10/3\)
and \(2w_0+(17/3)w_1=23/3\), giving \(w=[32/15,3/5]^T\).

"""
    example = pd.DataFrame(results, columns=["bias", "slope"])
    example.insert(
        0,
        "method",
        ["normal equation", "one GD update", "one SGD update", "ridge lambda=1"],
    )
    report += record_table(
        example,
        "b9_worked_example",
        out,
        "All four results match the hand-computed example in Section 3.2.",
    )
    train = dataset["X"][0]
    y_train, y_validation, y_test = dataset["y"]
    point = np.random.default_rng(42).normal(0, 0.1, train.shape[1])
    errors = []
    for lam in [0.0, 0.1]:
        error = grad_check(
            lambda w: linear_loss(w, train, y_train, lam),
            lambda w: linear_grad(w, train, y_train, lam),
            point,
        )
        assert error < 1e-6
        errors.append([lam, error])
    report += record_table(
        pd.DataFrame(errors, columns=["lambda", "gradient_relative_error"]),
        "b9_gradient_checks",
        out,
        "Both the squared-error gradient and ridge gradient pass the required numerical check.",
    )
    expanded = [polynomial_degree2(raw, True)[0] for raw in dataset["raw"]]
    assert expanded[0].shape[1] == 64
    scaler = Standardizer().fit(expanded[0])
    expanded = tuple(add_bias(scaler.transform(X)) for X in expanded)
    names = list(dataset["names"])
    pairs = polynomial_degree2(dataset["raw"][0], True)[1]
    polynomial_names = names + [f"{names[i]}*{names[j]}" for i, j in pairs]
    for label, matrices, feature_names in [
        ("original", dataset["X"], names),
        ("degree2", expanded, polynomial_names),
    ]:
        train, validation, test = matrices
        lambdas = np.r_[0.0, np.logspace(-3, 1, 9)]
        weights = [normal_equation(train, y_train, lam) for lam in lambdas]
        scores = [rmse(y_validation, validation @ w) for w in weights]
        best = int(np.argmin(scores))
        report += record_table(
            pd.DataFrame({"lambda": lambdas, "validation_RMSE": scores}),
            f"b9_{label}_validation",
            out,
            f"Lambda {lambdas[best]:g} is selected using validation RMSE only.",
        )
        coefficients = pd.DataFrame(weights, columns=["bias"] + feature_names)
        coefficients.insert(0, "lambda", lambdas)
        # Save all 65 coefficient paths as a separate full CSV for notebook assembly.
        coefficients.to_csv(out / f"b9_{label}_coefficients.csv", index=False)
        report += f"Full coefficient table: [CSV](b9_{label}_coefficients.csv).\n\n"
        report += "The table contains every coefficient at each lambda, including zero; the path plot below summarizes the shrinkage.\n\n"
        _, axes = plt.subplots(1, 2, figsize=(14, 6))
        for index, name in enumerate(["bias"] + feature_names):
            axes[0].plot(lambdas[1:], np.asarray(weights)[1:, index], label=name)
        axes[0].set_xscale("log")
        axes[0].set(
            title=f"Task 9: {label} coefficient paths",
            xlabel="Lambda (positive grid)",
            ylabel="Coefficient",
        )
        axes[0].legend(fontsize=5, ncol=4, loc="best")
        axes[1].plot(lambdas, scores, "o-", label="Validation RMSE")
        axes[1].set_xscale("symlog", linthresh=1e-3)
        axes[1].axvline(
            lambdas[best], linestyle="--", color="gray", label="Selected lambda"
        )
        axes[1].set(
            title=f"Task 9: {label} validation",
            xlabel="Lambda (zero + log grid)",
            ylabel="RMSE",
        )
        axes[1].legend()
        report += figure_note(
            out,
            f"b9_{label}_ridge",
            "A log coefficient axis shows positive lambda values; lambda-zero coefficients are in the CSV. "
            "The validation plot includes zero using a symmetric-log axis.",
        )
        report += record_table(
            pd.DataFrame(
                [
                    [0.0, scores[0], rmse(y_test, test @ weights[0])],
                    [lambdas[best], scores[best], rmse(y_test, test @ weights[best])],
                ],
                columns=["lambda", "validation_RMSE", "test_RMSE"],
            ),
            f"b9_{label}_test",
            out,
            "Test reporting compares the requested unregularized baseline and the validation-selected model.",
        )
        condition, _, _ = hessian_spectrum(train)
        report += f"The {label} condition number of X^T X is {condition:.6g}.\n\n"
        if label == "degree2":
            assert 1e8 < condition < 1e10
            assert scores[0] > scores[best] + 3
    train = dataset["X"][0]
    target = normal_equation(train, y_train, 0.1)
    weights, _, _ = gd_trace(train, y_train, 3000, 0.4, target, lam=0.1)
    model = Ridge(alpha=0.1 * len(y_train)).fit(train[:, 1:], y_train)
    library_weights = np.r_[model.intercept_, model.coef_]
    gd_error = relative_distance(weights, target)
    library_error = relative_distance(library_weights, target)
    assert gd_error < 1e-6
    assert np.max(np.abs(library_weights - target)) < 1e-6
    report += record_table(
        pd.DataFrame(
            [[gd_error, library_error]],
            columns=["GD_relative_error", "sklearn_Ridge_relative_error"],
        ),
        "b9_ridge_checks",
        out,
        "GD reaches the ridge closed form and Ridge uses alpha = lambda times the training-row count.",
    )
    report += (
        "The ten-feature model has relatively few parameters for 264 training rows. "
        "The 64-feature expansion adds correlated terms and has a much higher condition number. "
        "Ridge reduces the instability of those coefficients, producing a larger validation improvement for the expansion. "
        "The bias is excluded from regularization so shrinking feature effects does not force the baseline prediction toward zero.\n\n"
    )
    return report


def task_10(dataset, reference, out):
    from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor

    train, _, test = dataset["X"]
    y_train, _, y_test = dataset["y"]
    rows, fitted = [], {}
    methods = [
        ("NumPy normal equation", lambda: normal_equation(train, y_train)),
        ("NumPy batch GD", lambda: gd_trace(train, y_train, 2000, 0.5, reference)[0]),
        (
            "NumPy mini-batch B=16",
            lambda: mini_batch_trace(train, y_train, 16, 100, 0.01)[0],
        ),
        ("NumPy ridge", lambda: normal_equation(train, y_train, 0.1)),
    ]
    for label, fit in methods:
        start = perf_counter()
        weights = fit()
        seconds = perf_counter() - start
        fitted[label] = weights
        rows.append(
            [
                label,
                rmse(y_test, test @ weights),
                relative_distance(weights, reference),
                seconds,
            ]
        )
    models = [
        ("sklearn LinearRegression", LinearRegression()),
        ("sklearn Ridge", Ridge(alpha=0.1 * len(y_train))),
    ]
    models += [
        (
            f"sklearn SGD eta={rate:g}",
            SGDRegressor(
                penalty=None,
                learning_rate="constant",
                eta0=rate,
                max_iter=1000,
                tol=None,
                random_state=42,
            ),
        )
        for rate in [0.001, 0.01, 0.1]
    ]
    for label, model in models:
        start = perf_counter()
        model.fit(train[:, 1:], y_train)
        seconds = perf_counter() - start
        weights = np.r_[model.intercept_, model.coef_]
        fitted[label] = weights
        rows.append(
            [
                label,
                rmse(y_test, test @ weights),
                relative_distance(weights, reference),
                seconds,
            ]
        )
    assert np.max(np.abs(fitted["sklearn LinearRegression"] - reference)) < 1e-10
    assert np.max(np.abs(fitted["sklearn Ridge"] - fitted["NumPy ridge"])) < 1e-10
    assert rows[-1][1] > rows[-3][1]
    report = "## Task 10 — From scratch versus frameworks\n\n"
    report += record_table(
        pd.DataFrame(
            rows, columns=["method", "test_RMSE", "distance_to_Task5", "fit_seconds"]
        ),
        "b10_comparison",
        out,
        "Every distance uses the Task 5 unregularized solution, so ridge has nonzero distance by design. "
        "Timings include history recording for NumPy iterative methods and vary with machine and load.",
    )
    report += (
        "Normal equations and LinearRegression agree to numerical precision, and the ridge solvers agree with one another. "
        "Finite-epoch GD and mini-batch GD stop before the exact solution, and SGDRegressor also uses a different update sequence. "
        "The larger constant SGD step gives worse test RMSE in this run because stochastic updates fluctuate more strongly near the solution.\n"
    )
    return report


def part_b(dataset, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    weights, report = task_5(dataset, out)
    report = "# Part B — Linear regression\n\n" + report
    report += tasks_6_and_7(dataset, weights, out)
    report += task_8(dataset, out)
    report += task_9(dataset, out)
    report += task_10(dataset, weights, out)
    (out / "part_b.md").write_text(report)
    print(f"\nPart B passed. Tables, figures, and explanations saved to {out}.")
    return {"weights": weights, "report": out / "part_b.md"}
