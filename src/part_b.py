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
from part_a import save_table
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
            break
        losses[epoch] = loss
        distances[epoch] = distance
    return (weights, losses, distances)


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
    return (weights, np.asarray(epoch_losses), np.asarray(update_losses))


def task_5(dataset, out):
    from sklearn.linear_model import LinearRegression

    train, validation, test = dataset["X"]
    y_train, y_validation, y_test = dataset["y"]
    weights = normal_equation(train, y_train)
    model = LinearRegression().fit(train[:, 1:], y_train)
    difference = np.max(np.abs(weights - np.r_[model.intercept_, model.coef_]))
    condition, _, _ = hessian_spectrum(train)
    test_r2 = r2(y_test, test @ weights)
    assert difference < 1e-06
    assert 0.4 <= test_r2 <= 0.5 and 100 < condition < 1000
    names = ["bias"] + list(dataset["names"])
    save_table(pd.DataFrame({"feature": names, "weight": weights}), "b5_weights", out)
    rows = [
        [name, rmse(y, X @ weights)]
        for name, X, y in zip(
            ["train", "validation", "test"], dataset["X"], dataset["y"]
        )
    ]
    save_table(pd.DataFrame(rows, columns=["split", "RMSE"]), "b5_rmse", out)
    save_table(
        pd.DataFrame(
            [[test_r2, condition, difference]],
            columns=["test_R2", "condition_XTX", "max_weight_difference_sklearn"],
        ),
        "b5_checks",
        out,
    )
    return weights


def learning_rates(X, y, reference, rates, out, prefix):
    rows, histories = ([], [])
    plt.figure()
    for rate in rates:
        _, loss, distance = gd_trace(X, y, 2000, rate, reference)
        reached = np.flatnonzero(distance[1:] < 0.001) + 1
        first = int(reached[0]) if len(reached) else "not reached"
        status = "diverged" if not np.isfinite(loss[-1]) else "finite"
        rows.append([rate, loss[-1], distance[-1], first, status])
        histories.append((loss, distance))
        visible = np.where(np.isfinite(loss) & (loss <= 1000000000000.0), loss, np.nan)
        plt.semilogy(np.arange(len(loss)), visible, label=f"eta={rate:g}")
    plt.title(prefix + ": gradient descent")
    plt.xlabel("Epoch")
    plt.ylabel("Full-training data loss")
    plt.legend()
    savefig(out, prefix)
    save_table(
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
    )
    return histories


def tasks_6_and_7(dataset, weights, out):
    train = dataset["X"][0]
    y = dataset["y"][0]
    histories = learning_rates(
        train, y, weights, [0.001, 0.01, 0.1, 0.5, 1.0], out, "b6_rates"
    )
    assert sum((not np.isfinite(loss[-1]) for loss, _ in histories)) == 1
    assert sum((np.any(distance[1:] < 0.001) for _, distance in histories)) == 1
    condition, largest, _ = hessian_spectrum(train)
    threshold = 2 / largest
    rows = []
    plt.figure()
    for factor in [0.95, 1.05]:
        _, loss, distance = gd_trace(train, y, 2000, factor * threshold, weights)
        rows.append([factor, factor * threshold, loss[-1], distance[-1]])
        plt.semilogy(loss, label=f"{factor:g} times stability limit")
    assert rows[0][-1] < 0.001 and rows[1][2] > rows[0][2] * 100
    plt.title("Task 6: stability threshold")
    plt.xlabel("Epoch")
    plt.ylabel("Full-training data loss")
    plt.legend()
    savefig(out, "b6_stability")
    save_table(
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
    )
    raw = add_bias(dataset["raw"][0])
    raw_weights = normal_equation(raw, y)
    raw_condition, raw_largest, _ = hessian_spectrum(raw)
    raw_histories = learning_rates(
        raw, y, raw_weights, [1e-08, 1e-06, 1e-05, 0.0001], out, "b7_raw_rates"
    )
    assert not np.isfinite(raw_histories[-1][0][-1])
    assert 10000000.0 < raw_condition < 100000000.0
    assert all(
        (
            loss[-1] > linear_loss(weights, train, y) * 1.01
            for loss, _ in raw_histories[:-1]
        )
    )
    save_table(
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
    )


def task_8(dataset, out):
    train = dataset["X"][0]
    y = dataset["y"][0]
    rows, traces = ([], {})
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
    budget = min((len(trace) - 1 for trace in traces.values()))
    best_update = min(traces, key=lambda size: traces[size][budget])
    save_table(
        pd.DataFrame(
            [[best_epoch, best_update, budget]],
            columns=[
                "best_batch_at_100_epochs",
                "best_batch_at_common_updates",
                "common_update_budget",
            ],
        ),
        "b8_best_batches",
        out,
    )
    savefig(out, "b8_batches")
    save_table(
        pd.DataFrame(rows, columns=["batch_size", "updates", "final_data_loss"]),
        "b8_batches_table",
        out,
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
    savefig(out, "b8_sgd_rates")
    save_table(
        pd.DataFrame(
            rows, columns=["learning_rate", "final_data_loss", "last_20_epoch_loss_std"]
        ),
        "b8_sgd_noise",
        out,
    )


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
    example = pd.DataFrame(results, columns=["bias", "slope"])
    example.insert(
        0,
        "method",
        ["normal equation", "one GD update", "one SGD update", "ridge lambda=1"],
    )
    save_table(example, "b9_worked_example", out)
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
        assert error < 1e-06
        errors.append([lam, error])
    save_table(
        pd.DataFrame(errors, columns=["lambda", "gradient_relative_error"]),
        "b9_gradient_checks",
        out,
    )
    expanded = [polynomial_degree2(raw, True)[0] for raw in dataset["raw"]]
    assert expanded[0].shape[1] == 64
    scaler = Standardizer().fit(expanded[0])
    expanded = tuple((add_bias(scaler.transform(X)) for X in expanded))
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
        save_table(
            pd.DataFrame({"lambda": lambdas, "validation_RMSE": scores}),
            f"b9_{label}_validation",
            out,
        )
        coefficients = pd.DataFrame(weights, columns=["bias"] + feature_names)
        coefficients.insert(0, "lambda", lambdas)
        coefficients.to_csv(out / f"b9_{label}_coefficients.csv", index=False)
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
        axes[1].set_xscale("symlog", linthresh=0.001)
        axes[1].axvline(
            lambdas[best], linestyle="--", color="gray", label="Selected lambda"
        )
        axes[1].set(
            title=f"Task 9: {label} validation",
            xlabel="Lambda (zero + log grid)",
            ylabel="RMSE",
        )
        axes[1].legend()
        savefig(out, f"b9_{label}_ridge")
        save_table(
            pd.DataFrame(
                [
                    [0.0, scores[0], rmse(y_test, test @ weights[0])],
                    [lambdas[best], scores[best], rmse(y_test, test @ weights[best])],
                ],
                columns=["lambda", "validation_RMSE", "test_RMSE"],
            ),
            f"b9_{label}_test",
            out,
        )
        condition, _, _ = hessian_spectrum(train)
        save_table(
            pd.DataFrame(
                [[label, condition, lambdas[best]]],
                columns=["feature_set", "condition_XTX", "selected_lambda"],
            ),
            f"b9_{label}_selection",
            out,
        )
        if label == "degree2":
            assert 100000000.0 < condition < 10000000000.0
            assert scores[0] > scores[best] + 3
    train = dataset["X"][0]
    target = normal_equation(train, y_train, 0.1)
    weights, _, _ = gd_trace(train, y_train, 3000, 0.4, target, lam=0.1)
    model = Ridge(alpha=0.1 * len(y_train)).fit(train[:, 1:], y_train)
    library_weights = np.r_[model.intercept_, model.coef_]
    gd_error = relative_distance(weights, target)
    library_error = relative_distance(library_weights, target)
    assert gd_error < 1e-06
    assert np.max(np.abs(library_weights - target)) < 1e-06
    save_table(
        pd.DataFrame(
            [[gd_error, library_error]],
            columns=["GD_relative_error", "sklearn_Ridge_relative_error"],
        ),
        "b9_ridge_checks",
        out,
    )


def task_10(dataset, reference, out):
    from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor

    train, _, test = dataset["X"]
    y_train, _, y_test = dataset["y"]
    rows, fitted = ([], {})
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
    save_table(
        pd.DataFrame(
            rows, columns=["method", "test_RMSE", "distance_to_Task5", "fit_seconds"]
        ),
        "b10_comparison",
        out,
    )


def part_b(dataset, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    weights = task_5(dataset, out)
    tasks_6_and_7(dataset, weights, out)
    task_8(dataset, out)
    task_9(dataset, out)
    task_10(dataset, weights, out)
    print(f"\nPart B passed. Tables and figures saved to {out}.")
    return {"weights": weights}
