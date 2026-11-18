"""Part A (Tasks 1–4): data preparation and toolkit checks. Owner: Mostafa."""

import numpy as np
import pandas as pd
from pathlib import Path

from assignment2_core import (
    Standardizer,
    add_bias,
    class_metrics,
    macro_metrics,
    mse,
    rmse,
    r2,
    accuracy,
    grad_check,
    standardization_checks,
)


def splits(X, y, classification):
    from sklearn.model_selection import train_test_split

    strat = y if classification else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )
    strat = ytr if classification else None
    Xtr, Xva, ytr, yva = train_test_split(
        Xtr, ytr, test_size=0.25, random_state=42, stratify=strat
    )
    return (Xtr, Xva, Xte, ytr, yva, yte)


def load_all():
    from sklearn.datasets import (
        load_diabetes,
        load_breast_cancer,
        load_digits,
        load_iris,
    )

    db = load_diabetes(scaled=False)
    bc = load_breast_cancer()
    dg = load_digits()
    ir = load_iris()
    return {
        "diabetes": (db.data, db.target, False, db.feature_names),
        "breast_cancer": (bc.data, 1 - bc.target, True, bc.feature_names),
        "digits": (dg.data / 16.0, dg.target, True, [f"pixel_{i}" for i in range(64)]),
        "iris": (ir.data, ir.target, True, ir.feature_names),
    }


def prepare(data):
    prepared = {}
    for name, (X, y, is_cls, names) in data.items():
        Xtr, Xva, Xte, ytr, yva, yte = splits(X, y, is_cls)
        scaler = Standardizer().fit(Xtr) if name != "digits" else None
        transform = (lambda z: scaler.transform(z)) if scaler else lambda z: z
        prepared[name] = dict(
            raw=(Xtr, Xva, Xte),
            X=(
                add_bias(transform(Xtr)),
                add_bias(transform(Xva)),
                add_bias(transform(Xte)),
            ),
            y=(ytr, yva, yte),
            scaler=scaler,
            names=names,
        )
    return prepared


def record_table(frame, name, output_dir, interpretation):
    """Print a result and return Markdown for the final notebook."""
    print("\n" + name)
    print(frame.to_string(index=False))
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_dir / (name + ".csv"), index=False)
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [
            (
                format(value, ".8g")
                if isinstance(value, (float, np.floating))
                else str(value)
            )
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n\n" + interpretation + "\n\n"


def part_a(P, out=None):
    """Run Tasks 1–4 and save the tables and explanation when out is given."""
    expected_sizes = {
        "diabetes": (264, 89, 89),
        "breast_cancer": (341, 114, 114),
        "digits": (1077, 360, 360),
        "iris": (90, 30, 30),
    }
    split_rows, class_rows, scaling_rows = [], [], []
    for name, dataset in P.items():
        targets = dataset["y"]
        sizes = tuple(len(y) for y in targets)
        assert sizes == expected_sizes[name], (name, sizes)
        split_rows.append([name, *sizes])
        if name != "diabetes":
            K = int(np.concatenate(targets).max()) + 1
            proportions = np.bincount(np.concatenate(targets), minlength=K) / sum(sizes)
            for split_name, y in zip(["train", "validation", "test"], targets):
                counts = np.bincount(y, minlength=K)
                assert np.all(np.abs(counts - len(y) * proportions) <= 1.0 + 1e-12)
                for label, count in enumerate(counts):
                    class_rows.append([name, split_name, label, int(count)])
        if dataset["scaler"] is not None:
            train, validation, test = (X[:, 1:] for X in dataset["X"])
            stats = standardization_checks(train, validation, test)
            scaling_rows.append(
                [
                    name,
                    stats["train_mean_max_abs"],
                    stats["train_std_max_abs_error"],
                    float(np.abs(validation.mean(0)).max()),
                    float(np.abs(test.mean(0)).max()),
                    float(np.abs(validation.std(0) - 1).max()),
                    float(np.abs(test.std(0) - 1).max()),
                ]
            )

    # Exercise sigma=0 on a separate small example.
    constant_example = np.array([[1.0, 4.0], [3.0, 4.0], [5.0, 4.0]])
    scaler = Standardizer().fit(constant_example)
    assert scaler.scale_[1] == 1
    assert np.all(scaler.transform(constant_example)[:, 1] == 0)
    assert np.all(add_bias(constant_example)[:, 0] == 1)

    from sklearn import metrics as reference

    rng = np.random.default_rng(42)
    y = rng.integers(0, 4, size=500)
    predicted = rng.integers(0, 4, size=500)
    cm, precision, recall, f1 = class_metrics(y, predicted, 4)
    reference_scores = reference.precision_recall_fscore_support(
        y, predicted, labels=np.arange(4), zero_division=0
    )
    assert np.array_equal(
        cm, reference.confusion_matrix(y, predicted, labels=np.arange(4))
    )
    checks = {
        "mse": (mse(y, predicted), reference.mean_squared_error(y, predicted)),
        "rmse": (
            rmse(y, predicted),
            np.sqrt(reference.mean_squared_error(y, predicted)),
        ),
        "r2": (r2(y, predicted), reference.r2_score(y, predicted)),
        "accuracy": (accuracy(y, predicted), reference.accuracy_score(y, predicted)),
        "precision": (precision, reference_scores[0]),
        "recall": (recall, reference_scores[1]),
        "f1": (f1, reference_scores[2]),
        "macro": (
            macro_metrics(y, predicted, 4),
            reference.precision_recall_fscore_support(
                y, predicted, labels=np.arange(4), average="macro", zero_division=0
            )[:3],
        ),
    }
    metric_rows = []
    for name, (actual, expected) in checks.items():
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        assert error <= 1e-12, (name, error)
        metric_rows.append([name, error])
    _, absent_precision, _, _ = class_metrics(y, np.zeros_like(y), 4)
    assert np.all(absent_precision[1:] == 0)

    # Task 4 checks the checker itself against a quadratic with a known derivative.
    point = rng.normal(0, 0.1, size=5)
    gradient_error = grad_check(lambda w: 0.5 * (w @ w), lambda w: w, point)
    assert gradient_error < 1e-6

    report = "# Part A — Data and toolkit\n\n"
    report += record_table(
        pd.DataFrame(split_rows, columns=["dataset", "train", "validation", "test"]),
        "a_split_sizes",
        out,
        "All four split sizes match the assignment.",
    )
    report += record_table(
        pd.DataFrame(class_rows, columns=["dataset", "split", "class", "count"]),
        "a_class_counts",
        out,
        "Every class count differs from its proportional allocation by at most one sample.",
    )
    report += record_table(
        pd.DataFrame(
            scaling_rows,
            columns=[
                "dataset",
                "train_max_abs_mean",
                "train_max_std_error",
                "validation_max_abs_mean",
                "test_max_abs_mean",
                "validation_max_std_error",
                "test_max_std_error",
            ],
        ),
        "a_standardization",
        out,
        "Training columns have mean zero and standard deviation one within tolerance; held-out statistics generally differ. Digits uses pixels divided by 16, as specified.",
    )
    report += (
        "The mean and standard deviation must be learned from training rows because they are part of the fitted preprocessing. "
        "Computing them from all rows lets held-out observations influence the fitted pipeline, making the test estimate less representative of unseen data.\n\n"
    )
    report += record_table(
        pd.DataFrame(metric_rows, columns=["metric", "max_absolute_error"]),
        "a_metric_checks",
        out,
        "All metric comparisons pass the required 1e-12 tolerance, including macro averages and the never-predicted-class convention.",
    )
    report += f"The central-difference checker has relative error {gradient_error:.3g} on a quadratic at a random point, below 1e-6. "
    report += "The bias-column and zero-variance-column checks also pass.\n"
    if out is not None:
        Path(out, "part_a.md").write_text(report)
    print(f"Part A passed; gradient-check error = {gradient_error:.3g}.")
    return {"gradient_error": gradient_error, "metric_errors": metric_rows}
