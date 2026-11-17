"""Part A (Tasks 1–4): data preparation and toolkit checks. Owner: Mostafa."""

import numpy as np
import matplotlib.pyplot as plt

from assignment2_core import (
    Standardizer,
    add_bias,
    class_metrics,
)
from plot_utils import savefig


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


def part_a(P):
    for name, d in P.items():
        ytr, yva, yte = d["y"]
        sizes = tuple(map(len, (ytr, yva, yte)))
        print(f"{name}: train/val/test={sizes}")
        if name != "diabetes":
            print(
                " class counts:",
                [
                    np.bincount(
                        y, minlength=int(max(ytr.max(), yva.max(), yte.max())) + 1
                    ).tolist()
                    for y in d["y"]
                ],
            )
    rng = np.random.default_rng(42)
    y, p = (rng.integers(0, 4, 500), rng.integers(0, 4, 500))
    cm, pr, re, f1 = class_metrics(y, p, 4)
    from sklearn.metrics import confusion_matrix as scm, precision_recall_fscore_support

    assert np.array_equal(cm, scm(y, p, labels=np.arange(4)))
    assert np.allclose(
        np.r_[pr, re, f1],
        np.r_[
            precision_recall_fscore_support(y, p, labels=np.arange(4), zero_division=0)[
                :3
            ]
        ],
    )
    print("Part A metric checks passed.")
