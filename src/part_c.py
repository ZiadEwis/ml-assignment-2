"""Part C (Tasks 11–16): logistic regression. Owner: Member 2."""

import numpy as np
import matplotlib.pyplot as plt

from assignment2_core import (
    accuracy,
    batch_gd,
    class_metrics,
    grad_check,
    logistic_grad,
    logistic_loss,
    roc_curve_from_scratch,
    sigmoid,
)
from plot_utils import savefig


def part_c(d, out):
    Xtr, Xva, Xte = d["X"]
    ytr, yva, yte = d["y"]
    rng = np.random.default_rng(42)
    assert (
        grad_check(
            lambda w: logistic_loss(w, Xtr, ytr, 0.1),
            lambda w: logistic_grad(w, Xtr, ytr, 0.1),
            rng.normal(0, 0.1, Xtr.shape[1]),
        )
        < 1e-06
    )
    records = []
    for lam in [0, 0.001, 0.01, 0.1, 1]:
        w, h = batch_gd(
            lambda q: (
                logistic_loss(q, Xtr, ytr, lam),
                logistic_grad(q, Xtr, ytr, lam),
            ),
            np.zeros(Xtr.shape[1]),
            3000,
            0.5,
        )
        records.append(
            (
                lam,
                accuracy(ytr, sigmoid(Xtr @ w) >= 0.5),
                accuracy(yva, sigmoid(Xva @ w) >= 0.5),
                logistic_loss(w, Xva, yva, lam),
                w,
            )
        )
    best = min(records, key=lambda r: r[3])
    w = best[-1]
    print(
        "C selected lambda/test accuracy",
        best[0],
        accuracy(yte, sigmoid(Xte @ w) >= 0.5),
    )
    fpr, tpr, th, auc = roc_curve_from_scratch(yva, sigmoid(Xva @ w))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("Task 15 validation ROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    savefig(out, "c_roc")
    candidates = th[tpr == 1]
    tau = candidates.max() if len(candidates) else 0.5
    for label, t in [("0.5", 0.5), ("validated", tau)]:
        print("C threshold", label, class_metrics(yte, sigmoid(Xte @ w) >= t, 2))
    for z in [-800.0, 800.0]:
        print(
            "C stable sigmoid/loss at z=",
            z,
            sigmoid(np.array([z]))[0],
            np.logaddexp(0, z),
        )
    for eta in [0.01, 0.1, 1.0, 5.0, 20.0]:
        _, h = batch_gd(
            lambda q: (
                logistic_loss(q, Xtr, ytr, 0.01),
                logistic_grad(q, Xtr, ytr, 0.01),
            ),
            np.zeros(Xtr.shape[1]),
            1000,
            eta,
        )
        print("C learning-rate", eta, "final loss", h[-1])
    w0, h0 = batch_gd(
        lambda q: (logistic_loss(q, Xtr, ytr), logistic_grad(q, Xtr, ytr)),
        np.zeros(Xtr.shape[1]),
        20000,
        0.5,
    )
    print("C separability final norm/loss:", np.linalg.norm(w0), h0[-1])
    plt.semilogy(h0)
    plt.title("Task 13: unregularized logistic loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    savefig(out, "c_separability")


def library_comparisons(P, out):
    """Current Task 14 library comparisons."""
    from sklearn.linear_model import LogisticRegression

    db = P["breast_cancer"]
    Xtr, Xva, Xte = db["X"]
    ytr, yva, yte = db["y"]
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = LogisticRegression(C=C, penalty="l2", tol=1e-12, max_iter=100000).fit(
            Xtr, ytr
        )
        print(
            "Task 14 LogisticRegression",
            C,
            accuracy(ytr, model.predict(Xtr)),
            accuracy(yva, model.predict(Xva)),
            accuracy(yte, model.predict(Xte)),
        )
    for solver in ["lbfgs", "liblinear", "saga"]:
        model = LogisticRegression(
            C=1.0, solver=solver, max_iter=100000, tol=1e-12
        ).fit(Xtr, ytr)
        print(
            "Task 14 solver",
            solver,
            "iterations",
            model.n_iter_,
            "test",
            accuracy(yte, model.predict(Xte)),
        )
