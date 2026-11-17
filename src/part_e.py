"""Part E (Tasks 23–28): support vector machines. Owner: Member 3."""

import numpy as np
import matplotlib.pyplot as plt

from assignment2_core import (
    accuracy,
    batch_gd,
    binary_svm_loss,
    binary_svm_subgrad,
    grad_check,
    ww_hinge_grad,
    ww_hinge_loss,
)
from plot_utils import savefig


def part_e(d, out):
    Xtr, Xva, Xte = d["X"]
    ytr, yva, yte = d["y"]
    ytr = 2 * ytr - 1
    yva = 2 * yva - 1
    yte = 2 * yte - 1
    rng = np.random.default_rng(42)
    assert (
        grad_check(
            lambda w: binary_svm_loss(w, Xtr, ytr, 0.01),
            lambda w: binary_svm_subgrad(w, Xtr, ytr, 0.01),
            rng.normal(0, 0.1, Xtr.shape[1]),
            eps=1e-06,
        )
        < 1e-05
    )
    for label, eta in [
        ("constant_0.1", 0.1),
        ("constant_1", 1.0),
        ("inverse_sqrt", lambda t: 1 / np.sqrt(t)),
    ]:
        w, h = batch_gd(
            lambda q: (
                binary_svm_loss(q, Xtr, ytr, 0.01),
                binary_svm_subgrad(q, Xtr, ytr, 0.01),
            ),
            np.zeros(Xtr.shape[1]),
            3000,
            eta,
        )
        plt.semilogy(h, label=label)
        print("E", label, accuracy(yte, np.sign(Xte @ w)))
    plt.legend()
    plt.title("Task 23 SVM objective")
    plt.xlabel("iteration")
    plt.ylabel("objective")
    savefig(out, "e_schedules")


def task_25(P, out, rng=None):
    """Current multiclass hinge gradient check; training is still pending."""
    if rng is None:
        rng = np.random.default_rng(42)
    digits = P["digits"]
    Xtr, Xva, Xte = digits["X"]
    ytr, yva, yte = digits["y"]
    K = 10
    W0 = rng.normal(0, 0.1, size=(Xtr.shape[1], K))
    entries = rng.choice(W0.size, 100, replace=False)
    err = grad_check(
        lambda q: ww_hinge_loss(q.reshape(W0.shape), Xtr, ytr, 0.1),
        lambda q: ww_hinge_grad(q.reshape(W0.shape), Xtr, ytr, 0.1).ravel(),
        W0.ravel(),
        eps=1e-06,
        entries=entries,
    )
    assert err < 1e-05, f"WW hinge gradient check failed: {err}"
    print("Task 25 initial loss:", ww_hinge_loss(W0, Xtr, ytr), "gradient error:", err)


def library_comparisons(P, out):
    """Current Tasks 24, 26 and 27 library comparisons."""
    from sklearn.svm import SVC, LinearSVC

    db = P["breast_cancer"]
    Xtr, Xva, Xte = db["X"]
    ytr, yva, yte = db["y"]
    for kernel in ["linear", "poly", "rbf"]:
        model = SVC(C=1, kernel=kernel, degree=3, coef0=1, gamma="scale").fit(Xtr, ytr)
        print(
            "SVC",
            kernel,
            accuracy(ytr, model.predict(Xtr)),
            accuracy(yva, model.predict(Xva)),
            accuracy(yte, model.predict(Xte)),
            len(model.support_),
        )
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        linear = SVC(C=C, kernel="linear").fit(Xtr, ytr)
        linear_svc = LinearSVC(C=C, loss="hinge", max_iter=100000).fit(Xtr, ytr)
        print(
            "Task 24 C",
            C,
            "SVC/LinearSVC validation",
            accuracy(yva, linear.predict(Xva)),
            accuracy(yva, linear_svc.predict(Xva)),
        )
    X2tr, X2va, X2te = (Xtr[:, [1, 2]], Xva[:, [1, 2]], Xte[:, [1, 2]])
    for gamma in [0.001, 0.01, 0.1, 1.0, 10.0]:
        model = SVC(C=1, kernel="rbf", gamma=gamma).fit(X2tr, ytr)
        print(
            "Task 27 gamma",
            gamma,
            "train/val/support",
            accuracy(ytr, model.predict(X2tr)),
            accuracy(yva, model.predict(X2va)),
            len(model.support_),
        )


def task_28(P, out):
    """Current grid search. Training-only CV correction is tracked in README."""
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    dg = P["digits"]
    Xtr, Xva, Xte = dg["X"]
    ytr, yva, yte = dg["y"]
    grid = GridSearchCV(
        SVC(kernel="rbf"),
        {"C": [0.1, 1, 10, 100], "gamma": [0.0001, 0.001, 0.01, 0.1, 1]},
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
    )
    grid.fit(np.vstack([Xtr, Xva]), np.r_[ytr, yva])
    print("Task 28 best", grid.best_params_, accuracy(yte, grid.predict(Xte)))
