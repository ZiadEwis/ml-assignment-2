"""Part D (Tasks 17–22): softmax classification. Owner: Member 2."""

import numpy as np
import matplotlib.pyplot as plt

from assignment2_core import (
    accuracy,
    class_metrics,
    grad_check,
    minibatch_gd,
    softmax_grad,
    softmax_loss,
)
from plot_utils import savefig


def part_d(d, out):
    Xtr, Xva, Xte = d["X"]
    ytr, yva, yte = d["y"]
    K = 10
    rng = np.random.default_rng(42)
    W0 = rng.normal(0, 0.1, (Xtr.shape[1], K))
    picks = []
    for eta in [0.01, 0.1, 0.5, 1.0]:
        f = lambda W, ix: (
            softmax_loss(
                W, Xtr if ix is None else Xtr[ix], ytr if ix is None else ytr[ix]
            ),
            softmax_grad(
                W, Xtr if ix is None else Xtr[ix], ytr if ix is None else ytr[ix]
            ),
        )
        W, h, _ = minibatch_gd(f, W0, len(ytr), 64, 100, eta, rng)
        picks.append((softmax_loss(W, Xva, yva), eta, W, h))
    _, eta, W, h = min(picks, key=lambda r: r[0])
    print("D chosen eta, test acc:", eta, accuracy(yte, (Xte @ W).argmax(1)))
    cm, pr, re, f1 = class_metrics(yte, (Xte @ W).argmax(1), 10)
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.title("Task 20 test confusion matrix")
    plt.xlabel("predicted")
    plt.ylabel("true")
    savefig(out, "d_confusion")
    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    for k, ax in enumerate(axs.ravel()):
        ax.imshow(W[1:, k].reshape(8, 8), cmap="coolwarm")
        ax.set_title(str(k))
        ax.axis("off")
    savefig(out, "d_weights")
    entries = rng.choice(W0.size, size=100, replace=False)
    err = grad_check(
        lambda q: softmax_loss(q.reshape(W0.shape), Xtr, ytr, 0.1),
        lambda q: softmax_grad(q.reshape(W0.shape), Xtr, ytr, 0.1).ravel(),
        W0.ravel(),
        entries=entries,
    )
    assert err < 1e-06, f"softmax gradient check failed: {err}"
    candidates = []
    for lam in [0.0, 0.0001, 0.001, 0.01, 0.1]:
        f = lambda W, ix: (
            softmax_loss(
                W, Xtr if ix is None else Xtr[ix], ytr if ix is None else ytr[ix], lam
            ),
            softmax_grad(
                W, Xtr if ix is None else Xtr[ix], ytr if ix is None else ytr[ix], lam
            ),
        )
        W_lam, _, _ = minibatch_gd(f, W0, len(ytr), 64, 100, eta, rng)
        candidates.append((accuracy(yva, (Xva @ W_lam).argmax(1)), lam, W_lam))
    best_acc, best_lam, best_W = max(candidates, key=lambda row: (row[0], row[1]))
    print(
        "D chosen lambda/test accuracy:",
        best_lam,
        accuracy(yte, (Xte @ best_W).argmax(1)),
    )
    wrong = np.flatnonzero((Xte @ best_W).argmax(1) != yte)[:6]
    fig, axs = plt.subplots(2, 3, figsize=(7, 5))
    for ax, idx in zip(axs.ravel(), wrong):
        ax.imshow(Xte[idx, 1:].reshape(8, 8), cmap="gray")
        ax.set_title(f"true={yte[idx]}, pred={(Xte[idx] @ best_W).argmax()}")
        ax.axis("off")
    savefig(out, "d_misclassified")


def task_22(P, out, rng=None):
    """Current Iris experiment. See the README for outstanding Task 22 requirements."""
    iris = P["iris"]
    Xtr, Xva, Xte = iris["X"]
    ytr, yva, yte = iris["y"]
    if rng is None:
        rng = np.random.default_rng(42)
    Xtr2, Xva2, Xte2 = (Xtr[:, [0, 3]], Xva[:, [0, 3]], Xte[:, [0, 3]])
    W0 = np.zeros((Xtr2.shape[1], 3))
    choices = []
    for eta in [0.1, 0.5, 1.0]:
        f = lambda W, ix: (
            softmax_loss(
                W, Xtr2 if ix is None else Xtr2[ix], ytr if ix is None else ytr[ix]
            ),
            softmax_grad(
                W, Xtr2 if ix is None else Xtr2[ix], ytr if ix is None else ytr[ix]
            ),
        )
        W, _, _ = minibatch_gd(f, W0, len(ytr), len(ytr), 2000, eta, rng)
        choices.append((accuracy(yva, (Xva2 @ W).argmax(1)), eta, W))
    _, eta, W = max(choices, key=lambda row: row[0])
    print("Task 22 softmax iris accuracy:", accuracy(yte, (Xte2 @ W).argmax(1)))


def library_comparisons(P, out):
    """Current Task 19 library comparisons."""
    from sklearn.linear_model import LogisticRegression

    dg = P["digits"]
    Xtr, Xva, Xte = dg["X"]
    ytr, yva, yte = dg["y"]
    for C in [0.1, 1.0, 10.0, 100.0]:
        model = LogisticRegression(C=C, max_iter=10000).fit(Xtr, ytr)
        print(
            "Task 19 C",
            C,
            "validation/test",
            accuracy(yva, model.predict(Xva)),
            accuracy(yte, model.predict(Xte)),
        )
