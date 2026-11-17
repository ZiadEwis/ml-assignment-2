"""Part B (Tasks 5–10): linear and ridge regression. Owner: Mostafa."""

import numpy as np
import matplotlib.pyplot as plt

from assignment2_core import (
    Standardizer,
    add_bias,
    accuracy,
    batch_gd,
    grad_check,
    hessian_spectrum,
    linear_grad,
    linear_loss,
    minibatch_gd,
    normal_equation,
    polynomial_degree2,
    r2,
    rmse,
)
from plot_utils import savefig


def part_b(d, out):
    Xtr, Xva, Xte = d["X"]
    ytr, yva, yte = d["y"]
    rng = np.random.default_rng(42)
    w_star = normal_equation(Xtr, ytr)
    print("B closed form:", rmse(yte, Xte @ w_star), r2(yte, Xte @ w_star))
    assert (
        grad_check(
            lambda w: linear_loss(w, Xtr, ytr, 0.1),
            lambda w: linear_grad(w, Xtr, ytr, 0.1),
            rng.normal(0, 0.1, Xtr.shape[1]),
        )
        < 1e-06
    )
    fig, ax = plt.subplots()
    for eta in [0.001, 0.01, 0.1, 0.5, 1.0]:
        w, h = batch_gd(
            lambda q: (linear_loss(q, Xtr, ytr), linear_grad(q, Xtr, ytr)),
            np.zeros(Xtr.shape[1]),
            2000,
            eta,
        )
        ax.semilogy(h, label=f"eta={eta}")
        print(
            "B GD",
            eta,
            linear_loss(w, Xtr, ytr),
            np.linalg.norm(w - w_star) / np.linalg.norm(w_star),
        )
    ax.set(title="Task 6: batch GD loss", xlabel="epoch", ylabel="training objective")
    ax.legend()
    savefig(out, "b_batch_gd")
    for B in [1, 16, 64, len(ytr)]:
        f = lambda w, ix: (
            linear_loss(
                w, Xtr if ix is None else Xtr[ix], ytr if ix is None else ytr[ix]
            ),
            linear_grad(
                w, Xtr if ix is None else Xtr[ix], ytr if ix is None else ytr[ix]
            ),
        )
        _, hist, updates = minibatch_gd(
            f, np.zeros(Xtr.shape[1]), len(ytr), B, 100, 0.01, rng
        )
        plt.plot(hist, label=f"B={B}")
    plt.title("Task 8: mini-batch loss")
    plt.xlabel("epoch")
    plt.ylabel("objective")
    plt.legend()
    savefig(out, "b_minibatch")
    lambdas = np.logspace(-3, 1, 9)
    val = []
    for lam in lambdas:
        w = normal_equation(Xtr, ytr, lam)
        val.append(rmse(yva, Xva @ w))
    plt.semilogx(lambdas, val, marker="o")
    plt.title("Task 9: ridge validation RMSE")
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    savefig(out, "b_ridge")
    cond_scaled, lmax_scaled, _ = hessian_spectrum(Xtr)
    Xraw_train, Xraw_val, _ = d["raw"]
    Xraw_train = add_bias(Xraw_train)
    cond_raw, lmax_raw, _ = hessian_spectrum(Xraw_train)
    for label, X, y, lmax in [
        ("scaled", Xtr, ytr, lmax_scaled),
        ("raw", Xraw_train, ytr, lmax_raw),
    ]:
        for eta in [0.95 * 2 / lmax, 1.05 * 2 / lmax]:
            _, h = batch_gd(
                lambda q: (linear_loss(q, X, y), linear_grad(q, X, y)),
                np.zeros(X.shape[1]),
                2000,
                eta,
            )
            print(
                f"B {label}: eta={eta:.3g}, final J={h[-1]:.5g}, cond={(cond_scaled if label == 'scaled' else cond_raw):.3g}"
            )
    Xp_tr, pairs = polynomial_degree2(d["raw"][0], drop_redundant_sex_square=True)
    Xp_va, _ = polynomial_degree2(d["raw"][1], drop_redundant_sex_square=True)
    Xp_te, _ = polynomial_degree2(d["raw"][2], drop_redundant_sex_square=True)
    poly_scaler = Standardizer().fit(Xp_tr)
    Xp_tr, Xp_va, Xp_te = map(
        add_bias, [poly_scaler.transform(z) for z in (Xp_tr, Xp_va, Xp_te)]
    )
    assert Xp_tr.shape[1] == 65, "bias plus 64 required polynomial features"
    poly_scores = [
        (lam, rmse(yva, Xp_va @ normal_equation(Xp_tr, ytr, lam)))
        for lam in np.r_[0.0, np.logspace(-3, 1, 9)]
    ]
    print(
        "B degree-two best validation lambda:", min(poly_scores, key=lambda row: row[1])
    )
