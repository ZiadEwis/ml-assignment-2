"""From-scratch utilities for Assignment 2.

The functions deliberately use NumPy arrays rather than sklearn estimators so
that every loss, gradient, and optimizer required by the assignment is visible.
"""

from __future__ import annotations

import numpy as np


class Standardizer:
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def add_bias(X):
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(X.shape[0]), X])


def mse(y, pred):
    return float(np.mean((np.asarray(y) - np.asarray(pred)) ** 2))


def rmse(y, pred):
    return float(np.sqrt(mse(y, pred)))


def r2(y, pred):
    y = np.asarray(y)
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def accuracy(y, pred):
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def confusion_matrix(y, pred, K):
    cm = np.zeros((K, K), dtype=int)
    np.add.at(cm, (np.asarray(y, int), np.asarray(pred, int)), 1)
    return cm


def class_metrics(y, pred, K):
    cm = confusion_matrix(y, pred, K)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(K), where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(K), where=cm.sum(1) != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(K),
        where=(precision + recall) != 0,
    )
    return cm, precision, recall, f1


def grad_check(f, grad, w, eps=1e-5, entries=None, rng=None):
    """Relative error for central finite differences; optionally sample entries."""
    w = np.asarray(w, dtype=float).copy()
    analytic = grad(w).ravel()
    idx = np.arange(w.size) if entries is None else np.asarray(entries)
    numeric = np.empty(len(idx))
    flat = w.ravel()
    for j, i in enumerate(idx):
        old = flat[i]
        flat[i] = old + eps
        plus = f(w)
        flat[i] = old - eps
        minus = f(w)
        flat[i] = old
        numeric[j] = (plus - minus) / (2 * eps)
    a = analytic[idx]
    return np.linalg.norm(a - numeric) / max(
        np.linalg.norm(a), np.linalg.norm(numeric), 1e-12
    )


def ridge_mask(w):
    mask = np.ones_like(w, dtype=float)
    mask[0] = 0.0
    return mask


def linear_loss(w, X, y, lam=0.0):
    err = X @ w - y
    return 0.5 * np.mean(err**2) + 0.5 * lam * np.sum(w[1:] ** 2)


def linear_grad(w, X, y, lam=0.0):
    return X.T @ (X @ w - y) / len(y) + lam * w * ridge_mask(w)


def normal_equation(X, y, lam=0.0):
    I = np.eye(X.shape[1])
    I[0, 0] = 0
    return np.linalg.solve(X.T @ X / len(y) + lam * I, X.T @ y / len(y))


def batch_gd(loss_grad, w0, epochs, eta):
    w = np.asarray(w0, dtype=float).copy()
    history = []
    for t in range(epochs):
        loss, grad = loss_grad(w)
        history.append(loss)
        w -= (eta(t + 1) if callable(eta) else eta) * grad
    return w, np.asarray(history)


def minibatch_gd(loss_grad, w0, n, batch_size, epochs, eta, rng):
    w = np.asarray(w0, dtype=float).copy()
    history = []
    updates = []
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            ix = order[start : start + batch_size]
            loss, grad = loss_grad(w, ix)
            w -= eta * grad
            updates.append(loss)
        history.append(loss_grad(w, None)[0])
    return w, np.asarray(history), np.asarray(updates)


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z, dtype=float)
    positive = z >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)
    return result


def logistic_loss(w, X, y, lam=0.0):
    z = X @ w
    return float(np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * np.sum(w[1:] ** 2))


def logistic_grad(w, X, y, lam=0.0):
    return X.T @ (sigmoid(X @ w) - y) / len(y) + lam * w * ridge_mask(w)


def softmax(scores):
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def softmax_loss(W, X, y, lam=0.0):
    scores = X @ W
    shifted = scores - scores.max(1, keepdims=True)
    data = np.mean(np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(y)), y])
    return float(data + 0.5 * lam * np.sum(W[1:] ** 2))


def softmax_grad(W, X, y, lam=0.0):
    P = softmax(X @ W)
    P[np.arange(len(y)), y] -= 1
    G = X.T @ P / len(y)
    G[1:] += lam * W[1:]
    return G


def binary_svm_loss(w, X, y, lam=0.0):
    margins = 1 - y * (X @ w)
    return float(np.mean(np.maximum(0, margins)) + 0.5 * lam * np.sum(w[1:] ** 2))


def binary_svm_subgrad(w, X, y, lam=0.0):
    active = y * (X @ w) < 1
    g = -(X[active].T @ y[active]) / len(y)
    g[1:] += lam * w[1:]
    return g


def ww_hinge_loss(W, X, y, lam=0.0, delta=1.0):
    S = X @ W
    correct = S[np.arange(len(y)), y, None]
    margins = np.maximum(0, S - correct + delta)
    margins[np.arange(len(y)), y] = 0
    return float(margins.sum() / len(y) + 0.5 * lam * np.sum(W[1:] ** 2))


def ww_hinge_grad(W, X, y, lam=0.0, delta=1.0):
    S = X @ W
    correct = S[np.arange(len(y)), y, None]
    B = ((S - correct + delta) > 0).astype(float)
    B[np.arange(len(y)), y] = 0
    B[np.arange(len(y)), y] = -B.sum(1)
    G = X.T @ B / len(y)
    G[1:] += lam * W[1:]
    return G


def polynomial_degree2(X, drop_redundant_sex_square=False, sex_index=1):
    """Return original features followed by every x_i*x_j for i <= j.

    Diabetes encodes sex as 1/2, so its square is a linear combination of sex
    and the bias. Setting ``drop_redundant_sex_square=True`` removes only that
    column and leaves the assignment's required 64 degree-two features.
    """
    X = np.asarray(X, dtype=float)
    cols = [X[:, j] for j in range(X.shape[1])]
    pairs = []
    for i in range(X.shape[1]):
        for j in range(i, X.shape[1]):
            if drop_redundant_sex_square and i == j == sex_index:
                continue
            cols.append(X[:, i] * X[:, j])
            pairs.append((i, j))
    return np.column_stack(cols), pairs


def standardization_checks(X_train, X_validation, X_test, atol=1e-9):
    """Confirm only training columns are exactly standardized."""
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    if not np.allclose(train_mean, 0.0, atol=atol):
        raise AssertionError("standardized training means are not zero")
    if not np.allclose(train_std, 1.0, atol=atol):
        raise AssertionError("standardized training standard deviations are not one")
    return {
        "train_mean_max_abs": float(np.abs(train_mean).max()),
        "train_std_max_abs_error": float(np.abs(train_std - 1).max()),
        "validation_mean": np.mean(X_validation, axis=0),
        "test_mean": np.mean(X_test, axis=0),
    }


def hessian_spectrum(X):
    """Condition number and largest eigenvalue of X^T X / N."""
    H = X.T @ X / len(X)
    eigenvalues = np.linalg.eigvalsh(H)
    return float(np.linalg.cond(X.T @ X)), float(eigenvalues[-1]), eigenvalues


def roc_curve_from_scratch(y, scores):
    """Threshold distinct scores in descending order and integrate trapezoids."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.r_[np.inf, np.unique(scores)[::-1], -np.inf]
    tpr, fpr = [], []
    positives, negatives = (y == 1).sum(), (y == 0).sum()
    for t in thresholds:
        pred = scores >= t
        tpr.append(((pred & (y == 1)).sum() / positives) if positives else 0.0)
        fpr.append(((pred & (y == 0)).sum() / negatives) if negatives else 0.0)
    tpr, fpr = np.asarray(tpr), np.asarray(fpr)
    return fpr, tpr, thresholds, float(np.trapezoid(tpr, fpr))


def precision_recall_from_scratch(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.r_[np.inf, np.unique(scores)[::-1]]
    precision, recall = [], []
    positives = (y == 1).sum()
    for t in thresholds:
        pred = scores >= t
        tp = (pred & (y == 1)).sum()
        fp = (pred & (y == 0)).sum()
        precision.append(tp / (tp + fp) if tp + fp else 1.0)
        recall.append(tp / positives if positives else 0.0)
    return np.asarray(precision), np.asarray(recall), thresholds
