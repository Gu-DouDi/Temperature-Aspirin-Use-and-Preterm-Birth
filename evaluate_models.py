"""Evaluate the fitted single model on the temporal test cohort, without refitting."""

import os

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import json

import numpy as np
import pandas as pd
from numba import njit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
import train_models as engine


@njit(nogil=True)
def weighted_auc_draws(y, order, starts, weights):
    result = np.empty(len(weights))
    for draw in range(len(weights)):
        cumulative_neg = 0.0
        positive_total = 0.0
        area = 0.0
        for group in range(len(starts)):
            start = starts[group]
            stop = starts[group + 1] if group + 1 < len(starts) else len(order)
            pg = 0.0
            ng = 0.0
            for j in range(start, stop):
                index = order[j]
                if y[index] == 1:
                    pg += weights[draw, index]
                else:
                    ng += weights[draw, index]
            area += pg * (cumulative_neg + ng * 0.5)
            cumulative_neg += ng
            positive_total += pg
        result[draw] = area / (positive_total * cumulative_neg)
    return result


def bootstrap_auc(y, p, weights):
    """Weighted AUROC with exact score-tie handling for shared bootstrap draws."""
    order = np.argsort(p, kind="stable")
    s = np.asarray(p)[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1]
    return weighted_auc_draws(np.asarray(y), order, starts, weights)


def performance(y, probability, groups, seed, draws=2000):
    """AUROC CI uses a mother bootstrap conditional on this fitted model."""
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(unique), np.full(len(unique), 1 / len(unique)), size=draws
    )[:, inverse]
    valid = (weights[:, y == 1].sum(axis=1) > 0) & (weights[:, y == 0].sum(axis=1) > 0)
    weights = weights[valid]
    scores = bootstrap_auc(y, probability, weights)
    lower, upper = np.quantile(scores, [0.025, 0.975])
    return dict(
        test_n=len(y),
        test_PTB=int(y.sum()),
        AUROC=float(roc_auc_score(y, probability)),
        AUROC_CI95_lower=float(lower),
        AUROC_CI95_upper=float(upper),
        AUPRC=float(average_precision_score(y, probability)),
        Brier=float(brier_score_loss(y, probability)),
        log_loss=float(log_loss(y, probability)),
        bootstrap_draws=draws,
        bootstrap_valid_draws=int(valid.sum()),
    )


def main(data, bundle):
    # 1. 使用内存中的训练模型，在测试集计算性能。
    y, groups = data["ytest"], data["gtest"]
    probability = engine.predict(bundle, data["Xtest"])
    result = dict(
        model=bundle["config"]["family"],
        **performance(y, probability, groups, bundle["config"]["seed"]),
    )

    print(json.dumps(result, indent=2), flush=True)
    return {"Performance": pd.DataFrame([result])}
