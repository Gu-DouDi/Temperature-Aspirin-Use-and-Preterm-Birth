"""Recompute exact ExtraTrees SHAP and temperature–aspirin pair contributions."""

import os

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import numpy as np
import pandas as pd
import joblib
import shap
from numba import njit
import train_models as engine



# 精确条件TreeSHAP改编自SHAP 0.52的pytree.py（MIT）。
# 原始许可证见同目录SHAP_LICENSE.txt。
@njit(nogil=True)
def aspirin_pairs(
    children_left,
    children_right,
    children_default,
    features,
    thresholds,
    values,
    node_sample_weight,
    max_depth,
    X,
    aspirin_index,
    temp_indices,
):
    """Two symmetric SHAP interaction terms, conditioning aspirin on/off.

    This computes exactly one row of the TreeSHAP interaction matrix and
    avoids calculating unrelated feature pairs. Values already include the
    ensemble tree weights supplied by SHAP's TreeEnsemble.
    """
    result = np.zeros((len(X), len(temp_indices)))
    n_features = X.shape[1]
    n_outputs = values.shape[2]
    scratch = (max_depth + 2) * (max_depth + 3) // 2
    relevant = np.zeros(len(features), dtype=np.bool_)
    for t in range(len(features)):
        for j in range(features.shape[1]):
            if features[t, j] == aspirin_index:
                relevant[t] = True
                break
    for i in range(len(X)):
        missing = np.isnan(X[i])
        on = np.zeros((n_features + 1, n_outputs))
        off = np.zeros((n_features + 1, n_outputs))
        for t in range(len(features)):
            if not relevant[t]:
                continue
            for condition in [1, -1]:
                indexes = np.full(scratch, -1, dtype=np.int32)
                zero = np.zeros(scratch)
                one = np.zeros(scratch)
                weights = np.zeros(scratch)
                phi = on if condition == 1 else off
                tree_shap_recursive(
                    children_left[t],
                    children_right[t],
                    children_default[t],
                    features[t],
                    thresholds[t],
                    values[t],
                    node_sample_weight[t],
                    X[i],
                    missing,
                    phi,
                    0,
                    0,
                    indexes,
                    zero,
                    one,
                    weights,
                    1.0,
                    1.0,
                    -1,
                    condition,
                    aspirin_index,
                    1.0,
                )
        for j in range(len(temp_indices)):
            result[i, j] = on[temp_indices[j], 1] - off[temp_indices[j], 1]
    return result


@njit(nogil=True)
def extend_path(
    feature_indexes,
    zero_fractions,
    one_fractions,
    pweights,
    unique_depth,
    zero_fraction,
    one_fraction,
    feature_index,
):
    feature_indexes[unique_depth] = feature_index
    zero_fractions[unique_depth] = zero_fraction
    one_fractions[unique_depth] = one_fraction
    if unique_depth == 0:
        pweights[unique_depth] = 1.0
    else:
        pweights[unique_depth] = 0.0

    for i in range(unique_depth - 1, -1, -1):
        pweights[i + 1] += one_fraction * pweights[i] * (i + 1.0) / (unique_depth + 1.0)
        pweights[i] = (
            zero_fraction * pweights[i] * (unique_depth - i) / (unique_depth + 1.0)
        )


@njit(nogil=True)
def unwind_path(
    feature_indexes, zero_fractions, one_fractions, pweights, unique_depth, path_index
):
    one_fraction = one_fractions[path_index]
    zero_fraction = zero_fractions[path_index]
    next_one_portion = pweights[unique_depth]

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.0:
            tmp = pweights[i]
            pweights[i] = (
                next_one_portion * (unique_depth + 1.0) / ((i + 1.0) * one_fraction)
            )
            next_one_portion = tmp - pweights[i] * zero_fraction * (
                unique_depth - i
            ) / (unique_depth + 1.0)
        else:
            pweights[i] = (pweights[i] * (unique_depth + 1)) / (
                zero_fraction * (unique_depth - i)
            )

    for i in range(path_index, unique_depth):
        feature_indexes[i] = feature_indexes[i + 1]
        zero_fractions[i] = zero_fractions[i + 1]
        one_fractions[i] = one_fractions[i + 1]


@njit(nogil=True)
def unwound_path_sum(
    feature_indexes, zero_fractions, one_fractions, pweights, unique_depth, path_index
):
    one_fraction = one_fractions[path_index]
    zero_fraction = zero_fractions[path_index]
    next_one_portion = pweights[unique_depth]
    total = 0

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.0:
            tmp = next_one_portion * (unique_depth + 1.0) / ((i + 1.0) * one_fraction)
            total += tmp
            next_one_portion = pweights[i] - tmp * zero_fraction * (
                (unique_depth - i) / (unique_depth + 1.0)
            )
        else:
            total += (pweights[i] / zero_fraction) / (
                (unique_depth - i) / (unique_depth + 1.0)
            )

    return total


@njit(nogil=True)
def tree_shap_recursive(
    children_left,
    children_right,
    children_default,
    features,
    thresholds,
    values,
    node_sample_weight,
    x,
    x_missing,
    phi,
    node_index,
    unique_depth,
    parent_feature_indexes,
    parent_zero_fractions,
    parent_one_fractions,
    parent_pweights,
    parent_zero_fraction,
    parent_one_fraction,
    parent_feature_index,
    condition,
    condition_feature,
    condition_fraction,
):
    # stop if we have no weight coming down to us
    if condition_fraction == 0.0:
        return

    # extend the unique path
    feature_indexes = parent_feature_indexes[unique_depth + 1 :]
    feature_indexes[: unique_depth + 1] = parent_feature_indexes[: unique_depth + 1]
    zero_fractions = parent_zero_fractions[unique_depth + 1 :]
    zero_fractions[: unique_depth + 1] = parent_zero_fractions[: unique_depth + 1]
    one_fractions = parent_one_fractions[unique_depth + 1 :]
    one_fractions[: unique_depth + 1] = parent_one_fractions[: unique_depth + 1]
    pweights = parent_pweights[unique_depth + 1 :]
    pweights[: unique_depth + 1] = parent_pweights[: unique_depth + 1]

    if condition == 0 or condition_feature != parent_feature_index:
        extend_path(
            feature_indexes,
            zero_fractions,
            one_fractions,
            pweights,
            unique_depth,
            parent_zero_fraction,
            parent_one_fraction,
            parent_feature_index,
        )

    split_index = features[node_index]

    # leaf node
    if children_right[node_index] == -1:
        for i in range(1, unique_depth + 1):
            w = unwound_path_sum(
                feature_indexes,
                zero_fractions,
                one_fractions,
                pweights,
                unique_depth,
                i,
            )
            phi[feature_indexes[i], :] += (
                w
                * (one_fractions[i] - zero_fractions[i])
                * values[node_index, :]
                * condition_fraction
            )

    # internal node
    else:
        # find which branch is "hot" (meaning x would follow it)
        hot_index = 0
        cleft = children_left[node_index]
        cright = children_right[node_index]
        if x_missing[split_index] == 1:
            hot_index = children_default[node_index]
        elif x[split_index] < thresholds[node_index]:
            hot_index = cleft
        else:
            hot_index = cright
        cold_index = cright if hot_index == cleft else cleft
        w = node_sample_weight[node_index]
        hot_zero_fraction = node_sample_weight[hot_index] / w
        cold_zero_fraction = node_sample_weight[cold_index] / w
        incoming_zero_fraction = 1.0
        incoming_one_fraction = 1.0

        # see if we have already split on this feature,
        # if so we undo that split so we can redo it for this node
        path_index = 0
        while path_index <= unique_depth:
            if feature_indexes[path_index] == split_index:
                break
            path_index += 1

        if path_index != unique_depth + 1:
            incoming_zero_fraction = zero_fractions[path_index]
            incoming_one_fraction = one_fractions[path_index]
            unwind_path(
                feature_indexes,
                zero_fractions,
                one_fractions,
                pweights,
                unique_depth,
                path_index,
            )
            unique_depth -= 1

        # divide up the condition_fraction among the recursive calls
        hot_condition_fraction = condition_fraction
        cold_condition_fraction = condition_fraction
        if condition > 0 and split_index == condition_feature:
            cold_condition_fraction = 0.0
            unique_depth -= 1
        elif condition < 0 and split_index == condition_feature:
            hot_condition_fraction *= hot_zero_fraction
            cold_condition_fraction *= cold_zero_fraction
            unique_depth -= 1

        tree_shap_recursive(
            children_left,
            children_right,
            children_default,
            features,
            thresholds,
            values,
            node_sample_weight,
            x,
            x_missing,
            phi,
            hot_index,
            unique_depth + 1,
            feature_indexes,
            zero_fractions,
            one_fractions,
            pweights,
            hot_zero_fraction * incoming_zero_fraction,
            incoming_one_fraction,
            split_index,
            condition,
            condition_feature,
            hot_condition_fraction,
        )

        tree_shap_recursive(
            children_left,
            children_right,
            children_default,
            features,
            thresholds,
            values,
            node_sample_weight,
            x,
            x_missing,
            phi,
            cold_index,
            unique_depth + 1,
            feature_indexes,
            zero_fractions,
            one_fractions,
            pweights,
            cold_zero_fraction * incoming_zero_fraction,
            0,
            split_index,
            condition,
            condition_feature,
            cold_condition_fraction,
        )

def main(data, bundle, jobs):

    # 1. 使用与评估相同的内存模型，将测试特征转换到模型输入空间。
    X = data["Xtest"]
    D = data["Xdev"]
    g = data["gtest"]
    a, _, orig = bundle["prep"].transform_all(X)
    a = np.ascontiguousarray(a)
    orig = np.asarray(orig)
    ai = list(orig).index("Aspirin")
    tis = np.array([list(orig).index(t) for t in engine.TEMPS])
    explainer = shap.TreeExplainer(
        bundle["model"], feature_perturbation="tree_path_dependent", model_output="raw"
    )
    p = bundle["model"].predict_proba(a)[:, 1]
    params = (
        explainer.model.children_left,
        explainer.model.children_right,
        explainer.model.children_default,
        explainer.model.features,
        explainer.model.thresholds,
        explainer.model.values,
        explainer.model.node_sample_weight,
        explainer.model.max_depth,
    )

    # 2. 分批计算全局SHAP贡献。
    values = []
    for start in range(0, len(a), 256):
        values.append(explainer.shap_values(a[start : start + 256])[..., 1])
        print(f"Global exact TreeSHAP: {min(start+256,len(a))}/{len(a)}", flush=True)
    sv = np.concatenate(values)
    bias = float(np.asarray(explainer.expected_value).ravel()[1])

    # 3. 计算阿司匹林与三个温度输入的完整成对贡献。
    pair = np.empty((len(a), 3))
    batches = [(start, min(start + 64, len(a))) for start in range(0, len(a), 64)]

    def calculate(lo, hi):
        return lo, hi, aspirin_pairs(*params, a[lo:hi], ai, tis)

    results = joblib.Parallel(
        n_jobs=jobs, backend="threading", return_as="generator_unordered"
    )(joblib.delayed(calculate)(lo, hi) for lo, hi in batches)
    completed = 0
    for lo, hi, v in results:
        pair[lo:hi] = v
        completed += hi - lo
        if completed % 256 == 0 or completed == len(a):
            print(f"Exact temperature–aspirin pairs: {completed}/{len(a)}", flush=True)

    # 4. 聚合one-hot贡献，并计算切换阿司匹林输入的预测概率差。
    x0 = X.copy()
    x1 = X.copy()
    x0["Aspirin"] = 0.0
    x1["Aspirin"] = 1.0
    diff = engine.predict(bundle, x1) - engine.predict(bundle, x0)
    roots = list(dict.fromkeys(orig))
    ag = np.column_stack([sv[:, orig == c].sum(axis=1) for c in roots])
    imp = np.abs(ag).mean(axis=0)
    importance = pd.DataFrame(dict(variable=roots, mean_absolute_SHAP=imp)).sort_values(
        "mean_absolute_SHAP", ascending=False
    )
    shap_values = pd.DataFrame(ag, columns=roots)
    shap_values.insert(0, "source_row", data["source_rows_test"])
    shap_values.insert(1, "bias", bias)
    shap_values.insert(2, "predicted_probability", p)
    pairs = pd.DataFrame(pair, columns=engine.TEMPS)
    pairs.insert(0, "source_row", data["source_rows_test"])
    pairs["probability_difference"] = diff

    # 5. 在已拟合模型条件下进行母亲层面bootstrap。
    rng = np.random.default_rng(bundle["config"]["seed"])
    unique, inverse = np.unique(g, return_inverse=True)
    weights = rng.multinomial(
        len(unique), np.full(len(unique), 1 / len(unique)), size=1000
    )[:, inverse]
    groups = X.Aspirin.to_numpy(int)

    def interval(values, ii):
        counts = weights[:, ii].sum(axis=1)
        valid = counts > 0
        boot = (weights[valid][:, ii] @ values[ii]) / counts[valid]
        return float(values[ii].mean()), *map(float, np.quantile(boot, [0.025, 0.975]))

    # 6. 在共同温度支持范围内，汇总三段温度及其合计的分箱SHAP贡献。
    full = "Full lag 0-11 weeks (sum of pairs)"
    panels = [
        (t, X[t].to_numpy(), D[t], pair[:, j]) for j, t in enumerate(engine.TEMPS)
    ]
    panels.append(
        (
            full,
            X[engine.TEMPS].mean(axis=1).to_numpy(),
            D[engine.TEMPS].mean(axis=1),
            pair.sum(axis=1),
        )
    )
    detail = []
    for j, (t, temp, dtemp, values) in enumerate(panels):
        low = max(dtemp[D.Aspirin.eq(0)].min(), dtemp[D.Aspirin.eq(1)].min())
        high = min(dtemp[D.Aspirin.eq(0)].max(), dtemp[D.Aspirin.eq(1)].max())
        edges = np.unique(np.quantile(temp, np.linspace(0, 1, 9)))
        binid = np.searchsorted(edges[1:-1], temp, side="right")
        for k in range(len(edges) - 1):
            ii = (binid == k) & (temp >= low) & (temp <= high)
            n0 = int((ii & (groups == 0)).sum())
            n1 = int((ii & (groups == 1)).sum())
            supported = min(n0, n1) >= 20
            if not supported:
                continue
            for group, sel in [
                (-1, ii),
                (0, ii & (groups == 0)),
                (1, ii & (groups == 1)),
            ]:
                mean, lo, hi = interval(values, sel)
                detail.append(
                    dict(
                        window=t,
                        bin=k + 1,
                        aspirin_group=group,
                        UTCI_K=float(temp[sel].mean() + 273.15),
                        value=mean,
                        CI95_lower=lo,
                        CI95_upper=hi,
                        value_pp=100 * mean,
                        CI95_lower_pp=100 * lo,
                        CI95_upper_pp=100 * hi,
                        n0=n0,
                        n1=n1,
                    )
                )
    return {
        "SHAP_Importance": importance,
        "SHAP_Values": shap_values,
        "SHAP_Pairs": pairs,
        "Pair_Bin_Summary": pd.DataFrame(detail),
    }
