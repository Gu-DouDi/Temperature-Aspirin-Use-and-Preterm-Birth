"""读取Excel数据，准备早产队列并运行主DLNM分析。"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2
from scipy.linalg.blas import dsyrk
from patsy import dmatrix
import statsmodels.api as sm


INPUT_EXCEL = Path(__file__).resolve().parent
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


# 临床字段映射与主模型调整集。
MAP = {
    "gc": "用药_阿司匹林",
    "bmi": "孕前BMI",
    "parity": "产次",
    "gravidity": "孕次",
    "PE": "子痫前期",
    "FGR": "胎儿生长受限",
    "PROM": "胎膜早破",
}
BASE = ["bmi", "parity", "gravidity", "PE", "FGR", "PROM"]


class ChunkLogit(sm.Logit):
    """Same analytic logistic Hessian, summed in bounded memory blocks."""

    def hessian(self, params):
        H = np.zeros((self.exog.shape[1], self.exog.shape[1]))
        for start in range(0, len(self.endog), 8000):
            x = np.ascontiguousarray(self.exog[start : start + 8000])
            probability = expit(x @ params)
            weighted = x * np.sqrt(probability * (1 - probability))[:, None]
            H -= dsyrk(1.0, weighted.T, lower=1)
        return np.tril(H) + np.tril(H, -1).T


def fit_logit(X, y, groups):
    """使用原始设计矩阵和零初值，通过BFGS拟合人周Logistic模型。"""
    iterations = [0]

    def progress(parameters):
        iterations[0] += 1
        if iterations[0] % 25 == 0:
            print("Raw-design BFGS iteration", iterations[0], flush=True)

    model = ChunkLogit(y, X, hasconst=True, check_rank=False)
    return model.fit(
        method="bfgs",
        maxiter=1000,
        disp=False,
        callback=progress,
        cov_type="cluster",
        cov_kwds={"groups": groups},
    )


def contrast(v, beta, cov):
    v = np.atleast_2d(v)
    eta = v @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", v, cov, v))
    return np.column_stack(
        [
            np.exp(eta),
            np.exp(eta - 1.959963984540054 * se),
            np.exp(eta + 1.959963984540054 * se),
        ]
    )


def cumulative_interaction_test(beta, cov, lagbasis):
    """联合检验累计温度基系数为零，不包含阿司匹林主效应。"""
    # 交互系数按lag基分块；沿lag求和后由12维约化为4维。
    n_temperature_basis = len(beta) // lagbasis.shape[1]
    reduction = np.hstack(
        [weight * np.eye(n_temperature_basis) for weight in lagbasis.sum(axis=0)]
    )
    cumulative_beta = reduction @ beta
    cumulative_cov = reduction @ cov @ reduction.T
    statistic = float(
        cumulative_beta @ np.linalg.solve(cumulative_cov, cumulative_beta)
    )
    return (
        statistic,
        n_temperature_basis,
        float(chi2.sf(statistic, n_temperature_basis)),
    )


def temperature_quintile_summary(mean_utci, events, cb_at, ref, beta, cov):
    """按全部入模人周均温分五组，平均恒温历史的log ROR后取指数。

    每个人周等权；相同均温保留在同一组。CI使用平均对比向量与完整交互
    参数协方差计算，不平均逐点CI，也不将人周视为独立参数估计。
    """
    quintiles, edges = pd.qcut(mean_utci, 5, labels=False, retbins=True)
    reference_basis = cb_at(ref)[0]
    mean_contrasts = np.zeros((5, len(beta)))
    rows = []
    for q in range(5):
        selected = quintiles == q
        temperatures = mean_utci[selected]
        # 分块累加，避免额外分配全体人周的累计交叉基矩阵。
        for start in range(0, len(temperatures), 8000):
            mean_contrasts[q] += cb_at(temperatures[start : start + 8000]).sum(axis=0)
        mean_contrasts[q] = mean_contrasts[q] / len(temperatures) - reference_basis
        rows.append(
            {
                "quintile": f"Q{q + 1}",
                "UTCI_lower_K": float(edges[q]),
                "UTCI_upper_K": float(edges[q + 1]),
                "mean_UTCI_K": float(temperatures.mean()),
                "person_weeks": int(selected.sum()),
                "PTB_events": int(events[selected].sum()),
                "reference_K": ref,
            }
        )
    summary = pd.DataFrame(rows)
    summary["log_ROR"] = mean_contrasts @ beta
    summary["SE_log_ROR"] = np.sqrt(
        np.einsum("ij,jk,ik->i", mean_contrasts, cov, mean_contrasts)
    )
    summary[["ROR", "CI_lower", "CI_upper"]] = contrast(mean_contrasts, beta, cov)
    return summary


def run(d, dest, skip_standardization=False):

    # 1. 读取全队列，设置主模型结局和滞后窗口。
    name = "main"
    print("START", name, flush=True)
    beg = time.time()
    outcome = "早产"
    L = 11
    mn = L + 1
    mx = 36
    n_source = len(d)

    # 2. 选取协变量完整的样本，并整理逐孕周温度。
    covs = BASE
    required = [MAP[x] for x in ["gc"] + covs] + ["住院号", "分娩孕周", outcome]
    missing = d[required].isna().any(axis=1)
    n_missing = int(missing.sum())
    d = d.loc[~missing].reset_index(drop=True)
    cols = sorted(
        [
            (int(re.match(r"^孕(\d+)周UTCI$", c).group(1)), c)
            for c in d
            if re.match(r"^孕(\d+)周UTCI$", c)
        ]
    )
    values = d[[c for w, c in cols]].to_numpy(float)
    weekly = {w: i for i, (w, c) in enumerate(cols)}

    # 3. 分别建立温度维度和滞后维度的样条基。
    print(name, "construct temperature basis", flush=True)
    alltemps = values[np.isfinite(values)]
    ref = float(np.median(alltemps))
    # Same basis implementation as source (ordinary cubic B-splines, not natural splines).
    # A degree-3, df-4 B-spline has one internal knot: the source median.
    # Specify exactly those knots to obtain its design information without
    # allocating a redundant million-row temperature design matrix.
    lower, upper = float(alltemps.min()), float(alltemps.max())
    formula = (
        f"bs(x, df=4, degree=3, knots=({ref!r},), "
        f"lower_bound={lower!r}, upper_bound={upper!r})"
    )
    tb = dmatrix(formula, {"x": np.array([ref])})
    tinfo = tb.design_info
    lb = dmatrix(
        f"bs(x, df=3, degree=3, lower_bound=-1, upper_bound={L+1})",
        {"x": np.arange(L + 1, dtype=float)},
    )
    linfo = lb.design_info
    lagbasis = np.asarray(lb)[:, 1:]

    # 4. 按完整滞后历史构建人周风险集，并定位事件所在孕周。
    print(name, "construct risk dataset", flush=True)
    ga = d["分娩孕周"].to_numpy(float)
    delivery = np.floor(ga).astype(int)
    entry = np.full(
        len(d), mn, dtype=int
    )  # All pregnancies: no first-visit restriction.
    # The source temperature column numbered w contains the w-th seven-day
    # block from LMP. At completed week w, column w is the most recently
    # completed exposure block. lag0 is therefore a block before risk-week start.
    pp = []
    ww = []
    temps = []
    dropped = 0
    for w in range(mn, mx + 1):
        idx = np.flatnonzero((entry <= w) & (delivery >= w))
        tv = values[np.ix_(idx, [weekly[w - k] for k in range(L + 1)])]
        valid = np.isfinite(tv).all(axis=1)
        dropped += int((~valid).sum())
        pp.extend(idx[valid])
        ww.extend([w] * int(valid.sum()))
        temps.extend(tv[valid])
    pp = np.asarray(pp, int)
    ww = np.asarray(ww, int)
    temps = np.asarray(temps, float)
    y = ((ww == delivery[pp]) & (d[outcome].to_numpy()[pp] == 1)).astype(float)
    ids = pd.factorize(d["住院号"])[0][pp]
    n_risk = len(y)
    # Retain every risk row and all unrestricted week intercepts. BFGS uses
    # the same raw design as the original program, including zero-event weeks.
    event_weeks = np.unique(ww[y == 1])
    informative = np.isin(ww, event_weeks)
    no_event_weeks = np.unique(ww[~informative]).tolist()
    zero_event_rows = int((~informative).sum())
    print(
        name,
        "risk person-weeks",
        n_risk,
        "events",
        int(y.sum()),
        "zero-event weeks retained",
        no_event_weeks,
        flush=True,
    )

    # 5. 组合温度基与滞后基，构造交叉基和完整设计矩阵。
    n = len(y)
    cb = np.zeros((n, 12))
    for row in range(0, n, 8000):
        for k in range(L + 1):
            t = np.asarray(dmatrix(tinfo, {"x": temps[row : row + 8000, k]}))[:, 1:]
            for j in range(3):
                cb[row : row + 8000, j * 4 : (j + 1) * 4] += t * lagbasis[k, j]
    print(name, "cross-basis complete", flush=True)
    aspirin = d["用药_阿司匹林"].to_numpy(float)[pp]
    covvals = d[[MAP[x] for x in covs]].to_numpy(float)[pp]
    weeklevels = np.unique(ww)
    weekX = ww[:, None] == weeklevels[1:]
    cbnames = [f"cb_l{k}_t{j}" for k in range(3) for j in range(4)]
    names = (
        ["const"]
        + [f"week_{w}" for w in weeklevels[1:]]
        + cbnames
        + ["aspirin"]
        + ["aspirin_x_" + x for x in cbnames]
        + covs
    )
    # Fill contiguous columns instead of repeatedly traversing a large
    # row-major allocation. All design columns remain mathematically identical.
    columns = (
        [np.ones(n)]
        + [weekX[:, j] for j in range(weekX.shape[1])]
        + [cb[:, j] for j in range(12)]
        + [aspirin]
        + [cb[:, j] * aspirin for j in range(12)]
        + [covvals[:, j] for j in range(covvals.shape[1])]
    )
    X = np.empty((n, len(columns)), order="F")
    for j, x in enumerate(columns):
        X[:, j] = x
    del columns, weekX, covvals

    # 6. 使用原始BFGS设置拟合主模型。
    fit = fit_logit(X, y, ids)
    b = np.asarray(fit.params)
    V = np.asarray(fit.cov_params())
    converged = bool(fit.mle_retvals["converged"])
    likelihood_metrics = dict(
        log_likelihood=float(fit.llf), AIC=float(fit.aic), BIC=float(fit.bic)
    )
    del fit
    # 7. 计算回归系数，并检验全部阿司匹林交互项。
    se = np.sqrt(np.diag(V))
    coef = pd.DataFrame(
        {
            "term": names,
            "coefficient": b,
            "SE_cluster": se,
            "z": b / se,
            "P": 2 * __import__("scipy").stats.norm.sf(np.abs(b / se)),
        }
    )
    ti = [names.index(x) for x in cbnames]
    ai = [names.index("aspirin_x_" + x) for x in cbnames]
    ia = names.index("aspirin")
    wstat = float(b[ai] @ np.linalg.solve(V[np.ix_(ai, ai)], b[ai]))
    wp = float(chi2.sf(wstat, len(ai)))
    cumulative_wstat, cumulative_df, cumulative_wp = cumulative_interaction_test(
        b[ai], V[np.ix_(ai, ai)], lagbasis
    )
    interaction_tests = pd.DataFrame(
        [
            {
                "test": "full_temperature_lag_interaction",
                "chi2": wstat,
                "df": len(ai),
                "P": wp,
            },
            {
                "test": "cumulative_temperature_interaction",
                "chi2": cumulative_wstat,
                "df": cumulative_df,
                "P": cumulative_wp,
            },
        ]
    )

    # 8. 构造累计及各lag的温度对比，计算OR、ROR和置信区间。
    def cb_at(t, k=None):
        z = np.asarray(dmatrix(tinfo, {"x": np.atleast_1d(t)}))[:, 1:]
        weights = lagbasis.sum(axis=0) if k is None else lagbasis[k]
        return np.hstack([z * v for v in weights])

    grid = np.unique(
        np.r_[np.linspace(275, 305, 301), ref, np.percentile(alltemps, [5, 95])]
    )
    total = cb_at(grid)
    diff = total - cb_at(ref)
    v = np.zeros((len(grid), len(b)))
    v[:, ai] = diff
    ci = contrast(v, b, V)
    cumulative = pd.DataFrame(
        {"UTCI_K": grid, "ROR": ci[:, 0], "CI_lower": ci[:, 1], "CI_upper": ci[:, 2]}
    )
    absvec = np.zeros_like(v)
    absvec[:, ia] = 1
    absvec[:, ai] = total
    absolute = contrast(absvec, b, V)
    for j, k in enumerate(["aspirin_OR", "aspirin_OR_CI_lower", "aspirin_OR_CI_upper"]):
        cumulative[k] = absolute[:, j]
    # Both aspirin strata: within-stratum temperature ORs vs constant median history.
    curves = []
    for a in [0, 1]:
        v = np.zeros_like(absvec)
        v[:, ti] = diff
        v[:, ai] = a * diff
        cs = contrast(v, b, V)
        curves.append(
            pd.DataFrame(
                {
                    "UTCI_K": grid,
                    "aspirin": a,
                    "OR": cs[:, 0],
                    "CI_lower": cs[:, 1],
                    "CI_upper": cs[:, 2],
                }
            )
        )
    hmgrid = np.unique(
        np.r_[
            np.linspace(np.percentile(alltemps, 10), np.percentile(alltemps, 90), 121),
            ref,
        ]
    )
    heat = []
    slices = []
    t95 = float(np.percentile(alltemps, 95))
    for k in range(L + 1):
        delta = cb_at(hmgrid, k) - cb_at(ref, k)
        for label, inds in [("temperature_OR_nonaspirin", ti), ("interaction_ROR", ai)]:
            v = np.zeros((len(hmgrid), len(b)))
            v[:, inds] = delta
            cs = contrast(v, b, V)
            heat.append(
                pd.DataFrame(
                    {
                        "UTCI_K": hmgrid,
                        "lag_index": k,
                        "estimand": label,
                        "estimate": cs[:, 0],
                        "CI_lower": cs[:, 1],
                        "CI_upper": cs[:, 2],
                    }
                )
            )
        for a in [0, 1]:
            v = np.zeros((1, len(b)))
            v[:, ti] = cb_at(t95, k) - cb_at(ref, k)
            v[:, ai] = a * (cb_at(t95, k) - cb_at(ref, k))
            cs = contrast(v, b, V)[0]
            slices.append(
                {
                    "lag_index": k,
                    "aspirin": a,
                    "UTCI_contrast_K": t95,
                    "reference_K": ref,
                    "OR": cs[0],
                    "CI_lower": cs[1],
                    "CI_upper": cs[2],
                }
            )

    # 9. 汇总累计ROR五分位，并按孕周和温度组划分标准化OR计算单元。
    avg = temps.mean(axis=1)
    quintile_summary = temperature_quintile_summary(
        avg, y, cb_at, ref, b[ai], V[np.ix_(ai, ai)]
    )
    q5, q95 = np.percentile(avg[informative], [5, 95])
    group = np.where(avg <= q5, "Low", np.where(avg >= q95, "High", "Mid"))
    keys = sorted(set(zip(ww, group)))
    lookup = {k: i for i, k in enumerate(keys)}
    cell = np.array([lookup[(w, g)] for w, g in zip(ww, group)])
    counts = np.bincount(cell)

    # 10. 从完整参数协方差抽样，计算标准化用药OR及区间。
    if not skip_standardization:
        rng = np.random.default_rng(20260528)
        draws = rng.multivariate_normal(b, V, size=2000)
        X0 = X
        X0[:, ia] = 0
        X0[:, ai] = 0

        def cell_sums(p, cc):
            p = np.atleast_2d(p)
            return np.column_stack(
                [
                    np.bincount(cc, weights=p[:, j], minlength=len(keys))
                    for j in range(p.shape[1])
                ]
            )

        def probabilities(bb):
            bb = np.atleast_2d(bb)
            s0 = np.zeros((len(keys), bb.shape[1]))
            s1 = np.zeros_like(s0)
            for row in range(0, n, 8000):
                eta = np.ascontiguousarray(X0[row : row + 8000]) @ bb
                delta = np.ascontiguousarray(cb[row : row + 8000]) @ bb[ai] + bb[ia]
                cc = cell[row : row + 8000]
                s0 += cell_sums(expit(eta), cc)
                s1 += cell_sums(expit(eta + delta), cc)
            return s0 / counts[:, None], s1 / counts[:, None]

        def odds(p):
            return p / (1 - p)

        p0, p1 = probabilities(b[:, None])
        point = (odds(p1) / odds(p0))[:, 0]
        sim = np.empty((len(keys), 2000))
        for start in range(0, 2000, 10):
            bb = draws[start : start + 10].T
            m0, m1 = probabilities(bb)
            sim[:, start : start + bb.shape[1]] = odds(m1) / odds(m0)
            if start % 400 == 0:
                print(name, "simulation", start, "/2000", flush=True)
        lo, hi = np.percentile(sim, [2.5, 97.5], axis=1)
        pv = 2 * np.minimum((sim <= 1).mean(axis=1), (sim >= 1).mean(axis=1))
        forest = pd.DataFrame(
            {
                "gestational_week": [k[0] for k in keys],
                "temp_group": [k[1] for k in keys],
                "OR": point,
                "CI_lower": lo,
                "CI_upper": hi,
                "P_simulation": pv,
                "n_person_weeks": counts,
            }
        )
    else:
        forest = pd.DataFrame()

    # 11. 汇总模型信息、样条结点和数值结果。
    event_ids = np.unique(pp[y == 1])
    all_event = int((d[outcome] == 1).sum())
    stats = {
        "model": name,
        "outcome": outcome,
        "lag_max_index": L,
        "lag_positions": L + 1,
        "min_completed_week": mn,
        "max_completed_week": mx,
        "source_participants": n_source,
        "excluded_missing_required": n_missing,
        "eligible_source_participants": len(d),
        "analyzed_participants": len(np.unique(pp)),
        "person_weeks": n_risk,
        "regression_person_weeks": n,
        "zero_event_weeks_profiled": [],
        "zero_event_weeks_retained": no_event_weeks,
        "zero_event_person_weeks": zero_event_rows,
        "events": int(y.sum()),
        "source_outcome_events_after_missing_exclusion": all_event,
        "source_events_not_modeled": all_event - len(event_ids),
        "rows_missing_lag": dropped,
        "delivery_week_rule": (
            "floor only for week index; " "endpoint determined from unrounded GA <37"
        ),
        "entry_rule": "complete-history availability only; no first-visit restriction",
        "covariates": covs,
        "temperature_df": 4,
        "lag_df": 3,
        "crossbasis_dimension": 12,
        "reference_K": ref,
        "group_P5_K": float(q5),
        "group_P95_K": float(q95),
        "joint_interaction_chi2": wstat,
        "joint_interaction_df": 12,
        "joint_interaction_P": wp,
        "cumulative_interaction_chi2": cumulative_wstat,
        "cumulative_interaction_df": cumulative_df,
        "cumulative_interaction_P": cumulative_wp,
        "quintile_definition": (
            "person-week mean UTCI over all modeled risk rows; ties kept together"
        ),
        "quintile_ROR_method": (
            "exp(mean constant-history log ROR); CI from mean contrast covariance"
        ),
        **likelihood_metrics,
        "converged": converged,
        "covariance": "patient-cluster sandwich",
        "runtime_seconds": time.time() - beg,
        "simulations": 0 if skip_standardization else 2000,
        "simulation_seed": 20260528,
    }
    for label, info in [("temperature", tinfo), ("lag", linfo)]:
        tr = next(iter(info.factor_infos.values())).state["transforms"]
        obj = next(iter(tr.values()))
        stats[f"{label}_knots"] = np.asarray(obj._all_knots).tolist()
    # 12. 将全部汇总结果写入一个Excel文件。
    with pd.ExcelWriter(dest / "analysis_results_revised.xlsx") as wr:
        pd.DataFrame(
            {
                "Metric": stats.keys(),
                "Value": [
                    str(x) if isinstance(x, (list, dict)) else x for x in stats.values()
                ],
            }
        ).to_excel(wr, sheet_name="Model_Stats", index=False)
        coef.to_excel(wr, sheet_name="Coefficients", index=False)
        pd.DataFrame(V, index=names, columns=names).rename_axis("term").to_excel(
            wr, sheet_name="Covariance"
        )
        cumulative.to_excel(wr, sheet_name="Cumulative_ROR", index=False)
        interaction_tests.to_excel(wr, sheet_name="Interaction_Tests", index=False)
        quintile_summary.to_excel(wr, sheet_name="Cumulative_ROR_Quintiles", index=False)
        pd.concat(curves).to_excel(wr, sheet_name="Temperature_Curves", index=False)
        pd.concat(heat).to_excel(wr, sheet_name="Temperature_Lag", index=False)
        pd.DataFrame(slices).to_excel(wr, sheet_name="Lag_Slices", index=False)
        forest.to_excel(wr, sheet_name="OR_by_Week_Temp", index=False)
    print(
        "DONE",
        name,
        "n=",
        n,
        "events=",
        int(y.sum()),
        "joint P=",
        wp,
        "cumulative P=",
        cumulative_wp,
        "seconds=",
        round(time.time() - beg),
        flush=True,
    )


def main():
    # 1. 读取Excel路径、输出路径和分析参数。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "excel", nargs="?", type=Path, default=INPUT_EXCEL, help="输入Excel数据文件路径"
    )
    parser.add_argument(
        "output", nargs="?", type=Path, default=OUTPUT_DIR, help="分析结果输出文件夹"
    )
    parser.add_argument("--sheet-name", default=0, help="工作表名称，默认读取第一张表")
    parser.add_argument(
        "--skip-standardization",
        action="store_true",
        help="跳过标准化OR的2000次参数抽样",
    )
    args = parser.parse_args()
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)

    # 2. 读取Excel，按未四舍五入的分娩孕周定义早产。
    d = pd.read_excel(
        args.excel.expanduser(),
        sheet_name=args.sheet_name,
        dtype={"住院号": str},
    ).reset_index(drop=True)
    ga = pd.to_numeric(d["分娩孕周"])
    d["分娩孕周"] = ga
    d["早产"] = ga.lt(37).astype(int)
    # 3. 拟合主DLNM并输出一个结果汇总Excel文件。
    run(d, output, skip_standardization=args.skip_standardization)
    print("结果已保存至：", (output / "analysis_results.xlsx").resolve(), flush=True)


if __name__ == "__main__":
    main()
