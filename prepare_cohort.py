"""All pregnancies with 12 complete gestational weeks aligned before delivery.

All source records are retained. Missing outcomes are recovered from GA <37;
all outcomes are defined by recorded unrounded delivery GA <37; conflicts corrected.
"""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from train_models import TEMPS, CORE

from analysis_config import BUNDLE

# 从受限临床文本中提取基础特征。
RULES = {
    "Text assisted reproduction": r"辅助生殖|试管|体外受精|胚胎移植",
    "Text cervical insufficiency": r"宫颈(?:机能|功能)不全",
    "Text cervical procedure history": r"宫颈锥切|LEEP|利普",
    "Text uterine anomaly": r"子宫畸形|双角子宫|纵隔子宫|单角子宫",
    "Text cervical cerclage record": r"(?:子宫颈|宫颈)环扎",
    "Text chronic renal disease": r"慢性肾|肾功能不全",
}
FIELDS = (
    ["高危因素", "historyname", "主要诊断"]
    + ["其他诊断" + str(i) for i in range(1, 10)]
    + ["手术名称"]
)


def extract(d):
    features = pd.DataFrame(index=d.index)

    # 1. 按概念与文本片段匹配，区分肯定、否定和不确定记录。
    for name, pattern in RULES.items():
        vals = []
        for _, row in d[FIELDS].iterrows():
            yes = False
            maybe = False
            for c in FIELDS:
                for clause in re.split(r"[,，;；|^。\n]", str(row[c])):
                    for match in re.finditer(pattern, clause, re.I):
                        prefix = clause[max(0, match.start() - 12) : match.start()]
                        if re.search(r"(否认|排除|未发现|未见|无).{0,4}$", prefix):
                            continue
                        if re.search(r"[?？]|怀疑|疑似|待排", clause):
                            maybe = True
                            continue
                        yes = True
            vals.append(1.0 if yes else np.nan if maybe else 0.0)
        features[name] = vals
    vals = []

    # 2. 既往早产只使用病史字段，并要求出现早于本次妊娠的年份。
    for t, date in zip(d.historyname, d["末次月经时间"]):
        yes = False
        amb = False
        for clause in re.split(r"[;；|^。\n]", str(t)):
            for m in re.finditer("早产", clause):
                prefix = clause[: m.start()]
                years = re.findall(r"(?:19|20)\d{2}", prefix)
                if years and int(years[-1]) < date.year:
                    yes = True
                else:
                    amb = True
        vals.append(1.0 if yes else np.nan if amb else 0.0)
    name = "Text documented prior PTB"
    features[name] = vals
    return features


MAPPING = {
    "职业": "Occupation",
    "婚姻": "Marital status",
    "所在区县": "District",
    "产次": "Parity",
    "孕次": "Gravidity",
    "孕前体重": "Prepregnancy weight",
    "身高": "Height",
    "孕前BMI": "Prepregnancy BMI",
    "初经": "Age at menarche",
    "月经周期": "Menstrual cycle length",
    "月经持续时间": "Menstrual duration",
    "手术史": "Surgery history",
    "过敏史": "Allergy history",
    "不良孕产史": "Adverse pregnancy history",
    "自然流产": "Miscarriage history",
    "异位妊娠": "Ectopic pregnancy history",
    "死产": "Stillbirth history",
    "吸烟史": "Smoking history",
    "饮酒史": "Alcohol history",
    "孕前高血压": "Preexisting hypertension",
    "孕前糖尿病": "Preexisting diabetes",
    "孕前甲状腺功能减退": "Preexisting hypothyroidism",
    "孕前甲状腺功能亢进": "Preexisting hyperthyroidism",
    "孕前贫血": "Prepregnancy anemia",
}


def main(excel_path, sheet_name=0):
    # 1. 读取Excel和固定特征配置。
    d = pd.read_excel(
        Path(excel_path), sheet_name=sheet_name, dtype={"住院号": str}
    ).reset_index(drop=True)
    for column in ["末次月经时间", "分娩日期", "出生日期", "初检日期"]:
        d[column] = pd.to_datetime(d[column])
    plan = json.loads((BUNDLE / "config/candidate_plan.json").read_text())
    base_covariates = [
        c for c in plan["base_covariates"] if c not in ["Landmark year", "Season day"]
    ]

    # 2. 计算年龄、母亲分组及分娩前12个完整孕周的温度。
    clinical_anchor = d["末次月经时间"] + pd.Timedelta(days=84)
    age = (clinical_anchor - d["出生日期"]).dt.total_seconds() / 86400 / 365.25
    ga = (d["分娩日期"] - d["末次月经时间"]).dt.total_seconds() / 86400 / 7
    groups = pd.factorize(d["姓名"].astype(str) + "|" + d["出生日期"].astype(str))[0]
    completed_weeks = np.floor(ga).astype(int)
    weekly_cols = [f"孕{i}周UTCI" for i in range(1, 46)]
    source_week_numbers = completed_weeks.to_numpy()[:, None] - np.arange(12)[None, :]
    lag_utci = d[weekly_cols].to_numpy(float)[
        np.arange(len(d))[:, None], source_week_numbers - 1
    ]

    # 3. 用记录孕周定义早产，以2022年1月1日划分开发集和测试集。
    target_y = pd.to_numeric(d["分娩孕周"]).lt(37).astype(int)
    dev = d["分娩日期"].lt(pd.Timestamp("2022-01-01"))
    test = ~dev

    # 4. 整理临床变量，按既定清洗规则将无效值标记为缺失。
    X = d[list(MAPPING)].rename(columns=MAPPING).copy()
    X["Maternal age"] = age.where(np.isfinite(age) & age.gt(0), np.nan)
    X["First visit gestational age"] = (
        (d["初检日期"] - d["末次月经时间"]).dt.total_seconds() / 86400 / 7
    )
    bounds = {
        "Prepregnancy weight": (25, 250),
        "Height": (100, 220),
        "Prepregnancy BMI": (10, 80),
        "Age at menarche": (8, 25),
        "Menstrual cycle length": (15, 120),
        "Menstrual duration": (1, 20),
    }
    for c, (lo, hi) in bounds.items():
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X.loc[~X[c].between(lo, hi), c] = np.nan
    for c in ["Occupation", "Marital status", "District"]:
        X[c] = X[c].replace({"不详": np.nan})
    fv = X["First visit gestational age"]
    X.loc[(fv < 0) | (fv > ga), "First visit gestational age"] = np.nan

    # 5. 提取文本特征，并计算三段温度均值和阿司匹林输入。
    text = extract(d)
    for c in plan["new_covariates"]:
        X[c] = text[c] if c in text else d[c]
    for c, lags in zip(TEMPS, [slice(0, 4), slice(4, 8), slice(8, 12)]):
        X[c] = lag_utci[:, lags].mean(axis=1) - 273.15
    X["Aspirin"] = d["用药_阿司匹林"].astype(float)
    X = X.loc[:, base_covariates + CORE].copy()

    # 6. 返回内存队列，插补、编码和缩放留到训练阶段。
    out = {"base_covariates": base_covariates}
    for scope, mask in [("dev", dev), ("test", test)]:
        out["X" + scope] = X.loc[mask].reset_index(drop=True)
        out["y" + scope] = target_y.loc[mask].to_numpy(int)
        out["g" + scope] = groups[mask]
        out["source_rows_" + scope] = np.flatnonzero(mask)
    return out
