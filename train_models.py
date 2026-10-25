"""Fit one fixed ExtraTrees configuration once on the full development cohort."""

import os

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260924
TEMPS = ["UTCI lag 0-3 weeks", "UTCI lag 4-7 weeks", "UTCI lag 8-11 weeks"]
CORE = TEMPS + ["Aspirin"]
CATEGORICAL = [
    "Occupation",
    "Marital status",
    "District",
    "高危等级",
    "多胎妊娠合并子痫前期",
]


class Prep:
    """Development-only imputation/encoding/scaling; keep all nonconstant base inputs.

    This is the selected source configuration's basic/all transformation.
    No feature-ranking model or supervised feature selection is fitted.
    """

    def fit(self, X, base_covariates):
        self.candidates = list(base_covariates)
        self.fill, self.enc, self.scaler = {}, {}, {}
        self.active = []
        for column in self.candidates:
            if column in CATEGORICAL:
                values = X[column].astype("string")
                mode = values.mode()
                fill = str(mode.iloc[0])
                values = values.fillna(fill).astype(str)
                self.fill[column] = fill
                self.enc[column] = OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=30,
                    sparse_output=False,
                    drop=None,
                    dtype=np.float32,
                ).fit(values.to_numpy().reshape(-1, 1))
                varies = values.nunique() > 1
            else:
                values = pd.to_numeric(X[column], errors="coerce")
                fill = float(values.median())
                self.fill[column] = fill
                numeric = values.fillna(fill).to_numpy(float)
                self.scaler[column] = StandardScaler().fit(numeric.reshape(-1, 1))
                varies = np.ptp(numeric) > 0
            if varies:
                self.active.append(column)
        self.core_scaler = StandardScaler().fit(X[CORE])
        _, self.names, self.origins = self.transform_all(X)
        return self

    def transform_all(self, X):
        arrays, names, origins = [], [], []
        for column in self.active:
            if column in CATEGORICAL:
                values = (
                    X[column].astype("string").fillna(self.fill[column]).astype(str)
                )
                encoded = self.enc[column].transform(
                    values.to_numpy().reshape(-1, 1)
                )
                encoded_names = [
                    f"{column} category {i+1}" for i in range(encoded.shape[1])
                ]
            else:
                values = pd.to_numeric(X[column], errors="coerce").fillna(
                    self.fill[column]
                )
                encoded = self.scaler[column].transform(
                    values.to_numpy().reshape(-1, 1)
                )
                encoded_names = [column]
            arrays.append(encoded)
            names.extend(encoded_names)
            origins.extend([column] * encoded.shape[1])
        arrays.append(self.core_scaler.transform(X[CORE]))
        names.extend(CORE)
        origins.extend(CORE)
        return np.column_stack(arrays).astype(np.float32), names, origins


def load_config(path):
    return json.loads(Path(path).read_text())


def estimator(config, jobs):
    return ExtraTreesClassifier(
        **config["params"], random_state=config["seed"], n_jobs=jobs
    )


def predict(bundle, X):
    encoded, _, _ = bundle["prep"].transform_all(X)
    return bundle["model"].predict_proba(encoded)[:, 1]


def fit_model(data, config, jobs):
    """Development-only preprocessing followed by one classifier fit."""
    X, y = data["Xdev"], data["ydev"]
    prep = Prep().fit(X, data["base_covariates"])
    encoded, names, origins = prep.transform_all(X)
    model = estimator(config, jobs).fit(encoded, y)
    return dict(
        config=config, prep=prep, model=model, encoded_names=names, origins=origins
    )


def main(data, jobs, config_path):
    # 仅用开发集拟合预处理和一次固定参数ExtraTrees。
    config = load_config(config_path)
    return fit_model(data, config, jobs)
