from __future__ import annotations

import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import sys
import math
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Local data are read from datasets/; outputs are written under results_applied_intelligence/.
DATASET_ROOT = BASE_DIR / "datasets"
RAW_DATA_DIR = DATASET_ROOT / "raw"
PROCESSED_DATA_DIR = DATASET_ROOT / "processed"
OULAD_DIR = RAW_DATA_DIR / "oulad"
DROPOUT_DIR = RAW_DATA_DIR / "uci_dropout_success"

RESULT_DIR = BASE_DIR / "results_applied_intelligence"
CSV_DIR = RESULT_DIR / "csv"
FIG_DIR = RESULT_DIR / "figures"
REPORT_DIR = RESULT_DIR / "reports"

# Alias used by the OULAD loaders.
DATA_DIR = OULAD_DIR


REQUIRED_CSVS = [
    "studentInfo.csv",
    "studentVle.csv",
    "studentAssessment.csv",
    "assessments.csv",
    "vle.csv",
]

# Early prediction windows, measured in days after course start.
TIME_WINDOWS = [14, 28, 56, 84]
INTERVENTION_WINDOW = 28
TEST_SIZE = 0.20

# Set FAST_MODE=True for a quick local check.
FAST_MODE = False
FAST_MODE_SAMPLE_SIZE = 12000

# Number of test cases used for intervention generation.
MAX_INTERVENTION_CASES = 90

# Seeds used for the robustness check.
RUN_ROBUSTNESS_CHECK = True
ROBUSTNESS_SEEDS = [13, 42, 71]
ROBUSTNESS_MODELS = ["LogisticRegression", "ExtraTrees", "CausalTwin-Edu"]

# Bootstrap resamples for intervention summaries.
BOOTSTRAP_N = 300


# -----------------------------------------------------------------------------
# 2. Local dataset validation
# -----------------------------------------------------------------------------


def find_csv_path(filename: str) -> Optional[Path]:
    """Find an OULAD CSV under the configured data directory."""
    matches = list(DATA_DIR.rglob(filename))
    return matches[0] if matches else None


def require_csv_path(filename: str) -> Path:
    """Return a required OULAD CSV path."""
    path = find_csv_path(filename)
    if path is None:
        raise FileNotFoundError(
            f"Required OULAD file not found: {filename}\n"
            f"Expected under: {DATA_DIR.resolve()}\n"
            "Please place the already-downloaded OULAD CSV files in this folder. "
            "This script reads local datasets only and does not perform downloads."
        )
    return path


def dataset_ready() -> bool:
    return all(find_csv_path(name) is not None for name in REQUIRED_CSVS)


def validate_oulad_dataset() -> None:
    """Check that the required local OULAD files exist."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"OULAD directory does not exist: {DATA_DIR.resolve()}\n"
            "Create this folder and place studentInfo.csv, studentVle.csv, "
            "studentAssessment.csv, assessments.csv and vle.csv inside it."
        )
    missing = [name for name in REQUIRED_CSVS if find_csv_path(name) is None]
    if missing:
        found = [str(p.relative_to(DATA_DIR)) for p in DATA_DIR.rglob("*.csv")]
        raise FileNotFoundError(
            "Missing required OULAD CSV files: " + ", ".join(missing) + "\n"
            f"OULAD directory: {DATA_DIR.resolve()}\n"
            "Found CSV files: " + (", ".join(found[:30]) if found else "none")
        )
    print(f"[Data] Local OULAD dataset found at: {DATA_DIR.resolve()}")


# -----------------------------------------------------------------------------
# 3. Loading and feature construction
# -----------------------------------------------------------------------------


def read_oulad_tables() -> Dict[str, pd.DataFrame]:
    print("[Data] Reading CSV files. studentVle.csv is large; this may take a few minutes.")

    student_info = pd.read_csv(require_csv_path("studentInfo.csv"))

    # studentVle.csv is large, so only the required columns are loaded.
    student_vle = pd.read_csv(
        require_csv_path("studentVle.csv"),
        usecols=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
        dtype={
            "code_module": "category",
            "code_presentation": "category",
            "id_student": "int32",
            "id_site": "int32",
            "date": "int16",
            "sum_click": "int16",
        },
    )

    # OULAD marks some missing assessment values with "?"; coerce them after loading.
    student_assessment = pd.read_csv(
        require_csv_path("studentAssessment.csv"),
        na_values=["?"],
        keep_default_na=True,
        low_memory=False,
    )
    student_assessment["id_assessment"] = pd.to_numeric(
        student_assessment["id_assessment"], errors="coerce"
    ).astype("int32")
    student_assessment["id_student"] = pd.to_numeric(
        student_assessment["id_student"], errors="coerce"
    ).astype("int32")
    student_assessment["date_submitted"] = pd.to_numeric(
        student_assessment["date_submitted"], errors="coerce"
    ).astype("float32")
    student_assessment["is_banked"] = pd.to_numeric(
        student_assessment["is_banked"], errors="coerce"
    ).fillna(0).astype("int8")
    student_assessment["score"] = pd.to_numeric(
        student_assessment["score"], errors="coerce"
    ).astype("float32")
    assessments = pd.read_csv(require_csv_path("assessments.csv"))
    vle = pd.read_csv(require_csv_path("vle.csv"))

    return {
        "student_info": student_info,
        "student_vle": student_vle,
        "student_assessment": student_assessment,
        "assessments": assessments,
        "vle": vle,
    }


def make_student_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["code_module"].astype(str)
        + "_"
        + df["code_presentation"].astype(str)
        + "_"
        + df["id_student"].astype(str)
    )


def prepare_base_student_frame(student_info: pd.DataFrame) -> pd.DataFrame:
    df = student_info.copy()
    df["student_key"] = make_student_key(df)

    # Risk label: Fail/Withdrawn = 1; Pass/Distinction = 0.
    risk_map = {"Fail": 1, "Withdrawn": 1, "Pass": 0, "Distinction": 0}
    df["risk_label"] = df["final_result"].map(risk_map)
    df = df[df["risk_label"].notna()].copy()
    df["risk_label"] = df["risk_label"].astype(int)

    for col in ["gender", "region", "highest_education", "imd_band", "age_band", "disability", "code_module", "code_presentation"]:
        if col in df.columns:
            df[col] = df[col].astype("object").fillna("Unknown")

    for col in ["num_of_prev_attempts", "studied_credits"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = [
        "student_key",
        "code_module",
        "code_presentation",
        "id_student",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "num_of_prev_attempts",
        "studied_credits",
        "disability",
        "risk_label",
        "final_result",
    ]
    return df[keep_cols]


def build_vle_features(student_vle: pd.DataFrame, cutoff_day: int, full_digital_twin: bool = True) -> pd.DataFrame:
    """Aggregate VLE activity before a cutoff day."""
    vle_logs = student_vle[(student_vle["date"] >= 0) & (student_vle["date"] <= cutoff_day)].copy()
    if vle_logs.empty:
        return pd.DataFrame(columns=["student_key"])
    vle_logs["student_key"] = make_student_key(vle_logs)

    # Overall activity counts.
    basic = vle_logs.groupby("student_key").agg(
        total_clicks=("sum_click", "sum"),
        vle_records=("sum_click", "size"),
        active_days=("date", "nunique"),
        last_activity_day=("date", "max"),
        mean_clicks_per_record=("sum_click", "mean"),
        max_clicks_in_record=("sum_click", "max"),
    ).reset_index()

    basic["days_since_last_activity"] = cutoff_day - basic["last_activity_day"]
    basic.drop(columns=["last_activity_day"], inplace=True)

    if not full_digital_twin:
        return basic[["student_key", "total_clicks", "vle_records", "active_days"]]

    # Daily activity summary.
    daily = vle_logs.groupby(["student_key", "date"], as_index=False)["sum_click"].sum()
    daily_agg = daily.groupby("student_key").agg(
        mean_daily_clicks=("sum_click", "mean"),
        std_daily_clicks=("sum_click", "std"),
        max_daily_clicks=("sum_click", "max"),
    ).reset_index()

    # Recent activity and simple trends.
    recent_start = max(0, cutoff_day - 6)
    recent = vle_logs[vle_logs["date"] >= recent_start].groupby("student_key").agg(
        recent_7d_clicks=("sum_click", "sum"),
        recent_7d_active_days=("date", "nunique"),
    ).reset_index()

    half = max(1, cutoff_day // 2)
    early = vle_logs[vle_logs["date"] <= half].groupby("student_key")["sum_click"].sum().rename("early_half_clicks")
    late = vle_logs[vle_logs["date"] > half].groupby("student_key")["sum_click"].sum().rename("late_half_clicks")
    trend = pd.concat([early, late], axis=1).fillna(0).reset_index()
    trend["click_trend_ratio"] = (trend["late_half_clicks"] + 1.0) / (trend["early_half_clicks"] + 1.0)
    trend["click_trend_diff"] = trend["late_half_clicks"] - trend["early_half_clicks"]

    # Weekly activity profile.
    vle_logs["week_index"] = (vle_logs["date"] // 7).astype(int)
    max_week = int(cutoff_day // 7)
    weekly = vle_logs.groupby(["student_key", "week_index"], as_index=False)["sum_click"].sum()
    weekly_pivot = weekly.pivot_table(
        index="student_key", columns="week_index", values="sum_click", aggfunc="sum", fill_value=0
    )
    weekly_pivot = weekly_pivot.rename(columns={w: f"week_{int(w):02d}_clicks" for w in weekly_pivot.columns}).reset_index()

    # Keep weekly columns consistent across cutoffs.
    for w in range(max_week + 1):
        col = f"week_{w:02d}_clicks"
        if col not in weekly_pivot.columns:
            weekly_pivot[col] = 0

    out = basic.merge(daily_agg, on="student_key", how="left")
    out = out.merge(recent, on="student_key", how="left")
    out = out.merge(trend, on="student_key", how="left")
    out = out.merge(weekly_pivot, on="student_key", how="left")
    return out


def build_assessment_features(
    student_assessment: pd.DataFrame,
    assessments: pd.DataFrame,
    cutoff_day: int,
    full_digital_twin: bool = True,
) -> pd.DataFrame:
    """Build assessment features available before a cutoff day."""
    assessment_meta = assessments.copy()
    assessment_meta["date"] = pd.to_numeric(assessment_meta["date"], errors="coerce")
    assessment_meta = assessment_meta[assessment_meta["assessment_type"].astype(str).str.lower() != "exam"].copy()

    submissions = student_assessment.merge(assessment_meta, on="id_assessment", how="left")
    submissions = submissions[submissions["code_module"].notna()].copy()
    submissions["date_submitted"] = pd.to_numeric(submissions["date_submitted"], errors="coerce")
    submissions["score"] = pd.to_numeric(submissions["score"], errors="coerce")
    submissions["weight"] = pd.to_numeric(submissions["weight"], errors="coerce").fillna(0)
    submissions = submissions[(submissions["date_submitted"].notna()) & (submissions["date_submitted"] <= cutoff_day)].copy()

    if submissions.empty:
        return pd.DataFrame(columns=["student_key"])

    submissions["student_key"] = make_student_key(submissions)
    submissions["delay_days"] = submissions["date_submitted"] - submissions["date"]
    submissions["late_flag"] = (submissions["delay_days"] > 0).astype(int)
    submissions["weighted_score_part"] = submissions["score"].fillna(0) * submissions["weight"].fillna(0)

    basic = submissions.groupby("student_key").agg(
        submitted_count=("id_assessment", "count"),
        mean_score=("score", "mean"),
        min_score=("score", "min"),
        max_score=("score", "max"),
        score_std=("score", "std"),
        mean_delay_days=("delay_days", "mean"),
        late_submission_count=("late_flag", "sum"),
        banked_count=("is_banked", "sum"),
        total_assessment_weight=("weight", "sum"),
        weighted_score_sum=("weighted_score_part", "sum"),
    ).reset_index()
    basic["weighted_score_mean"] = basic["weighted_score_sum"] / basic["total_assessment_weight"].replace(0, np.nan)
    basic["late_submission_rate"] = basic["late_submission_count"] / basic["submitted_count"].replace(0, np.nan)
    basic.drop(columns=["weighted_score_sum"], inplace=True)

    if not full_digital_twin:
        return basic[["student_key", "submitted_count", "mean_score", "weighted_score_mean"]]

    # Score change before the cutoff.
    half = max(1, cutoff_day // 2)
    early = submissions[submissions["date_submitted"] <= half].groupby("student_key")["score"].mean().rename("early_mean_score")
    late = submissions[submissions["date_submitted"] > half].groupby("student_key")["score"].mean().rename("late_mean_score")
    trend = pd.concat([early, late], axis=1).reset_index()
    trend["score_trend_diff"] = trend["late_mean_score"].fillna(0) - trend["early_mean_score"].fillna(0)

    out = basic.merge(trend, on="student_key", how="left")
    return out


def build_feature_table(
    tables: Dict[str, pd.DataFrame],
    cutoff_day: int,
    full_digital_twin: bool = True,
) -> pd.DataFrame:
    """Build the student-state table for one cutoff day."""
    base = prepare_base_student_frame(tables["student_info"])
    vle_feat = build_vle_features(tables["student_vle"], cutoff_day, full_digital_twin=full_digital_twin)
    ass_feat = build_assessment_features(
        tables["student_assessment"], tables["assessments"], cutoff_day, full_digital_twin=full_digital_twin
    )

    df = base.merge(vle_feat, on="student_key", how="left")
    df = df.merge(ass_feat, on="student_key", how="left")

    # Missing dynamic values mean no observed activity before the cutoff.
    dynamic_cols = [c for c in df.columns if c not in base.columns]
    for col in dynamic_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[dynamic_cols] = df[dynamic_cols].fillna(0)
    df["cutoff_day"] = cutoff_day

    if FAST_MODE and len(df) > FAST_MODE_SAMPLE_SIZE:
        df = df.sample(FAST_MODE_SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

    return df


# -----------------------------------------------------------------------------
# 4. Modeling utilities
# -----------------------------------------------------------------------------


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder compatible with common scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def get_feature_target_groups(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    drop_cols = ["risk_label", "final_result", "student_key"]
    meta_cols = ["student_key", "code_module", "code_presentation", "id_student", "final_result"]
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y = df["risk_label"].astype(int)
    groups = df["id_student"].astype(str)
    meta = df[[c for c in meta_cols if c in df.columns]].copy()
    return feature_df, y, groups, meta


def group_train_test_split(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return train_idx, test_idx


def make_model_zoo(preprocessor: ColumnTransformer, random_state: int = RANDOM_STATE) -> Dict[str, Pipeline]:
    """Create the baseline models and the CausalTwin-Edu predictor."""
    models: Dict[str, Any] = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
        "RandomForest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=random_state,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=1,
            random_state=random_state,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(80, 40),
            activation="relu",
            alpha=1e-4,
            max_iter=220,
            early_stopping=True,
            random_state=random_state,
        ),
    }

    # LightGBM is optional; skip it when the package is not installed.
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            objective="binary",
            n_estimators=450,
            learning_rate=0.035,
            num_leaves=31,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=0.05,
            reg_lambda=0.20,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
            verbosity=-1,
        )
    except Exception as exc:
        print(f"[Model][Info] LightGBM is not available and will be skipped: {exc}")

    # CausalTwin-Edu risk estimator: soft voting over complementary learners.
    causaltwin = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(
                n_estimators=220,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=random_state,
            )),
            ("et", ExtraTreesClassifier(
                n_estimators=260,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=1,
                random_state=random_state + 1,
            )),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
        ],
        voting="soft",
        n_jobs=1,
    )
    models["CausalTwin-Edu"] = causaltwin

    return {name: Pipeline([("prep", clone(preprocessor)), ("clf", clf)]) for name, clf in models.items()}


def safe_predict_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # Fallback for estimators without predict_proba.
    pred = model.predict(X)
    return pred.astype(float)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        out["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["auc"] = np.nan
    return out


def train_evaluate_window(
    df: pd.DataFrame,
    cutoff_day: int,
    experiment: str,
    random_state: int = RANDOM_STATE,
    model_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Tuple[np.ndarray, np.ndarray], pd.DataFrame, pd.Series]:
    X, y, groups, meta = get_feature_target_groups(df)
    train_idx, test_idx = group_train_test_split(X, y, groups, random_state=random_state)
    preprocessor = make_preprocessor(X)
    models = make_model_zoo(preprocessor, random_state=random_state)
    if model_names is not None:
        models = {name: model for name, model in models.items() if name in set(model_names)}

    records = []
    fitted_models = {}
    print(f"[Model] Training {experiment} models for cutoff_day={cutoff_day}...")
    for name, model in models.items():
        try:
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_prob = safe_predict_proba(model, X.iloc[test_idx])
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = evaluate_predictions(y.iloc[test_idx], y_pred, y_prob)
            metrics.update({"cutoff_day": cutoff_day, "week": cutoff_day // 7, "model": name, "experiment": experiment, "seed": random_state})
            records.append(metrics)
            fitted_models[name] = model
            print(f"    {name:18s} AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} Recall={metrics['recall']:.4f}")
        except Exception as e:
            print(f"    [Warning] {name} failed: {e}")

    return pd.DataFrame(records), fitted_models, (train_idx, test_idx), meta, y


# -----------------------------------------------------------------------------
# 5. Counterfactual intervention module
# -----------------------------------------------------------------------------


def intervention_action_space() -> List[Dict[str, float]]:
    """Build the bounded intervention action set."""
    actions: List[Dict[str, float]] = []

    def add_action(
        click_pct: float = 0.0,
        recent_pct: float = 0.0,
        active_add: float = 0.0,
        delay_reduce: float = 0.0,
        assessment_add: float = 0.0,
        score_boost: float = 0.0,
        consistency_boost: float = 0.0,
    ) -> None:
        action = {
            "click_increase_pct": float(click_pct),
            "recent_click_increase_pct": float(recent_pct),
            "active_days_add": float(active_add),
            "delay_days_reduce": float(delay_reduce),
            "assessment_submission_add": float(assessment_add),
            "score_practice_boost": float(score_boost),
            "consistency_boost": float(consistency_boost),
        }
        # Keep individual actions within the effort budget.
        if action_cost(action) <= 1.00 and any(float(v) > 0 for v in action.values()):
            actions.append(action)

    # Engagement actions.
    for click_increase in [0.10, 0.20, 0.35, 0.50, 0.70]:
        add_action(click_pct=click_increase)
    for recent_click_increase in [0.20, 0.40, 0.70, 1.00]:
        add_action(recent_pct=recent_click_increase)
    for active_day_gain in [1, 2, 3, 5, 7]:
        add_action(active_add=active_day_gain)

    # Timeliness and consistency actions.
    for delay_reduction in [1, 2, 3, 5, 7]:
        add_action(delay_reduce=delay_reduction)
    for consistency_gain in [0.20, 0.40, 0.60]:
        add_action(consistency_boost=consistency_gain)

    # Assessment actions.
    for assessment_gain in [1, 2]:
        add_action(assessment_add=assessment_gain)
    for score_gain in [0.03, 0.05, 0.08, 0.10]:
        add_action(score_boost=score_gain)

    # Paired action templates.
    click_opts = [0.10, 0.20, 0.35, 0.50]
    recent_opts = [0.20, 0.40, 0.70]
    active_opts = [1, 2, 3, 5]
    delay_opts = [1, 2, 3, 5]
    assess_opts = [1, 2]
    score_opts = [0.03, 0.05, 0.08]
    consistency_opts = [0.20, 0.40]

    for click_increase in click_opts:
        for active_day_gain in active_opts:
            add_action(click_pct=click_increase, active_add=active_day_gain)
    for recent_click_increase in recent_opts:
        for active_day_gain in active_opts:
            add_action(recent_pct=recent_click_increase, active_add=active_day_gain)
    for click_increase in click_opts:
        for delay_reduction in delay_opts:
            add_action(click_pct=click_increase, delay_reduce=delay_reduction)
    for active_day_gain in active_opts:
        for delay_reduction in delay_opts:
            add_action(active_add=active_day_gain, delay_reduce=delay_reduction)
    for click_increase in click_opts:
        for consistency_gain in consistency_opts:
            add_action(click_pct=click_increase, consistency_boost=consistency_gain)
    for recent_click_increase in recent_opts:
        for consistency_gain in consistency_opts:
            add_action(recent_pct=recent_click_increase, consistency_boost=consistency_gain)
    for active_day_gain in [1, 2, 3]:
        for consistency_gain in consistency_opts:
            add_action(active_add=active_day_gain, consistency_boost=consistency_gain)

    # Balanced assessment combinations.
    for assessment_gain in assess_opts:
        for delay_reduction in [1, 2, 3]:
            add_action(assessment_add=assessment_gain, delay_reduce=delay_reduction)
    for assessment_gain in assess_opts:
        for score_gain in score_opts:
            add_action(assessment_add=assessment_gain, score_boost=score_gain)
    for assessment_gain in [1]:
        for recent_click_increase in [0.20, 0.40, 0.70]:
            add_action(assessment_add=assessment_gain, recent_pct=recent_click_increase)
    for assessment_gain in [1]:
        for active_day_gain in [1, 2, 3]:
            add_action(assessment_add=assessment_gain, active_add=active_day_gain)

    # Compact three-part packages.
    for click_increase in [0.10, 0.20, 0.35]:
        for active_day_gain in [1, 2, 3]:
            for delay_reduction in [1, 2, 3]:
                add_action(click_pct=click_increase, active_add=active_day_gain, delay_reduce=delay_reduction)
    for recent_click_increase in [0.20, 0.40]:
        for active_day_gain in [1, 2, 3]:
            for consistency_gain in [0.20, 0.40]:
                add_action(recent_pct=recent_click_increase, active_add=active_day_gain, consistency_boost=consistency_gain)
    for assessment_gain in [1]:
        for delay_reduction in [1, 2, 3]:
            for recent_click_increase in [0.20, 0.40]:
                add_action(assessment_add=assessment_gain, delay_reduce=delay_reduction, recent_pct=recent_click_increase)
    for assessment_gain in [1]:
        for active_day_gain in [1, 2]:
            for consistency_gain in [0.20, 0.40]:
                add_action(assessment_add=assessment_gain, active_add=active_day_gain, consistency_boost=consistency_gain)

    # Drop duplicate actions.
    seen = set()
    unique_actions: List[Dict[str, float]] = []
    for a in actions:
        key = tuple(sorted((k, round(float(v), 4)) for k, v in a.items()))
        if key not in seen:
            seen.add(key)
            unique_actions.append(a)
    return unique_actions


def action_family(action: Dict[str, float]) -> str:
    """Assign an action to a broad intervention family."""
    click_increase = action.get("click_increase_pct", 0.0)
    recent_click_increase = action.get("recent_click_increase_pct", 0.0)
    active_day_gain = action.get("active_days_add", 0.0)
    delay_reduction = action.get("delay_days_reduce", 0.0)
    assessment_gain = action.get("assessment_submission_add", 0.0)
    score_gain = action.get("score_practice_boost", 0.0)
    consistency_gain = action.get("consistency_boost", 0.0)

    components = {
        "engagement_intensity": click_increase + recent_click_increase,
        "learning_regularization": active_day_gain / 7.0 + consistency_gain,
        "submission_timeliness": delay_reduction / 7.0,
        "assessment_participation": assessment_gain / 2.0 + score_gain,
    }
    active_families = [k for k, v in components.items() if v > 1e-9]
    if len(active_families) >= 2:
        # Name common mixed packages.
        if assessment_gain > 0 and (recent_click_increase > 0 or click_increase > 0 or active_day_gain > 0):
            return "balanced_assessment_engagement"
        if delay_reduction > 0 and (click_increase > 0 or recent_click_increase > 0 or active_day_gain > 0):
            return "engagement_timeliness_package"
        if consistency_gain > 0 and (click_increase > 0 or recent_click_increase > 0 or active_day_gain > 0):
            return "regular_engagement_package"
        return "multi_component_package"
    if not active_families:
        return "none"
    return max(components, key=components.get)


def risk_band_from_prob(p: float) -> str:
    if p < 0.60:
        return "low_or_borderline"
    if p < 0.75:
        return "moderate_risk"
    if p < 0.90:
        return "high_risk"
    return "extreme_risk"


def apply_action(row: pd.Series, action: Dict[str, float], cutoff_day: int) -> pd.Series:
    """Apply one intervention action to a feature row."""
    x = row.copy()

    def has(col: str) -> bool:
        return col in x.index

    def num(col: str, default: float = 0.0) -> float:
        if not has(col):
            return default
        val = pd.to_numeric(pd.Series([x[col]]), errors="coerce").iloc[0]
        return default if pd.isna(val) else float(val)

    click_increase = action.get("click_increase_pct", 0.0)
    recent_click_increase = action.get("recent_click_increase_pct", 0.0)
    active_day_gain = action.get("active_days_add", 0.0)
    delay_reduction = action.get("delay_days_reduce", 0.0)
    # Additional assessment submissions are capped at two.
    assessment_gain = min(action.get("assessment_submission_add", 0.0), 2.0)
    score_gain = action.get("score_practice_boost", 0.0)
    consistency_gain = action.get("consistency_boost", 0.0)

    # VLE engagement changes.
    old_total = num("total_clicks")
    total_gain = max(0.0, old_total * click_increase)
    if has("total_clicks"):
        x["total_clicks"] = old_total + total_gain
    if has("vle_records"):
        x["vle_records"] = num("vle_records") * (1.0 + 0.30 * click_increase + 0.25 * recent_click_increase) + 0.5 * active_day_gain
    if has("mean_clicks_per_record"):
        x["mean_clicks_per_record"] = num("mean_clicks_per_record") * (1.0 + 0.30 * click_increase + 0.18 * recent_click_increase)
    if has("max_clicks_in_record"):
        x["max_clicks_in_record"] = max(num("max_clicks_in_record"), num("mean_clicks_per_record") * 1.2)

    if has("recent_7d_clicks"):
        base_recent = num("recent_7d_clicks")
        x["recent_7d_clicks"] = base_recent * (1.0 + recent_click_increase) + max(1.0, 2.0 * active_day_gain) + 0.20 * total_gain
    if has("active_days"):
        x["active_days"] = min(float(cutoff_day + 1), max(0.0, num("active_days") + active_day_gain))
    if has("recent_7d_active_days"):
        x["recent_7d_active_days"] = min(7.0, max(0.0, num("recent_7d_active_days") + min(active_day_gain, 7)))
    if has("days_since_last_activity"):
        x["days_since_last_activity"] = max(0.0, num("days_since_last_activity") - active_day_gain)

    weekly_cols = [c for c in x.index if c.startswith("week_") and c.endswith("_clicks")]
    if weekly_cols:
        weekly_vals = np.array([num(c) for c in weekly_cols], dtype=float)
        if weekly_vals.sum() > 0:
            mean_weekly = weekly_vals.mean()
            adjusted = weekly_vals + consistency_gain * np.maximum(mean_weekly - weekly_vals, 0)
            extra_recent = num("recent_7d_clicks") * recent_click_increase * 0.22 + total_gain * 0.12
            if len(adjusted) >= 1:
                adjusted[-1] += extra_recent * 0.65
            if len(adjusted) >= 2:
                adjusted[-2] += extra_recent * 0.35
            for col, val in zip(weekly_cols, adjusted):
                x[col] = max(0.0, val)
    if has("mean_daily_clicks"):
        denom = max(1.0, min(float(cutoff_day + 1), num("active_days", 1.0)))
        x["mean_daily_clicks"] = max(num("mean_daily_clicks"), num("total_clicks") / denom)
    if has("std_daily_clicks"):
        x["std_daily_clicks"] = max(0.0, num("std_daily_clicks") * (1.0 - 0.20 * consistency_gain))
    if has("max_daily_clicks"):
        x["max_daily_clicks"] = max(num("max_daily_clicks"), num("mean_daily_clicks") * 1.5)

    if has("early_half_clicks") or has("late_half_clicks"):
        if has("late_half_clicks"):
            x["late_half_clicks"] = num("late_half_clicks") * (1.0 + 0.40 * click_increase + 0.52 * recent_click_increase + 0.15 * consistency_gain) + max(0.0, 1.5 * active_day_gain)
        if has("early_half_clicks"):
            x["early_half_clicks"] = max(0.0, num("early_half_clicks"))
        if has("click_trend_ratio"):
            x["click_trend_ratio"] = (num("late_half_clicks") + 1.0) / (num("early_half_clicks") + 1.0)
        if has("click_trend_diff"):
            x["click_trend_diff"] = num("late_half_clicks") - num("early_half_clicks")

    # Assessment-related changes; static variables are left untouched.
    if has("submitted_count"):
        x["submitted_count"] = max(0.0, num("submitted_count") + assessment_gain)
    if has("total_assessment_weight"):
        x["total_assessment_weight"] = max(0.0, num("total_assessment_weight") + 4.0 * assessment_gain)
    if has("mean_delay_days"):
        x["mean_delay_days"] = num("mean_delay_days") - delay_reduction
    if has("late_submission_count"):
        x["late_submission_count"] = max(0.0, num("late_submission_count") - min(delay_reduction, num("late_submission_count")))
    if has("late_submission_rate"):
        x["late_submission_rate"] = max(0.0, min(1.0, num("late_submission_rate") * (1.0 - 0.12 * delay_reduction) - 0.025 * assessment_gain))

    score_cols = ["mean_score", "weighted_score_mean", "late_mean_score"]
    for col in score_cols:
        if has(col):
            old = num(col)
            x[col] = min(100.0, old * (1.0 + score_gain) + 0.75 * assessment_gain)
    if has("min_score"):
        x["min_score"] = min(100.0, num("min_score") * (1.0 + 0.5 * score_gain))
    if has("max_score"):
        x["max_score"] = min(100.0, max(num("max_score"), num("mean_score")))
    if has("score_std"):
        x["score_std"] = max(0.0, num("score_std") * (1.0 - 0.15 * consistency_gain))
    if has("score_trend_diff"):
        x["score_trend_diff"] = num("late_mean_score") - num("early_mean_score", num("late_mean_score"))

    return x


def action_cost(action: Dict[str, float]) -> float:
    """Compute the normalized effort cost of an action."""
    click_increase = action.get("click_increase_pct", 0.0) / 0.70
    recent_click_increase = action.get("recent_click_increase_pct", 0.0) / 1.00
    active_day_gain = action.get("active_days_add", 0.0) / 7.0
    delay_reduction = action.get("delay_days_reduce", 0.0) / 7.0
    assessment_gain = action.get("assessment_submission_add", 0.0) / 2.0
    score_gain = action.get("score_practice_boost", 0.0) / 0.10
    consistency_gain = action.get("consistency_boost", 0.0) / 0.60
    return float(
        0.15 * click_increase +
        0.13 * recent_click_increase +
        0.13 * active_day_gain +
        0.16 * delay_reduction +
        0.22 * assessment_gain +
        0.13 * score_gain +
        0.08 * consistency_gain
    )


def describe_action(action: Dict[str, float]) -> str:
    parts = []
    if action.get("click_increase_pct", 0) > 0:
        parts.append(f"increase total VLE clicks by {int(action['click_increase_pct'] * 100)}%")
    if action.get("recent_click_increase_pct", 0) > 0:
        parts.append(f"increase recent 7-day clicks by {int(action['recent_click_increase_pct'] * 100)}%")
    if action.get("active_days_add", 0) > 0:
        parts.append(f"add {int(action['active_days_add'])} active learning days")
    if action.get("delay_days_reduce", 0) > 0:
        parts.append(f"reduce submission delay by {int(action['delay_days_reduce'])} days")
    if action.get("assessment_submission_add", 0) > 0:
        parts.append(f"complete {int(action['assessment_submission_add'])} additional formative assessment(s)")
    if action.get("score_practice_boost", 0) > 0:
        parts.append(f"improve formative assessment preparation by {int(action['score_practice_boost'] * 100)}%")
    if action.get("consistency_boost", 0) > 0:
        parts.append(f"make weekly study pattern {int(action['consistency_boost'] * 100)}% more consistent")
    return "; ".join(parts)


def predict_one(model: Pipeline, x_row: pd.Series) -> float:
    x_df = pd.DataFrame([x_row])
    return float(safe_predict_proba(model, x_df)[0])


def _risk_thresholds(y_train_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    return {
        "candidate_min_drop": 0.040,
        "moderate_min_drop": 0.070,
        "high_min_drop": 0.100,
        "max_candidate_cost": 1.00,
        "max_high_cost": 0.88,
        "min_candidate_stability": 0.33,
        "min_high_stability": 0.50,
        "target_after_high": 0.60,
    }


def classify_reliability(
    risk_before: float,
    risk_after: float,
    drop: float,
    cost: float,
    stability: float,
    thresholds: Dict[str, float],
) -> str:
    if drop <= 0:
        return "NoImprovement"
    relative_drop = drop / max(risk_before, 1e-6)
    band = risk_band_from_prob(risk_before)

    # Extreme-risk cases can remain high-risk even after a useful reduction.
    if (
        drop >= thresholds["high_min_drop"]
        and relative_drop >= 0.10
        and cost <= thresholds["max_high_cost"]
        and stability >= thresholds["min_high_stability"]
        and (risk_after <= thresholds["target_after_high"] or drop >= 0.15 or band == "extreme_risk")
    ):
        return "HighReliability"
    if (
        drop >= thresholds["moderate_min_drop"]
        and cost <= thresholds["max_candidate_cost"]
        and stability >= thresholds["min_candidate_stability"]
    ):
        return "ModerateCandidate"
    if drop >= thresholds["candidate_min_drop"] and cost <= thresholds["max_candidate_cost"]:
        return "ManualReviewCandidate"
    return "NoImprovement"


def select_intervention_indices(prob: np.ndarray, cap: int) -> np.ndarray:
    """Select intervention cases from moderate, high and extreme risk bands."""
    rng = np.random.default_rng(RANDOM_STATE)
    bands = {
        "moderate_risk": np.where((prob >= 0.60) & (prob < 0.75))[0],
        "high_risk": np.where((prob >= 0.75) & (prob < 0.90))[0],
        "extreme_risk": np.where(prob >= 0.90)[0],
    }
    # Use a rough one-third split across risk bands.
    per_band = max(1, cap // 3)
    selected: List[int] = []
    for name in ["moderate_risk", "high_risk", "extreme_risk"]:
        idx = bands[name]
        if len(idx) == 0:
            continue
        if name == "extreme_risk":
            chosen = idx[np.argsort(-prob[idx])[:per_band]]
        else:
            # Pick cases near the center of the band.
            center = 0.675 if name == "moderate_risk" else 0.825
            order = np.argsort(np.abs(prob[idx] - center))
            chosen = idx[order[:per_band]]
        selected.extend(chosen.tolist())

    if len(selected) < cap:
        existing = set(selected)
        fallback = [int(i) for i in np.argsort(-prob) if int(i) not in existing and prob[int(i)] >= 0.60]
        selected.extend(fallback[: cap - len(selected)])

    # Keep a deterministic order.
    selected = selected[:cap]
    selected = sorted(selected, key=lambda i: (risk_band_from_prob(float(prob[i])), -float(prob[i])))
    return np.array(selected, dtype=int)


def _model_embedding(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return the transformed feature matrix used by a fitted pipeline."""
    prep = model.named_steps.get("prep") if hasattr(model, "named_steps") else None
    if prep is None:
        arr = X.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    else:
        arr = prep.transform(X)
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    arr = np.asarray(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _fit_plausibility_reference(
    model: Pipeline,
    X_train: Optional[pd.DataFrame],
    n_neighbors: int = 5,
) -> Optional[Dict[str, Any]]:
    """Fit a nearest-neighbor reference for plausibility checks."""
    if X_train is None or len(X_train) < 20:
        return None
    emb = _model_embedding(model, X_train)
    k = max(2, min(n_neighbors + 1, len(emb)))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(emb)
    train_dist, _ = nn.kneighbors(emb, n_neighbors=k)
    # Ignore the self-neighbor when it is present.
    ref_dist = train_dist[:, 1:].mean(axis=1) if train_dist.shape[1] > 1 else train_dist[:, 0]
    threshold = float(np.quantile(ref_dist, 0.95))
    threshold = max(threshold, 1e-8)
    return {"nn": nn, "threshold": threshold, "n_neighbors": min(n_neighbors, len(emb))}


def _plausibility_scores(
    model: Pipeline,
    X_candidate: pd.DataFrame,
    ref: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute distance, plausibility score and gate flag."""
    if ref is None:
        n = len(X_candidate)
        return np.zeros(n), np.ones(n), np.ones(n, dtype=bool)
    emb = _model_embedding(model, X_candidate)
    k = int(ref.get("n_neighbors", 5))
    nn = ref["nn"]
    dist, _ = nn.kneighbors(emb, n_neighbors=min(k, len(getattr(nn, "_fit_X", emb))))
    d = dist.mean(axis=1)
    thr = float(ref["threshold"])
    # Inside the training radius the score stays at 1, then decays smoothly.
    excess = np.maximum(0.0, d / (thr + 1e-12) - 1.0)
    score = np.exp(-excess)
    gate = d <= 1.25 * thr
    return d, score, gate


def _pareto_front_mask(risk_after: np.ndarray, costs: np.ndarray, stability: np.ndarray, plausibility: np.ndarray) -> np.ndarray:
    """Return the non-dominated candidates."""
    n = len(risk_after)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # Candidate j dominates candidate i.
        better_or_equal = (
            (risk_after <= risk_after[i] + 1e-12)
            & (costs <= costs[i] + 1e-12)
            & (stability >= stability[i] - 1e-12)
            & (plausibility >= plausibility[i] - 1e-12)
        )
        strictly_better = (
            (risk_after < risk_after[i] - 1e-12)
            | (costs < costs[i] - 1e-12)
            | (stability > stability[i] + 1e-12)
            | (plausibility > plausibility[i] + 1e-12)
        )
        if np.any(better_or_equal & strictly_better):
            mask[i] = False
    return mask


def make_threshold_sensitivity(cases: pd.DataFrame) -> pd.DataFrame:
    """Build a sensitivity table for reliability thresholds."""
    if cases is None or cases.empty:
        return pd.DataFrame()
    rows = []
    for min_drop in [0.05, 0.08, 0.10, 0.12, 0.15]:
        for min_stability in [0.50, 0.67, 0.80]:
            for min_plausibility in [0.40, 0.60, 0.80]:
                accepted = (
                    (cases["risk_drop"] >= min_drop)
                    & (cases["stability"] >= min_stability)
                    & (cases["plausibility_score"] >= min_plausibility)
                    & (cases["intervention_cost"] <= 1.0)
                )
                subset = cases[accepted]
                rows.append({
                    "min_drop": min_drop,
                    "min_stability": min_stability,
                    "min_plausibility": min_plausibility,
                    "accepted_rate": float(accepted.mean()),
                    "n_accepted": int(accepted.sum()),
                    "mean_risk_drop_if_accepted": float(subset["risk_drop"].mean()) if len(subset) else np.nan,
                    "mean_cost_if_accepted": float(subset["intervention_cost"].mean()) if len(subset) else np.nan,
                    "mean_IRI_if_accepted": float(subset["IRI"].mean()) if len(subset) else np.nan,
                })
    return pd.DataFrame(rows)


def generate_interventions(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    meta_test: pd.DataFrame,
    main_model: Pipeline,
    auxiliary_models: List[Pipeline],
    cutoff_day: int,
    X_train: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate counterfactual intervention recommendations."""
    print("[Intervention] Generating plausibility-aware Pareto counterfactual recommendations...")
    prob = safe_predict_proba(main_model, X_test)

    plaus_ref = _fit_plausibility_reference(main_model, X_train)
    if plaus_ref is not None:
        print(f"[Intervention] Plausibility reference fitted. 95% neighbor-distance threshold={plaus_ref['threshold']:.4f}")
    else:
        print("[Intervention] Plausibility reference unavailable; using neutral plausibility scores.")

    intervention_cap = min(MAX_INTERVENTION_CASES, 36) if FAST_MODE else MAX_INTERVENTION_CASES
    selected_idx = select_intervention_indices(prob, intervention_cap)
    if len(selected_idx) == 0:
        selected_idx = np.argsort(-prob)[:intervention_cap]

    actions = intervention_action_space()
    thresholds = _risk_thresholds()
    print(f"[Intervention] Selected {len(selected_idx)} risk-stratified students; evaluating {len(actions)} candidate actions per student.")
    records = []
    family_counts: Dict[str, int] = {}

    for case_no, local_pos in enumerate(selected_idx, start=1):
        original = X_test.iloc[local_pos]
        student_meta = meta_test.iloc[local_pos].to_dict()
        risk_before = float(prob[local_pos])
        risk_band = risk_band_from_prob(risk_before)

        modified_rows = [apply_action(original, action, cutoff_day) for action in actions]
        modified_df = pd.DataFrame(modified_rows)
        modified_df = modified_df[X_test.columns]

        risk_after_arr = safe_predict_proba(main_model, modified_df)
        drops = risk_before - risk_after_arr
        costs = np.array([action_cost(action) for action in actions], dtype=float)
        families = np.array([action_family(action) for action in actions])
        plaus_dist_arr, plaus_score_arr, plaus_gate_arr = _plausibility_scores(main_model, modified_df, plaus_ref)

        aux_drop_matrix = []
        aux_success_matrix = []
        for aux_model in auxiliary_models:
            try:
                aux_before = float(safe_predict_proba(aux_model, pd.DataFrame([original]))[0])
                aux_after_arr = safe_predict_proba(aux_model, modified_df)
                aux_drops = aux_before - aux_after_arr
                aux_drop_matrix.append(aux_drops)
                aux_success_matrix.append((aux_drops >= 0.020) | (aux_drops >= 0.06 * max(aux_before, 1e-6)))
            except Exception:
                continue
        if aux_success_matrix:
            stability_arr = np.vstack(aux_success_matrix).mean(axis=0)
            mean_aux_drop_arr = np.vstack(aux_drop_matrix).mean(axis=0)
        else:
            stability_arr = np.zeros(len(actions), dtype=float)
            mean_aux_drop_arr = np.zeros(len(actions), dtype=float)

        feasibility_arr = 1.0 - np.clip(costs, 0.0, 1.0)
        positive_drop_arr = np.maximum(drops, 0.0)
        pareto_mask = _pareto_front_mask(risk_after_arr, costs, stability_arr, plaus_score_arr)

        iri_arr = (
            (positive_drop_arr / (costs + 0.15))
            * (0.45 + 0.55 * stability_arr)
            * (0.55 + 0.45 * feasibility_arr)
            * (0.45 + 0.55 * plaus_score_arr)
        )

        base_levels = np.array([
            classify_reliability(
                risk_before=risk_before,
                risk_after=float(risk_after_arr[i]),
                drop=float(drops[i]),
                cost=float(costs[i]),
                stability=float(stability_arr[i]),
                thresholds=thresholds,
            )
            for i in range(len(actions))
        ])

        reliability_levels = []
        for i, level in enumerate(base_levels):
            # Downgrade candidates that leave the observed data manifold.
            if plaus_score_arr[i] < 0.35 or not plaus_gate_arr[i]:
                if drops[i] >= thresholds["candidate_min_drop"] and costs[i] <= thresholds["max_candidate_cost"]:
                    reliability_levels.append("ManualReviewCandidate")
                else:
                    reliability_levels.append("NoImprovement")
            elif plaus_score_arr[i] < 0.60 and level == "HighReliability":
                reliability_levels.append("ModerateCandidate")
            else:
                reliability_levels.append(str(level))
        reliability_levels = np.array(reliability_levels)

        tier_rank = {"HighReliability": 3, "ModerateCandidate": 2, "ManualReviewCandidate": 1, "NoImprovement": 0}
        ranks = np.array([tier_rank[level] for level in reliability_levels], dtype=float)

        total_selected = max(1, sum(family_counts.values()))
        diversity_penalty = np.array([family_counts.get(str(f), 0) / total_selected for f in families], dtype=float)
        pure_assessment_penalty = np.array([0.20 if f == "assessment_participation" else 0.0 for f in families], dtype=float)
        pareto_bonus = pareto_mask.astype(float)

        # Use this score only to break ties after reliability and Pareto checks.
        selection_score = (
            1.15 * ranks
            + 1.85 * positive_drop_arr
            + 0.65 * stability_arr
            + 0.45 * feasibility_arr
            + 0.65 * plaus_score_arr
            + 0.30 * pareto_bonus
            + 0.15 * iri_arr
            - 0.20 * risk_after_arr
            - 0.40 * diversity_penalty
            - pure_assessment_penalty
        )
        # Prefer accepted candidates on the Pareto front.
        accepted_or_review = ranks >= 1
        preferred_pool = accepted_or_review & pareto_mask & (plaus_score_arr >= 0.35)
        if preferred_pool.any():
            masked_score = np.where(preferred_pool, selection_score, -np.inf)
            best_idx = int(np.argmax(masked_score))
        else:
            best_idx = int(np.argmax(selection_score))

        action = actions[best_idx]
        family = str(families[best_idx])
        family_counts[family] = family_counts.get(family, 0) + 1
        level = str(reliability_levels[best_idx])
        candidate = {
            **student_meta,
            "risk_band": risk_band,
            "true_risk_label": int(y_test.iloc[local_pos]),
            "risk_before": risk_before,
            "risk_after": float(risk_after_arr[best_idx]),
            "risk_drop": float(drops[best_idx]),
            "relative_risk_drop": float(drops[best_idx] / max(risk_before, 1e-6)),
            "intervention_cost": float(costs[best_idx]),
            "stability": float(stability_arr[best_idx]),
            "plausibility_distance": float(plaus_dist_arr[best_idx]),
            "plausibility_score": float(plaus_score_arr[best_idx]),
            "plausibility_gate": bool(plaus_gate_arr[best_idx]),
            "on_pareto_front": bool(pareto_mask[best_idx]),
            "mean_auxiliary_risk_drop": float(mean_aux_drop_arr[best_idx]),
            "IRI": float(iri_arr[best_idx]),
            "reliability_level": level,
            "accepted_by_reliability_gate": bool(level in ["HighReliability", "ModerateCandidate"]),
            "candidate_for_manual_review": bool(level == "ManualReviewCandidate"),
            "action_family": family,
            **action,
            "action_description": describe_action(action),
        }
        records.append(candidate)

        if case_no == 1 or case_no % 5 == 0 or case_no == len(selected_idx):
            print(
                f"[Intervention] {case_no}/{len(selected_idx)} done | "
                f"band={risk_band}, risk {candidate['risk_before']:.3f}->{candidate['risk_after']:.3f}, "
                f"drop={candidate['risk_drop']:.3f}, stab={candidate['stability']:.2f}, "
                f"plaus={candidate['plausibility_score']:.2f}, pareto={candidate['on_pareto_front']}, "
                f"family={family}, level={candidate['reliability_level']}"
            )

    cases = pd.DataFrame(records)
    if cases.empty:
        summary = pd.DataFrame([{
            "n_cases": 0,
            "accepted_rate": np.nan,
            "high_reliability_rate": np.nan,
            "moderate_candidate_rate": np.nan,
            "manual_review_candidate_rate": np.nan,
            "no_improvement_rate": np.nan,
            "mean_risk_before": np.nan,
            "mean_risk_after": np.nan,
            "mean_risk_drop": np.nan,
            "mean_relative_risk_drop": np.nan,
            "mean_cost": np.nan,
            "mean_stability": np.nan,
            "mean_plausibility_score": np.nan,
            "plausible_recommendation_rate": np.nan,
            "pareto_selected_rate": np.nan,
            "mean_auxiliary_risk_drop": np.nan,
            "mean_IRI": np.nan,
            "action_family_count": np.nan,
            "action_family_entropy": np.nan,
        }])
        return cases, summary

    n_families = cases["action_family"].nunique()
    family_entropy = 0.0
    family_probs = cases["action_family"].value_counts(normalize=True).values
    if len(family_probs):
        family_entropy = float(-(family_probs * np.log(family_probs + 1e-12)).sum())

    summary = pd.DataFrame([{
        "n_cases": len(cases),
        "accepted_rate": cases["accepted_by_reliability_gate"].mean(),
        "high_reliability_rate": (cases["reliability_level"] == "HighReliability").mean(),
        "moderate_candidate_rate": (cases["reliability_level"] == "ModerateCandidate").mean(),
        "manual_review_candidate_rate": (cases["reliability_level"] == "ManualReviewCandidate").mean(),
        "no_improvement_rate": (cases["reliability_level"] == "NoImprovement").mean(),
        "mean_risk_before": cases["risk_before"].mean(),
        "mean_risk_after": cases["risk_after"].mean(),
        "mean_risk_drop": cases["risk_drop"].mean(),
        "mean_relative_risk_drop": cases["relative_risk_drop"].mean(),
        "mean_cost": cases["intervention_cost"].mean(),
        "mean_stability": cases["stability"].mean(),
        "mean_plausibility_score": cases["plausibility_score"].mean(),
        "plausible_recommendation_rate": cases["plausibility_gate"].mean(),
        "pareto_selected_rate": cases["on_pareto_front"].mean(),
        "mean_auxiliary_risk_drop": cases["mean_auxiliary_risk_drop"].mean(),
        "mean_IRI": cases["IRI"].mean(),
        "action_family_count": n_families,
        "action_family_entropy": family_entropy,
    }])
    return cases, summary


# -----------------------------------------------------------------------------
# 6. Robustness and uncertainty utilities
# -----------------------------------------------------------------------------


def run_seed_robustness(feature_tables: Dict[int, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the multi-seed robustness check."""
    if not RUN_ROBUSTNESS_CHECK:
        return pd.DataFrame(), pd.DataFrame()

    print("\n" + "-" * 80)
    print("[Robustness] Running lightweight multi-seed robustness check...")
    rows: List[pd.DataFrame] = []
    for seed in ROBUSTNESS_SEEDS:
        for cutoff in TIME_WINDOWS:
            df = feature_tables[cutoff]
            metrics, _, _, _, _ = train_evaluate_window(
                df=df,
                cutoff_day=cutoff,
                experiment="robustness_full_digital_twin",
                random_state=seed,
                model_names=ROBUSTNESS_MODELS,
            )
            rows.append(metrics)
    robust_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if robust_df.empty:
        return robust_df, pd.DataFrame()

    numeric_metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"]
    summary = robust_df.groupby(["week", "cutoff_day", "model"])[numeric_metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary = summary.reset_index().sort_values(["week", "model"])
    return robust_df, summary


def bootstrap_intervention_summary(cases: pd.DataFrame, n_boot: int = BOOTSTRAP_N, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Compute bootstrap intervals for intervention summaries."""
    if cases is None or cases.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    metrics = {
        "accepted_rate": lambda d: float(d["accepted_by_reliability_gate"].mean()),
        "high_reliability_rate": lambda d: float((d["reliability_level"] == "HighReliability").mean()),
        "mean_risk_before": lambda d: float(d["risk_before"].mean()),
        "mean_risk_after": lambda d: float(d["risk_after"].mean()),
        "mean_risk_drop": lambda d: float(d["risk_drop"].mean()),
        "mean_relative_risk_drop": lambda d: float(d["relative_risk_drop"].mean()),
        "mean_cost": lambda d: float(d["intervention_cost"].mean()),
        "mean_stability": lambda d: float(d["stability"].mean()),
        "mean_IRI": lambda d: float(d["IRI"].mean()),
    }
    records: List[Dict[str, float]] = []
    n = len(cases)
    for name, fn in metrics.items():
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            sample = cases.iloc[idx]
            vals.append(fn(sample))
        arr = np.asarray(vals, dtype=float)
        records.append({
            "metric": name,
            "observed": fn(cases),
            "bootstrap_mean": float(np.mean(arr)),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
            "n_bootstrap": int(n_boot),
        })
    return pd.DataFrame(records)


def make_quality_check(metrics: pd.DataFrame, ablation: pd.DataFrame, intervention_summary: pd.DataFrame) -> pd.DataFrame:
    """Create a compact check of the generated results."""
    checks: List[Dict[str, Any]] = []
    full = metrics[(metrics["experiment"] == "full_digital_twin") & (metrics["model"] == "CausalTwin-Edu")]
    week8_auc = float(full.loc[full["week"] == 8, "auc"].iloc[0]) if len(full.loc[full["week"] == 8]) else np.nan
    week12_auc = float(full.loc[full["week"] == 12, "auc"].iloc[0]) if len(full.loc[full["week"] == 12]) else np.nan
    checks.append({"item": "Week-8 CausalTwin-Edu AUC >= 0.88", "value": week8_auc, "status": "PASS" if week8_auc >= 0.88 else "REVIEW"})
    checks.append({"item": "Week-12 CausalTwin-Edu AUC >= 0.90", "value": week12_auc, "status": "PASS" if week12_auc >= 0.90 else "REVIEW"})

    if not intervention_summary.empty:
        row = intervention_summary.iloc[0]
        checks.extend([
            {"item": "Intervention accepted_rate >= 0.65", "value": float(row.get("accepted_rate", np.nan)), "status": "PASS" if float(row.get("accepted_rate", 0)) >= 0.65 else "REVIEW"},
            {"item": "Mean risk drop >= 0.08", "value": float(row.get("mean_risk_drop", np.nan)), "status": "PASS" if float(row.get("mean_risk_drop", 0)) >= 0.08 else "REVIEW"},
            {"item": "Mean stability >= 0.60", "value": float(row.get("mean_stability", np.nan)), "status": "PASS" if float(row.get("mean_stability", 0)) >= 0.60 else "REVIEW"},
            {"item": "Mean plausibility score >= 0.50", "value": float(row.get("mean_plausibility_score", np.nan)), "status": "PASS" if float(row.get("mean_plausibility_score", 0)) >= 0.50 else "REVIEW"},
            {"item": "Plausible recommendation rate >= 0.60", "value": float(row.get("plausible_recommendation_rate", np.nan)), "status": "PASS" if float(row.get("plausible_recommendation_rate", 0)) >= 0.60 else "REVIEW"},
            {"item": "At least 3 action families", "value": float(row.get("action_family_count", np.nan)), "status": "PASS" if float(row.get("action_family_count", 0)) >= 3 else "REVIEW"},
        ])
    return pd.DataFrame(checks)


def write_environment_snapshot() -> None:
    """Save the environment and experiment settings."""
    import platform
    import sklearn
    config = {
        "script_version": "causaltwin_edu_reproducible",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "time_windows": TIME_WINDOWS,
        "intervention_window": INTERVENTION_WINDOW,
        "fast_mode": FAST_MODE,
        "max_intervention_cases": MAX_INTERVENTION_CASES,
        "run_robustness_check": RUN_ROBUSTNESS_CHECK,
        "robustness_seeds": ROBUSTNESS_SEEDS,
        "robustness_models": ROBUSTNESS_MODELS,
        "bootstrap_n": BOOTSTRAP_N,
    }
    (REPORT_DIR / "experiment_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


# -----------------------------------------------------------------------------
# 7. UCI external validation
# -----------------------------------------------------------------------------


def find_dropout_csv() -> Optional[Path]:
    """Find the original UCI dropout/success CSV file."""
    if not DROPOUT_DIR.exists():
        return None

    skip_terms = {"distribution", "summary", "result", "metrics", "readme"}
    candidates = []
    for path in DROPOUT_DIR.rglob("*.csv"):
        name = path.name.lower()
        if any(term in name for term in skip_terms):
            continue
        if path.stat().st_size == 0:
            continue
        for sep in [None, ";"]:
            try:
                sample = pd.read_csv(path, sep=sep, nrows=20) if sep else pd.read_csv(path, nrows=20)
            except Exception:
                continue
            sample.columns = [str(c).strip() for c in sample.columns]
            if "Target" in sample.columns or any(c.lower() == "target" for c in sample.columns):
                candidates.append(path)
                break

    if not candidates:
        return None

    # Prefer the largest matching CSV, which should be the original dataset.
    candidates.sort(key=lambda x: x.stat().st_size, reverse=True)
    return candidates[0]

def dropout_dataset_ready() -> bool:
    return find_dropout_csv() is not None


def validate_dropout_dataset() -> None:
    """Check that the local UCI dropout/success CSV exists."""
    if not DROPOUT_DIR.exists():
        raise FileNotFoundError(
            f"UCI dropout/success directory does not exist: {DROPOUT_DIR.resolve()}\n"
            "Create this folder and place the UCI dropout/success CSV file inside it, "
            "for example data.csv containing the Target column."
        )
    path = find_dropout_csv()
    if path is None:
        found = [str(p.relative_to(DROPOUT_DIR)) for p in DROPOUT_DIR.rglob("*") if p.is_file()]
        raise FileNotFoundError(
            f"No usable UCI dropout/success CSV with a Target column was found under: {DROPOUT_DIR.resolve()}\n"
            "Found files: " + (", ".join(found[:30]) if found else "none")
        )
    print(f"[Data] Local UCI dropout/success dataset found at: {path.resolve()}")


def read_dropout_dataset() -> pd.DataFrame:
    """Read the UCI dropout/success CSV."""
    path = find_dropout_csv()
    if path is None:
        raise FileNotFoundError("UCI dropout/success CSV not found.")
    try:
        df = pd.read_csv(path)
        if len(df.columns) <= 2 and any(";" in str(c) for c in df.columns):
            df = pd.read_csv(path, sep=";")
    except Exception:
        df = pd.read_csv(path, sep=";")
    df.columns = [str(c).strip() for c in df.columns]
    if "Target" not in df.columns:
        # Some mirrors use slightly different casing.
        target_like = [c for c in df.columns if c.strip().lower() == "target"]
        if target_like:
            df = df.rename(columns={target_like[0]: "Target"})
    if "Target" not in df.columns:
        raise ValueError(f"Target column not found in UCI dropout/success data. Columns: {list(df.columns)[:20]}")
    return df


def run_dropout_external_validation(random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Run an external binary dropout-vs-graduate validation experiment.

    The UCI dataset is not temporal, so it is not used for digital-twin intervention.
    It is used as an external validation dataset to show that the prediction backbone
    also works on a different higher-education dropout dataset.
    """
    print("\n" + "-" * 80)
    print("[External] Running UCI dropout/success external validation")
    df = read_dropout_dataset()
    # Binary validation: Dropout = 1, Graduate = 0; Enrolled is excluded.
    df_bin = df[df["Target"].astype(str).isin(["Dropout", "Graduate"])].copy()
    if df_bin.empty:
        raise ValueError("No Dropout/Graduate rows found in UCI dropout/success dataset.")
    y = (df_bin["Target"].astype(str) == "Dropout").astype(int)
    X = df_bin.drop(columns=["Target"])
    # Convert columns carefully and keep non-numeric fields as categorical features.
    X_converted = X.copy()
    for col in X_converted.columns:
        converted_col = pd.to_numeric(X_converted[col], errors="coerce")
        non_missing = X_converted[col].notna()
        # Keep a column categorical if it cannot be fully parsed as numeric.
        if bool(converted_col[non_missing].notna().all()):
            X_converted[col] = converted_col
    X = X_converted

    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=y,
    )
    preprocessor = make_preprocessor(X)
    models = make_model_zoo(preprocessor, random_state=random_state)
    records = []
    for name, model in models.items():
        try:
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_prob = safe_predict_proba(model, X.iloc[test_idx])
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = evaluate_predictions(y.iloc[test_idx], y_pred, y_prob)
            metrics.update({
                "dataset": "UCI Predict Students Dropout and Academic Success",
                "task": "external_binary_dropout_vs_graduate",
                "n_samples": int(len(df_bin)),
                "n_features": int(X.shape[1]),
                "model": name,
                "seed": random_state,
            })
            records.append(metrics)
            print(f"    {name:18s} AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} Recall={metrics['recall']:.4f}")
        except Exception as e:
            print(f"    [External][Warning] Model {name} failed: {e}")
    out = pd.DataFrame(records)

    # Save the class distribution.
    dist = df["Target"].value_counts(dropna=False).rename_axis("target").reset_index(name="count")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dist.to_csv(PROCESSED_DATA_DIR / "uci_dropout_success_target_distribution.csv", index=False, encoding="utf-8-sig")

    # External validation figure.
    if not out.empty:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        out_plot = out.sort_values("auc", ascending=False)
        plt.bar(out_plot["model"], out_plot["auc"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("AUC")
        plt.title("External validation on UCI dropout/success dataset")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "external_validation_uci_dropout_auc.png", dpi=300)
        plt.close()
    return out

# -----------------------------------------------------------------------------
# 8. Plots and output
# -----------------------------------------------------------------------------


def save_plots(metrics: pd.DataFrame, ablation: pd.DataFrame, intervention_cases: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # AUC by week for the full experiment.
    full = metrics[metrics["experiment"] == "full_digital_twin"].copy()
    if not full.empty:
        plt.figure(figsize=(8, 5))
        for model, sub in full.groupby("model"):
            sub = sub.sort_values("week")
            plt.plot(sub["week"], sub["auc"], marker="o", label=model)
        plt.xlabel("Course week")
        plt.ylabel("AUC")
        plt.title("Early academic risk prediction across time windows")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "early_prediction_auc_by_week.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        ct_metrics = full[full["model"] == "CausalTwin-Edu"].sort_values("week")
        if not ct_metrics.empty:
            plt.plot(ct_metrics["week"], ct_metrics["f1"], marker="o", label="F1")
            plt.plot(ct_metrics["week"], ct_metrics["recall"], marker="o", label="Recall")
            plt.plot(ct_metrics["week"], ct_metrics["balanced_accuracy"], marker="o", label="Balanced accuracy")
            plt.xlabel("Course week")
            plt.ylabel("Score")
            plt.title("CausalTwin-Edu metrics across early windows")
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIG_DIR / "causaltwin_metrics_by_week.png", dpi=300)
            plt.close()

    if not ablation.empty:
        plt.figure(figsize=(8, 5))
        for exp, sub in ablation.groupby("experiment"):
            sub = sub[sub["model"] == "CausalTwin-Edu"].sort_values("week")
            if len(sub):
                plt.plot(sub["week"], sub["auc"], marker="o", label=exp)
        plt.xlabel("Course week")
        plt.ylabel("AUC")
        plt.title("Ablation study: effect of temporal digital twin features")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "ablation_auc_by_week.png", dpi=300)
        plt.close()

    if intervention_cases is not None and not intervention_cases.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(intervention_cases["risk_drop"].dropna(), bins=15)
        plt.xlabel("Predicted risk reduction")
        plt.ylabel("Number of cases")
        plt.title("Distribution of counterfactual intervention risk reduction")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "intervention_risk_reduction_hist.png", dpi=300)
        plt.close()


def write_readme(metrics: pd.DataFrame, ablation: pd.DataFrame, intervention_summary: pd.DataFrame, robustness_summary: Optional[pd.DataFrame] = None, quality_check: Optional[pd.DataFrame] = None) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    best = metrics.sort_values("auc", ascending=False).head(10)
    readme = []
    readme.append("CausalTwin-Edu experiment results\n")
    readme.append("=================================\n\n")
    readme.append("Generated files:\n")
    readme.append("- metrics_all_windows.csv: all model metrics at different early windows.\n")
    readme.append("- ablation_results.csv: comparison between full temporal twin features and simplified non-temporal features.\n")
    readme.append("- intervention_cases.csv: high-risk student counterfactual intervention recommendations.\n")
    readme.append("- intervention_summary.csv: summary of accepted recommendations, risk drop, cost, stability and IRI.\n")
    readme.append("- intervention_band_summary.csv: intervention effects stratified by moderate/high/extreme risk bands.\n")
    readme.append("- intervention_action_summary.csv: intervention effects summarized by action family.\n")
    readme.append("- intervention_bootstrap_summary.csv: bootstrap CIs for intervention metrics.\n")
    readme.append("- robustness_raw.csv and robustness_summary.csv: lightweight multi-seed robustness check.\n")
    readme.append("- external_validation_dropout_success.csv: external validation on UCI Predict Students' Dropout and Academic Success.\n")
    readme.append("- final_quality_check.csv: automatic checklist for result strength.\n")
    readme.append("- reports/experiment_config.json: reproducibility snapshot.\n")
    readme.append("- figures/: preliminary figures.\n\n")
    readme.append("Top rows by AUC:\n")
    readme.append(best.to_string(index=False))
    readme.append("\n\nIntervention summary:\n")
    readme.append(intervention_summary.to_string(index=False))
    if robustness_summary is not None and not robustness_summary.empty:
        readme.append("\n\nRobustness summary preview:\n")
        readme.append(robustness_summary.head(12).to_string(index=False))
    if quality_check is not None and not quality_check.empty:
        readme.append("\n\nFinal quality check:\n")
        readme.append(quality_check.to_string(index=False))
    readme.append("\n\nImportant interpretation note:\n")
    readme.append(
        "The intervention module generates model-based counterfactual recommendations. "
        "Because OULAD is observational rather than randomized experimental data, these recommendations "
        "should be described as model-based actionable counterfactual suggestions, not proven causal effects.\n"
    )
    (REPORT_DIR / "README_results.txt").write_text("".join(readme), encoding="utf-8")


# -----------------------------------------------------------------------------
# 9. Main pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    print("=" * 80)
    print("CausalTwin-Edu reproducible experiment")
    print("=" * 80)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    validate_oulad_dataset()
    validate_dropout_dataset()
    tables = read_oulad_tables()

    all_metrics = []
    all_ablation = []

    fitted_for_intervention: Dict[str, Pipeline] = {}
    intervention_data = None
    full_feature_cache: Dict[int, pd.DataFrame] = {}

    # Main temporal-feature experiments.
    for cutoff in TIME_WINDOWS:
        print("\n" + "-" * 80)
        print(f"[Feature] Building FULL digital twin features for day {cutoff} (week {cutoff // 7})")
        df_full = build_feature_table(tables, cutoff, full_digital_twin=True)
        full_feature_cache[cutoff] = df_full
        metrics, fitted, split_idx, meta, y = train_evaluate_window(df_full, cutoff, "full_digital_twin", random_state=RANDOM_STATE)
        all_metrics.append(metrics)

        if cutoff == INTERVENTION_WINDOW:
            X, y_full, groups, meta_full = get_feature_target_groups(df_full)
            train_idx, test_idx = split_idx
            fitted_for_intervention = fitted
            intervention_data = (
                X.iloc[test_idx].copy(),
                y_full.iloc[test_idx].copy(),
                meta_full.iloc[test_idx].copy(),
                X.iloc[train_idx].copy(),
            )

    # Ablation with simpler aggregate features.
    for cutoff in TIME_WINDOWS:
        print("\n" + "-" * 80)
        print(f"[Feature] Building ABLATION non-temporal features for day {cutoff} (week {cutoff // 7})")
        df_ablation = build_feature_table(tables, cutoff, full_digital_twin=False)
        ablation_metrics, _, _, _, _ = train_evaluate_window(df_ablation, cutoff, "without_temporal_twin_features", random_state=RANDOM_STATE)
        all_ablation.append(ablation_metrics)

    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    ablation_df = pd.concat(all_ablation, ignore_index=True) if all_ablation else pd.DataFrame()

    metrics_df.to_csv(CSV_DIR / "metrics_all_windows.csv", index=False, encoding="utf-8-sig")
    ablation_df.to_csv(CSV_DIR / "ablation_results.csv", index=False, encoding="utf-8-sig")

    robustness_raw, robustness_summary = run_seed_robustness(full_feature_cache)
    if not robustness_raw.empty:
        robustness_raw.to_csv(CSV_DIR / "robustness_raw.csv", index=False, encoding="utf-8-sig")
    if not robustness_summary.empty:
        robustness_summary.to_csv(CSV_DIR / "robustness_summary.csv", index=False, encoding="utf-8-sig")

    external_validation_df = run_dropout_external_validation(random_state=RANDOM_STATE)
    if not external_validation_df.empty:
        external_validation_df.to_csv(CSV_DIR / "external_validation_dropout_success.csv", index=False, encoding="utf-8-sig")

    # Intervention analysis on week-4 test students.
    intervention_cases = pd.DataFrame()
    intervention_summary = pd.DataFrame()
    if intervention_data is not None and "CausalTwin-Edu" in fitted_for_intervention:
        X_test, y_test, meta_test, X_train_intervention = intervention_data
        main_model = fitted_for_intervention["CausalTwin-Edu"]
        auxiliary_models = [
            m for name, m in fitted_for_intervention.items()
            if name in ["RandomForest", "ExtraTrees", "LogisticRegression"]
        ]
        intervention_cases, intervention_summary = generate_interventions(
            X_test=X_test,
            y_test=y_test,
            meta_test=meta_test,
            main_model=main_model,
            auxiliary_models=auxiliary_models,
            cutoff_day=INTERVENTION_WINDOW,
            X_train=X_train_intervention,
        )
        intervention_cases.to_csv(CSV_DIR / "intervention_cases.csv", index=False, encoding="utf-8-sig")
        intervention_summary.to_csv(CSV_DIR / "intervention_summary.csv", index=False, encoding="utf-8-sig")
        if not intervention_cases.empty:
            band_summary = intervention_cases.groupby("risk_band").agg(
                n_cases=("risk_band", "size"),
                accepted_rate=("accepted_by_reliability_gate", "mean"),
                mean_risk_before=("risk_before", "mean"),
                mean_risk_after=("risk_after", "mean"),
                mean_risk_drop=("risk_drop", "mean"),
                mean_cost=("intervention_cost", "mean"),
                mean_stability=("stability", "mean"),
                mean_plausibility_score=("plausibility_score", "mean"),
                plausible_rate=("plausibility_gate", "mean"),
                pareto_selected_rate=("on_pareto_front", "mean"),
                mean_IRI=("IRI", "mean"),
            ).reset_index()
            action_summary = intervention_cases.groupby("action_family").agg(
                n_cases=("action_family", "size"),
                accepted_rate=("accepted_by_reliability_gate", "mean"),
                mean_risk_drop=("risk_drop", "mean"),
                mean_cost=("intervention_cost", "mean"),
                mean_stability=("stability", "mean"),
                mean_plausibility_score=("plausibility_score", "mean"),
                plausible_rate=("plausibility_gate", "mean"),
                pareto_selected_rate=("on_pareto_front", "mean"),
                mean_IRI=("IRI", "mean"),
            ).reset_index().sort_values("n_cases", ascending=False)
            band_summary.to_csv(CSV_DIR / "intervention_band_summary.csv", index=False, encoding="utf-8-sig")
            action_summary.to_csv(CSV_DIR / "intervention_action_summary.csv", index=False, encoding="utf-8-sig")
            plausibility_summary = intervention_cases.groupby(["plausibility_gate", "reliability_level"]).agg(
                n_cases=("student_key", "size"),
                mean_risk_drop=("risk_drop", "mean"),
                mean_cost=("intervention_cost", "mean"),
                mean_stability=("stability", "mean"),
                mean_plausibility_score=("plausibility_score", "mean"),
                mean_IRI=("IRI", "mean"),
            ).reset_index()
            plausibility_summary.to_csv(CSV_DIR / "intervention_plausibility_summary.csv", index=False, encoding="utf-8-sig")
            threshold_sensitivity = make_threshold_sensitivity(intervention_cases)
            if not threshold_sensitivity.empty:
                threshold_sensitivity.to_csv(CSV_DIR / "intervention_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
            bootstrap_summary = bootstrap_intervention_summary(intervention_cases)
            if not bootstrap_summary.empty:
                bootstrap_summary.to_csv(CSV_DIR / "intervention_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    else:
        print("[Intervention] Skipped because the week-4 CausalTwin-Edu model was not available.")

    quality_check = make_quality_check(metrics_df, ablation_df, intervention_summary)
    quality_check.to_csv(CSV_DIR / "final_quality_check.csv", index=False, encoding="utf-8-sig")
    write_environment_snapshot()

    # Combined metrics used for plots.
    combined_ablation = pd.concat([metrics_df, ablation_df], ignore_index=True)
    save_plots(metrics_df, combined_ablation, intervention_cases)
    write_readme(metrics_df, ablation_df, intervention_summary, robustness_summary=robustness_summary, quality_check=quality_check)

    print("\n" + "=" * 80)
    print("Experiment finished.")
    print(f"Results saved to: {RESULT_DIR.resolve()}")
    print("Key files:")
    print(f"  - {CSV_DIR / 'metrics_all_windows.csv'}")
    print(f"  - {CSV_DIR / 'ablation_results.csv'}")
    print(f"  - {CSV_DIR / 'intervention_cases.csv'}")
    print(f"  - {CSV_DIR / 'intervention_summary.csv'}")
    print(f"  - {CSV_DIR / 'intervention_band_summary.csv'}")
    print(f"  - {CSV_DIR / 'intervention_action_summary.csv'}")
    print(f"  - {CSV_DIR / 'intervention_bootstrap_summary.csv'}")
    print(f"  - {CSV_DIR / 'intervention_plausibility_summary.csv'}")
    print(f"  - {CSV_DIR / 'intervention_threshold_sensitivity.csv'}")
    print(f"  - {CSV_DIR / 'robustness_summary.csv'}")
    print(f"  - {CSV_DIR / 'external_validation_dropout_success.csv'}")
    print(f"  - {CSV_DIR / 'final_quality_check.csv'}")
    print(f"  - {REPORT_DIR / 'README_results.txt'}")
    print(f"  - {REPORT_DIR / 'experiment_config.json'}")
    print(f"  - {FIG_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
