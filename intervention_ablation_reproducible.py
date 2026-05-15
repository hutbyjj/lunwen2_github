from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


METHOD_ORDER = [
    "Random actionable",
    "Risk-drop-only",
    "w/o plausibility gate",
    "w/o Pareto selection",
    "Full CausalTwin-Edu",
]


def load_experiment_module():
    """Import the main experiment file without triggering its main routine."""
    here = Path(__file__).resolve().parent
    main_file = here / "causaltwin_edu_reproducible.py"
    if not main_file.exists():
        raise FileNotFoundError(
            f"Cannot find {main_file}. Put this script next to causaltwin_edu_reproducible.py."
        )

    spec = importlib.util.spec_from_file_location("causaltwin_main", str(main_file))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def level_rank(level: str) -> int:
    ranks = {
        "HighReliability": 3,
        "ModerateCandidate": 2,
        "ManualReviewCandidate": 1,
        "NoImprovement": 0,
    }
    return ranks.get(level, 0)


def entropy_from_counts(counts: pd.Series) -> float:
    if counts.empty:
        return float("nan")
    shares = counts.astype(float) / max(float(counts.sum()), 1.0)
    shares = shares[shares > 0]
    return float(-(shares * np.log(shares)).sum())


def summarize_cases(cases: pd.DataFrame) -> pd.DataFrame:
    """Summarize the per-case recommendations for each ablation method."""
    rows = []
    for method_name, method_cases in cases.groupby("ablation_method"):
        family_counts = method_cases["action_family"].value_counts()
        rows.append({
            "ablation_method": method_name,
            "n_cases": int(len(method_cases)),
            "accepted_rate": float(method_cases["accepted_by_reliability_gate"].mean()),
            "high_reliability_rate": float((method_cases["reliability_level"] == "HighReliability").mean()),
            "manual_review_rate": float((method_cases["reliability_level"] == "ManualReviewCandidate").mean()),
            "no_improvement_rate": float((method_cases["reliability_level"] == "NoImprovement").mean()),
            "mean_risk_before": float(method_cases["risk_before"].mean()),
            "mean_risk_after": float(method_cases["risk_after"].mean()),
            "mean_risk_drop": float(method_cases["risk_drop"].mean()),
            "median_risk_drop": float(method_cases["risk_drop"].median()),
            "mean_relative_risk_drop": float(method_cases["relative_risk_drop"].mean()),
            "mean_cost": float(method_cases["intervention_cost"].mean()),
            "mean_stability": float(method_cases["stability"].mean()),
            "mean_plausibility_score": float(method_cases["plausibility_score"].mean()),
            "plausible_recommendation_rate": float(method_cases["plausibility_gate"].mean()),
            "pareto_selected_rate": float(method_cases["on_pareto_front"].mean()),
            "mean_auxiliary_risk_drop": float(method_cases["mean_auxiliary_risk_drop"].mean()),
            "mean_IRI": float(method_cases["IRI"].mean()),
            "action_family_count": int(method_cases["action_family"].nunique()),
            "action_family_entropy": entropy_from_counts(family_counts),
        })

    summary = pd.DataFrame(rows)
    summary["_order"] = summary["ablation_method"].apply(
        lambda name: METHOD_ORDER.index(name) if name in METHOD_ORDER else 99
    )
    return summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def build_candidate_table(
    exp,
    original: pd.Series,
    actions: List[Dict[str, float]],
    cutoff_day: int,
    main_model,
    auxiliary_models,
    plaus_ref,
) -> pd.DataFrame:
    """Evaluate every candidate intervention for a single student."""
    changed_rows = [exp.apply_action(original, action, cutoff_day) for action in actions]
    changed_data = pd.DataFrame(changed_rows)[original.index]

    risk_before = float(exp.safe_predict_proba(main_model, pd.DataFrame([original]))[0])
    risk_after = exp.safe_predict_proba(main_model, changed_data)
    risk_drop = risk_before - risk_after
    costs = np.array([exp.action_cost(action) for action in actions], dtype=float)
    families = np.array([exp.action_family(action) for action in actions])
    plaus_dist, plaus_score, plaus_gate = exp._plausibility_scores(main_model, changed_data, plaus_ref)

    aux_drop_rows = []
    aux_success_rows = []
    for aux_model in auxiliary_models:
        try:
            aux_before = float(exp.safe_predict_proba(aux_model, pd.DataFrame([original]))[0])
            aux_after = exp.safe_predict_proba(aux_model, changed_data)
            aux_drop = aux_before - aux_after
            aux_drop_rows.append(aux_drop)
            aux_success_rows.append((aux_drop >= 0.020) | (aux_drop >= 0.06 * max(aux_before, 1e-6)))
        except Exception:
            continue

    if aux_success_rows:
        stability = np.vstack(aux_success_rows).mean(axis=0)
        mean_aux_drop = np.vstack(aux_drop_rows).mean(axis=0)
    else:
        stability = np.zeros(len(actions), dtype=float)
        mean_aux_drop = np.zeros(len(actions), dtype=float)

    pareto_mask = exp._pareto_front_mask(risk_after, costs, stability, plaus_score)
    thresholds = exp._risk_thresholds()
    feasibility = 1.0 - np.clip(costs, 0.0, 1.0)
    positive_drop = np.maximum(risk_drop, 0.0)
    iri = (
        (positive_drop / (costs + 0.15))
        * (0.45 + 0.55 * stability)
        * (0.55 + 0.45 * feasibility)
        * (0.45 + 0.55 * plaus_score)
    )

    base_levels = []
    plaus_levels = []
    for i in range(len(actions)):
        base_level = exp.classify_reliability(
            risk_before=risk_before,
            risk_after=float(risk_after[i]),
            drop=float(risk_drop[i]),
            cost=float(costs[i]),
            stability=float(stability[i]),
            thresholds=thresholds,
        )
        base_levels.append(base_level)

        if plaus_score[i] < 0.35 or not plaus_gate[i]:
            if risk_drop[i] >= thresholds["candidate_min_drop"] and costs[i] <= thresholds["max_candidate_cost"]:
                plaus_levels.append("ManualReviewCandidate")
            else:
                plaus_levels.append("NoImprovement")
        elif plaus_score[i] < 0.60 and base_level == "HighReliability":
            plaus_levels.append("ModerateCandidate")
        else:
            plaus_levels.append(base_level)

    rows = []
    for i, action in enumerate(actions):
        rows.append({
            "idx": i,
            "risk_before": risk_before,
            "risk_after": float(risk_after[i]),
            "risk_drop": float(risk_drop[i]),
            "relative_risk_drop": float(risk_drop[i] / max(risk_before, 1e-6)),
            "intervention_cost": float(costs[i]),
            "stability": float(stability[i]),
            "plausibility_distance": float(plaus_dist[i]),
            "plausibility_score": float(plaus_score[i]),
            "plausibility_gate": bool(plaus_gate[i]),
            "on_pareto_front": bool(pareto_mask[i]),
            "mean_auxiliary_risk_drop": float(mean_aux_drop[i]),
            "IRI": float(iri[i]),
            "base_reliability_level": str(base_levels[i]),
            "plausibility_aware_level": str(plaus_levels[i]),
            "action_family": str(families[i]),
            "action_description": exp.describe_action(action),
            **action,
        })
    return pd.DataFrame(rows)


def choose_candidate(
    exp,
    candidates: pd.DataFrame,
    method: str,
    rng: np.random.Generator,
    family_counts: Dict[str, int],
) -> pd.Series:
    """Pick one candidate according to the requested ablation setting."""
    if candidates.empty:
        raise ValueError("Empty candidate table")

    if method == "Random actionable":
        pool = candidates[candidates["plausibility_gate"] & (candidates["intervention_cost"] <= 1.0)]
        if pool.empty:
            pool = candidates
        return pool.iloc[int(rng.integers(0, len(pool)))]

    if method == "Risk-drop-only":
        return candidates.sort_values(["risk_drop", "risk_after"], ascending=[False, True]).iloc[0]

    total_selected = max(1, sum(family_counts.values()))
    diversity_penalty = candidates["action_family"].map(
        lambda family: family_counts.get(str(family), 0) / total_selected
    ).to_numpy(float)
    assessment_penalty = candidates["action_family"].map(
        lambda family: 0.20 if family == "assessment_participation" else 0.0
    ).to_numpy(float)

    if method == "w/o plausibility gate":
        levels = candidates["base_reliability_level"].to_numpy(str)
        plausibility = np.ones(len(candidates), dtype=float)
        plaus_gate = np.ones(len(candidates), dtype=bool)
    else:
        levels = candidates["plausibility_aware_level"].to_numpy(str)
        plausibility = candidates["plausibility_score"].to_numpy(float)
        plaus_gate = candidates["plausibility_gate"].to_numpy(bool)

    ranks = np.array([level_rank(level) for level in levels], dtype=float)
    risk_after = candidates["risk_after"].to_numpy(float)
    positive_drop = np.maximum(candidates["risk_drop"].to_numpy(float), 0.0)
    cost = candidates["intervention_cost"].to_numpy(float)
    stability = candidates["stability"].to_numpy(float)
    feasibility = 1.0 - np.clip(cost, 0.0, 1.0)
    pareto_mask = candidates["on_pareto_front"].to_numpy(bool)
    iri = candidates["IRI"].to_numpy(float)

    pareto_bonus = pareto_mask.astype(float)
    if method == "w/o Pareto selection":
        pareto_bonus = np.zeros_like(pareto_bonus)

    score = (
        1.15 * ranks
        + 1.85 * positive_drop
        + 0.65 * stability
        + 0.45 * feasibility
        + 0.65 * plausibility
        + 0.30 * pareto_bonus
        + 0.15 * iri
        - 0.20 * risk_after
        - 0.40 * diversity_penalty
        - assessment_penalty
    )

    acceptable = ranks >= 1
    if method == "w/o Pareto selection":
        preferred = acceptable & plaus_gate & (plausibility >= 0.35)
    elif method == "w/o plausibility gate":
        preferred = acceptable & pareto_mask
    else:
        preferred = acceptable & pareto_mask & plaus_gate & (plausibility >= 0.35)

    if preferred.any():
        score = np.where(preferred, score, -np.inf)

    selected = candidates.iloc[int(np.nanargmax(score))].copy()
    if method == "w/o plausibility gate":
        selected["plausibility_aware_level"] = selected["base_reliability_level"]
    return selected


def plot_ablation(summary: pd.DataFrame, fig_path: Path) -> None:
    """Save the grouped bar chart used in the manuscript."""
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    methods = summary["ablation_method"].tolist()
    x_pos = np.arange(len(methods))
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x_pos - 1.5 * bar_width, summary["mean_risk_drop"], bar_width, label="Risk drop")
    ax.bar(x_pos - 0.5 * bar_width, summary["mean_stability"], bar_width, label="Stability")
    ax.bar(x_pos + 0.5 * bar_width, summary["mean_plausibility_score"], bar_width, label="Plausibility")
    ax.bar(x_pos + 1.5 * bar_width, summary["mean_IRI"], bar_width, label="IRI")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Metric value")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    exp = load_experiment_module()
    exp.CSV_DIR.mkdir(parents=True, exist_ok=True)
    exp.FIG_DIR.mkdir(parents=True, exist_ok=True)
    exp.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CausalTwin-Edu supplementary intervention ablation study")
    print("=" * 80)

    exp.validate_oulad_dataset()
    tables = exp.read_oulad_tables()
    cutoff_day = getattr(exp, "INTERVENTION_WINDOW", 28)
    print(f"[Ablation] Building week-{cutoff_day // 7} full digital twin features...")
    feature_table = exp.build_feature_table(tables, cutoff_day=cutoff_day, full_digital_twin=True)

    X, y, groups, meta = exp.get_feature_target_groups(feature_table)
    train_idx, test_idx = exp.group_train_test_split(X, y, groups, random_state=exp.RANDOM_STATE)
    preprocessor = exp.make_preprocessor(X)
    model_zoo = exp.make_model_zoo(preprocessor, random_state=exp.RANDOM_STATE)
    model_zoo = {name: model for name, model in model_zoo.items() if name in {"LogisticRegression", "ExtraTrees", "CausalTwin-Edu"}}

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    meta_test = meta.iloc[test_idx].reset_index(drop=True)

    print("[Ablation] Training risk estimator and auxiliary models...")
    fitted_models = {}
    for model_name, model in model_zoo.items():
        print(f"    Training {model_name}...")
        model.fit(X_train, y_train)
        fitted_models[model_name] = model

    main_model = fitted_models["CausalTwin-Edu"]
    auxiliary_models = [
        fitted_models[name]
        for name in ["LogisticRegression", "ExtraTrees"]
        if name in fitted_models
    ]

    risk_prob = exp.safe_predict_proba(main_model, X_test)
    max_cases = min(getattr(exp, "MAX_INTERVENTION_CASES", 90), len(X_test))
    selected_idx = exp.select_intervention_indices(risk_prob, max_cases)
    if len(selected_idx) == 0:
        selected_idx = np.argsort(-risk_prob)[:max_cases]
    print(f"[Ablation] Selected {len(selected_idx)} risk-stratified intervention cases.")

    actions = exp.intervention_action_space()
    plaus_ref = exp._fit_plausibility_reference(main_model, X_train)
    rng = np.random.default_rng(exp.RANDOM_STATE)
    family_counts: Dict[str, Dict[str, int]] = {method: {} for method in METHOD_ORDER}
    case_rows = []

    for case_no, test_pos in enumerate(selected_idx, start=1):
        test_pos = int(test_pos)
        original_row = X_test.iloc[test_pos]
        meta_row = meta_test.iloc[test_pos].to_dict()
        risk_before = float(risk_prob[test_pos])
        candidates = build_candidate_table(
            exp,
            original_row,
            actions,
            cutoff_day,
            main_model,
            auxiliary_models,
            plaus_ref,
        )

        for method in METHOD_ORDER:
            selected = choose_candidate(exp, candidates, method, rng, family_counts[method])
            family = str(selected["action_family"])
            family_counts[method][family] = family_counts[method].get(family, 0) + 1
            reliability = str(
                selected.get("plausibility_aware_level", selected.get("base_reliability_level", "NoImprovement"))
            )

            row = {
                "ablation_method": method,
                "case_no": case_no,
                **meta_row,
                "risk_band": exp.risk_band_from_prob(risk_before),
                "true_risk_label": int(y_test.iloc[test_pos]),
                "reliability_level": reliability,
                "accepted_by_reliability_gate": bool(reliability in ["HighReliability", "ModerateCandidate"]),
                "candidate_for_manual_review": bool(reliability == "ManualReviewCandidate"),
            }
            for col in selected.index:
                row[col] = selected[col]
            case_rows.append(row)

        if case_no == 1 or case_no % 5 == 0 or case_no == len(selected_idx):
            print(f"[Ablation] {case_no}/{len(selected_idx)} cases evaluated.")

    cases = pd.DataFrame(case_rows)
    summary = summarize_cases(cases)

    cases_path = exp.CSV_DIR / "intervention_ablation_cases.csv"
    summary_path = exp.CSV_DIR / "intervention_ablation_results.csv"
    fig_path = exp.FIG_DIR / "intervention_ablation_comparison.png"
    cases.to_csv(cases_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plot_ablation(summary, fig_path)

    print("\n[Ablation] Summary:")
    print(summary.to_string(index=False))
    print("\nSaved files:")
    print(f"  - {cases_path}")
    print(f"  - {summary_path}")
    print(f"  - {fig_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
