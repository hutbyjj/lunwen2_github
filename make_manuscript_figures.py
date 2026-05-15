from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "results_applied_intelligence" / "csv"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "submission_figures_clean"


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.linewidth": 0.8,
})


MODEL_ORDER_MAIN = [
    "CausalTwin-Edu",
    "ExtraTrees",
    "LogisticRegression",
    "MLP",
    "RandomForest",
]

MODEL_ORDER_GBDT = [
    "CausalTwin-Edu",
    "LogisticRegression",
    "ExtraTrees",
    "LightGBM",
]

MODEL_ORDER_EXTERNAL = [
    "LogisticRegression",
    "RandomForest",
    "ExtraTrees",
    "MLP",
    "LightGBM",
    "CausalTwin-Edu",
]


def find_latest_variant(data_dir: Path, patterns: Iterable[str]) -> Path:
    """Pick an exact CSV or the newest numbered copy such as file(4).csv."""
    matches: list[tuple[int, Path]] = []

    for filename in patterns:
        direct_path = data_dir / filename
        if direct_path.exists():
            matches.append((999999, direct_path))

        stem = Path(filename).stem
        suffix = Path(filename).suffix
        numbered = re.compile(rf"^{re.escape(stem)}\((\d+)\){re.escape(suffix)}$")

        for candidate_path in data_dir.iterdir():
            if not candidate_path.is_file():
                continue
            match = numbered.match(candidate_path.name)
            if match:
                matches.append((int(match.group(1)), candidate_path))

    if not matches:
        raise FileNotFoundError(
            f"Could not find any of these files in {data_dir}: {list(patterns)}"
        )

    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def savefig(fig: plt.Figure, out_path: Path, dpi: int = 600) -> None:
    """Save one figure with a little padding so labels are not clipped."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.2)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[Saved] {out_path}")


def pretty_label(label: str) -> str:
    labels = {
        "full_digital_twin": "Full digital twin",
        "without_temporal_twin_features": "Without temporal twin features",
        "moderate_risk": "Moderate risk",
        "high_risk": "High risk",
        "extreme_risk": "Extreme risk",
        "learning_regularization": "Learning regularization",
        "assessment_participation": "Assessment participation",
        "multi_component_package": "Multi-component package",
        "engagement_timeliness_package": "Engagement timeliness package",
        "balanced_assessment_engagement": "Balanced assessment engagement",
        "regular_engagement_package": "Regular engagement package",
    }
    return labels.get(label, label.replace("_", " "))


def wrap_label(label: str, width: int = 14) -> str:
    return "\n".join(textwrap.wrap(str(label), width=width, break_long_words=False))


def set_clean_axis(ax) -> None:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(axis="both", which="major", labelsize=10)


def draw_box(ax, xy, width, height, text, fontsize=10.5) -> None:
    box = Rectangle(xy, width, height, fill=False, linewidth=1.1)
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def draw_arrow(ax, start, end) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.1))


def create_fig1_framework(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    draw_box(ax, (3, 34), 16, 10, "Raw educational data\nOULAD / external dataset")
    draw_box(ax, (23, 34), 18, 10, "Dynamic student digital\ntwin state\nVLE, assessment, trend,\ncohort deviation")
    draw_box(ax, (45, 34), 15, 10, "Multi-window risk\nestimation\nWeeks 2, 4, 8, 12")
    draw_box(ax, (65, 34), 14, 10, "Risk stratification\nmoderate / high /\nextreme")
    draw_box(ax, (83, 34), 16, 10, "Actionable intervention\nrecommendation")

    draw_box(ax, (23, 8), 18, 10, "Education-specific\nactionable space\nimmutable, historical,\nfuture-actionable")
    draw_box(ax, (45, 8), 15, 10, "Plausibility gate\ntraining-manifold\nproximity")
    draw_box(ax, (65, 8), 14, 10, "Pareto selection\nrisk, cost, stability,\nplausibility")
    draw_box(ax, (83, 8), 16, 10, "Reliability output\nIRI, action family,\nreview level")

    draw_arrow(ax, (19, 39), (23, 39))
    draw_arrow(ax, (41, 39), (45, 39))
    draw_arrow(ax, (60, 39), (65, 39))
    draw_arrow(ax, (79, 39), (83, 39))
    draw_arrow(ax, (91, 34), (91, 18))
    draw_arrow(ax, (83, 13), (79, 13))
    draw_arrow(ax, (65, 13), (60, 13))
    draw_arrow(ax, (45, 13), (41, 13))
    draw_arrow(ax, (32, 18), (32, 34))

    savefig(fig, out_path)


def create_fig2_pareto(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    draw_box(ax, (3, 32), 16, 10, "Candidate action set")
    draw_box(ax, (23, 32), 18, 10, "Generate counterfactual\nstudent states")
    draw_box(ax, (45, 32), 18, 10, "Evaluate four objectives")
    draw_box(ax, (68, 32), 14, 10, "Pareto filtering")
    draw_box(ax, (86, 32), 12, 10, "Final\nrecommendation")

    draw_box(ax, (42, 6), 24, 12, "Objectives:\nmin post-risk\nmin intervention cost\nmax cross-model stability\nmax plausibility")
    draw_box(ax, (69, 6), 22, 12, "Reliability labeling:\nHighReliability\nModerateCandidate\nManualReviewCandidate")

    draw_arrow(ax, (19, 37), (23, 37))
    draw_arrow(ax, (41, 37), (45, 37))
    draw_arrow(ax, (63, 37), (68, 37))
    draw_arrow(ax, (82, 37), (86, 37))
    draw_arrow(ax, (54, 32), (54, 18))
    draw_arrow(ax, (77, 32), (80, 18))

    savefig(fig, out_path)


def create_fig3_early_auc(metrics_df: pd.DataFrame, out_path: Path) -> None:
    plot_data = metrics_df.copy()
    plot_data = plot_data[plot_data["experiment"] == "full_digital_twin"]
    plot_data = plot_data[plot_data["model"].isin(MODEL_ORDER_MAIN)]

    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    for model_name in MODEL_ORDER_MAIN:
        model_data = plot_data[plot_data["model"] == model_name].sort_values("week")
        if not model_data.empty:
            ax.plot(model_data["week"], model_data["auc"], marker="o", linewidth=1.8, label=model_name)
    ax.set_xlabel("Course week")
    ax.set_ylabel("AUC")
    ax.set_xticks([2, 4, 8, 12])
    ax.legend(frameon=True)
    set_clean_axis(ax)
    savefig(fig, out_path)


def create_fig4_cte_metrics(metrics_df: pd.DataFrame, out_path: Path) -> None:
    plot_data = metrics_df.copy()
    plot_data = plot_data[(plot_data["experiment"] == "full_digital_twin") & (plot_data["model"] == "CausalTwin-Edu")]
    plot_data = plot_data.sort_values("week")

    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    ax.plot(plot_data["week"], plot_data["f1"], marker="o", linewidth=1.8, label="F1")
    ax.plot(plot_data["week"], plot_data["recall"], marker="o", linewidth=1.8, label="Recall")
    ax.plot(plot_data["week"], plot_data["balanced_accuracy"], marker="o", linewidth=1.8, label="Balanced accuracy")
    ax.set_xlabel("Course week")
    ax.set_ylabel("Score")
    ax.set_xticks([2, 4, 8, 12])
    ax.legend(frameon=True)
    set_clean_axis(ax)
    savefig(fig, out_path)


def create_fig5_robustness(robust_df: pd.DataFrame, out_path: Path) -> None:
    model_order = ["CausalTwin-Edu", "ExtraTrees", "LogisticRegression"]
    plot_data = robust_df.copy()
    plot_data = plot_data[plot_data["model"].isin(model_order)]

    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    for model_name in model_order:
        model_data = plot_data[plot_data["model"] == model_name].sort_values("week")
        if not model_data.empty:
            ax.errorbar(
                model_data["week"],
                model_data["auc_mean"],
                yerr=model_data["auc_std"],
                marker="o",
                linewidth=1.8,
                capsize=4,
                label=model_name,
            )
    ax.set_xlabel("Learning week")
    ax.set_ylabel("AUC mean ± std")
    ax.set_xticks([2, 4, 8, 12])
    ax.legend(frameon=True)
    set_clean_axis(ax)
    savefig(fig, out_path)


def create_fig6_temporal_ablation(metrics_df: pd.DataFrame, ablation_df: pd.DataFrame, out_path: Path) -> None:
    full_data = metrics_df.copy()
    full_data = full_data[(full_data["experiment"] == "full_digital_twin") & (full_data["model"] == "CausalTwin-Edu")]
    full_data = full_data.sort_values("week")

    ablation_data = ablation_df.copy()
    ablation_data = ablation_data[(ablation_data["experiment"] == "without_temporal_twin_features") & (ablation_data["model"] == "CausalTwin-Edu")]
    ablation_data = ablation_data.sort_values("week")

    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    ax.plot(full_data["week"], full_data["auc"], marker="o", linewidth=1.8, label="Full digital twin")
    ax.plot(ablation_data["week"], ablation_data["auc"], marker="o", linewidth=1.8, label="Without temporal twin features")
    ax.set_xlabel("Course week")
    ax.set_ylabel("AUC")
    ax.set_xticks([2, 4, 8, 12])
    ax.legend(frameon=True)
    set_clean_axis(ax)
    savefig(fig, out_path)


def create_fig7_external_validation(external_df: pd.DataFrame, out_path: Path) -> None:
    plot_data = external_df.copy()
    order = [model for model in MODEL_ORDER_EXTERNAL if model in set(plot_data["model"])]
    plot_data["_order"] = plot_data["model"].map({model: idx for idx, model in enumerate(order)})
    plot_data = plot_data.sort_values("_order")

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    bars = ax.bar(range(len(plot_data)), plot_data["auc"])
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels([wrap_label(model, width=13) for model in plot_data["model"]], rotation=0, ha="center")
    ax.set_ylabel("AUC")
    ax.set_ylim(max(0, plot_data["auc"].min() - 0.03), min(1.0, plot_data["auc"].max() + 0.015))
    for bar, value in zip(bars, plot_data["auc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    set_clean_axis(ax)
    savefig(fig, out_path)


def create_fig8_risk_drop_hist(cases_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    ax.hist(cases_df["risk_drop"].dropna(), bins=20)
    ax.set_xlabel("Predicted risk reduction")
    ax.set_ylabel("Number of cases")
    set_clean_axis(ax)
    savefig(fig, out_path)


def create_fig9_risk_band_effect(band_df: pd.DataFrame, out_path: Path) -> None:
    plot_data = band_df.copy()
    band_order = ["moderate_risk", "high_risk", "extreme_risk"]
    plot_data["_order"] = plot_data["risk_band"].map({band: idx for idx, band in enumerate(band_order)})
    plot_data = plot_data.sort_values("_order")

    labels = [pretty_label(label) for label in plot_data["risk_band"]]
    fig, ax = plt.subplots(figsize=(8.8, 5.7))
    bars = ax.bar(labels, plot_data["mean_risk_drop"])
    ax.set_xlabel("Risk band")
    ax.set_ylabel("Mean predicted risk reduction")
    ax.set_ylim(0, plot_data["mean_risk_drop"].max() + 0.03)
    for bar, value in zip(bars, plot_data["mean_risk_drop"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, f"{value:.3f}", ha="center", va="bottom")
    set_clean_axis(ax)
    savefig(fig, out_path)


def create_fig10_action_family(action_df: pd.DataFrame, out_path: Path) -> None:
    plot_data = action_df.copy().sort_values("mean_risk_drop", ascending=True)
    labels = [pretty_label(label) for label in plot_data["action_family"]]

    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    bars = ax.barh(labels, plot_data["mean_risk_drop"])
    ax.set_xlabel("Mean predicted risk reduction")
    ax.set_ylabel("Action family")
    ax.set_xlim(0, plot_data["mean_risk_drop"].max() + 0.035)
    for bar, value in zip(bars, plot_data["mean_risk_drop"]):
        ax.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=10)
    set_clean_axis(ax)
    savefig(fig, out_path)


def create_fig11_threshold_heatmap(th_df: pd.DataFrame, out_path: Path) -> None:
    plot_data = th_df.copy()
    if "min_plausibility" in plot_data.columns:
        strictest_plausibility = sorted(plot_data["min_plausibility"].dropna().unique())[-1]
        plot_data = plot_data[plot_data["min_plausibility"] == strictest_plausibility]

    heatmap_data = plot_data.pivot(index="min_drop", columns="min_stability", values="accepted_rate")
    heatmap_data = heatmap_data.sort_index(ascending=True)
    heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)

    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    image = ax.imshow(heatmap_data.values, aspect="auto")
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels([f"{value:.2f}" for value in heatmap_data.columns], rotation=0)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels([f"{value:.2f}" for value in heatmap_data.index])
    ax.set_xlabel("Minimum stability threshold")
    ax.set_ylabel("Minimum risk-drop threshold")

    for row_idx in range(heatmap_data.shape[0]):
        for col_idx in range(heatmap_data.shape[1]):
            ax.text(col_idx, row_idx, f"{heatmap_data.values[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=10)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Accepted rate")
    savefig(fig, out_path)


def create_fig12_intervention_ablation(ab_df: pd.DataFrame, out_path: Path) -> None:
    plot_data = ab_df.copy()
    method_order = [
        "Random actionable",
        "Risk-drop-only",
        "w/o plausibility gate",
        "w/o Pareto selection",
        "Full CausalTwin-Edu",
    ]
    plot_data["_order"] = plot_data["ablation_method"].map({method: idx for idx, method in enumerate(method_order)})
    plot_data = plot_data.sort_values("_order")

    labels = [wrap_label(method, width=16) for method in plot_data["ablation_method"]]
    x_pos = np.arange(len(plot_data))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12.8, 6.7))
    ax.bar(x_pos - 1.5 * width, plot_data["mean_risk_drop"], width, label="Risk drop")
    ax.bar(x_pos - 0.5 * width, plot_data["mean_stability"], width, label="Stability")
    ax.bar(x_pos + 0.5 * width, plot_data["mean_plausibility_score"], width, label="Plausibility")
    ax.bar(x_pos + 1.5 * width, plot_data["mean_IRI"], width, label="IRI")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=True, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    set_clean_axis(ax)
    savefig(fig, out_path)


def create_optional_supfig_oulad_gbdt(metrics_gbdt_df: pd.DataFrame, out_path: Path) -> None:
    plot_data = metrics_gbdt_df.copy()
    plot_data = plot_data[plot_data["experiment"] == "full_digital_twin"]
    plot_data = plot_data[plot_data["model"].isin(MODEL_ORDER_GBDT)]

    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    for model_name in MODEL_ORDER_GBDT:
        model_data = plot_data[plot_data["model"] == model_name].sort_values("week")
        if not model_data.empty:
            ax.plot(model_data["week"], model_data["auc"], marker="o", linewidth=1.8, label=model_name)
    ax.set_xlabel("Week")
    ax.set_ylabel("AUC")
    ax.set_xticks([2, 4, 8, 12])
    ax.legend(title="Model", frameon=True)
    set_clean_axis(ax)
    savefig(fig, out_path)


def create_optional_supfig_external_gbdt(external_df: pd.DataFrame, out_path: Path) -> None:
    model_order = [model for model in ["CausalTwin-Edu", "LogisticRegression", "LightGBM", "RandomForest", "ExtraTrees", "MLP"] if model in set(external_df["model"])]
    plot_data = external_df.copy()
    plot_data["_order"] = plot_data["model"].map({model: idx for idx, model in enumerate(model_order)})
    plot_data = plot_data.sort_values("_order")

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    bars = ax.bar(range(len(plot_data)), plot_data["auc"])
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels([wrap_label(model, width=13) for model in plot_data["model"]], rotation=0, ha="center")
    ax.set_ylabel("AUC")
    ax.set_ylim(max(0, plot_data["auc"].min() - 0.03), min(1.0, plot_data["auc"].max() + 0.015))
    for bar, value in zip(bars, plot_data["auc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    set_clean_axis(ax)
    savefig(fig, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate manuscript figures from saved CSV result files."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Folder containing generated CSV result files.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output folder for the regenerated figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    main_fig_dir = output_root / "main_manuscript_figures"
    extra_fig_dir = output_root / "optional_extra_figures"
    main_fig_dir.mkdir(parents=True, exist_ok=True)
    extra_fig_dir.mkdir(parents=True, exist_ok=True)

    result_paths = {
        "metrics_main": find_latest_variant(data_dir, ["metrics_all_windows.csv", "metrics_all_windows(4).csv"]),
        "metrics_gbdt": find_latest_variant(data_dir, ["metrics_all_windows_with_gbdt.csv", "metrics_all_windows.csv"]),
        "temporal_ablation": find_latest_variant(data_dir, ["ablation_results.csv", "ablation_results(8).csv"]),
        "robustness": find_latest_variant(data_dir, ["robustness_summary.csv", "robustness_summary(1).csv"]),
        "external": find_latest_variant(data_dir, ["external_validation_with_gbdt.csv", "external_validation_dropout_success.csv"]),
        "cases": find_latest_variant(data_dir, ["intervention_cases.csv", "intervention_cases(4).csv"]),
        "risk_band": find_latest_variant(data_dir, ["intervention_band_summary.csv", "intervention_band_summary(2).csv"]),
        "action_family": find_latest_variant(data_dir, ["intervention_action_summary.csv", "intervention_action_summary(2).csv"]),
        "threshold": find_latest_variant(data_dir, ["intervention_threshold_sensitivity.csv"]),
        "intervention_ablation": find_latest_variant(data_dir, ["intervention_ablation_results.csv"]),
    }

    print("[Using files]")
    for path in result_paths.values():
        print(" -", path.name)

    metrics_main = pd.read_csv(result_paths["metrics_main"])
    metrics_gbdt = pd.read_csv(result_paths["metrics_gbdt"])
    temporal_ablation = pd.read_csv(result_paths["temporal_ablation"])
    robustness = pd.read_csv(result_paths["robustness"])
    external = pd.read_csv(result_paths["external"])
    cases = pd.read_csv(result_paths["cases"])
    risk_band = pd.read_csv(result_paths["risk_band"])
    action_family = pd.read_csv(result_paths["action_family"])
    threshold = pd.read_csv(result_paths["threshold"])
    intervention_ablation = pd.read_csv(result_paths["intervention_ablation"])

    create_fig1_framework(main_fig_dir / "Fig1.png")
    create_fig2_pareto(main_fig_dir / "Fig2.png")
    create_fig3_early_auc(metrics_main, main_fig_dir / "Fig3.png")
    create_fig4_cte_metrics(metrics_main, main_fig_dir / "Fig4.png")
    create_fig5_robustness(robustness, main_fig_dir / "Fig5.png")
    create_fig6_temporal_ablation(metrics_main, temporal_ablation, main_fig_dir / "Fig6.png")
    create_fig7_external_validation(external, main_fig_dir / "Fig7.png")
    create_fig8_risk_drop_hist(cases, main_fig_dir / "Fig8.png")
    create_fig9_risk_band_effect(risk_band, main_fig_dir / "Fig9.png")
    create_fig10_action_family(action_family, main_fig_dir / "Fig10.png")
    create_fig11_threshold_heatmap(threshold, main_fig_dir / "Fig11.png")
    create_fig12_intervention_ablation(intervention_ablation, main_fig_dir / "Fig12.png")

    create_optional_supfig_oulad_gbdt(metrics_gbdt, extra_fig_dir / "SupFigA_oulad_auc_with_gbdt.png")
    create_optional_supfig_external_gbdt(external, extra_fig_dir / "SupFigB_external_auc_with_gbdt.png")

    readme_path = output_root / "README_clean_figures.txt"
    readme_path.write_text(
        "This folder contains regenerated figures for submission. The figures are generated from local CSV results; no model is retrained here.\n"
        "- main_manuscript_figures/: the 12 main figures (Fig1-Fig12).\n"
        "- optional_extra_figures/: additional comparison figures kept separate from the main figures.\n"
        "The figures have no internal titles, and long labels are wrapped where needed.\n",
        encoding="utf-8",
    )
    print(f"[Done] Figures generated in: {output_root}")


if __name__ == "__main__":
    main()
