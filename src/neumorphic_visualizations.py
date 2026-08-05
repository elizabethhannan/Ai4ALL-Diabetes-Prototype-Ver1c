## Shared plotting helpers for the training scripts.
## These functions centralize the visual style for exported project figures.
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  ## render figures to files without opening a GUI window
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.patches import Circle, FancyBboxPatch, Wedge


## ---- color palette and depth settings used across all saved figures ----
CREAM = "#f7f3ea"
PANEL = "#fbfaf5"
NAVY = "#0d1a3d"
GREEN = "#75a95a"
GREEN_DARK = "#356a25"
BLUE = "#5279ad"
BLUE_LIGHT = "#a7c9f0"
RED = "#f36f5c"
RED_LIGHT = "#ffa69a"
GOLD = "#efe4ae"
GRID = "#d9d1c4"

# Reversible depth controls. Set EXTRA_DEPTH = False to flatten the charts later.
EXTRA_DEPTH = True
TILE_SHADOW_OFFSET = (5, -5) if EXTRA_DEPTH else (2, -2)
BAR_SHADOW_OFFSET = (3, -3) if EXTRA_DEPTH else (1, -1)
PANEL_SHADOW_OFFSET = (5, -5) if EXTRA_DEPTH else (3, -3)


def add_panel(ax):
    ## Draw the raised background panel behind a chart.
    ax.set_facecolor(PANEL)
    panel = FancyBboxPatch(
        (0, 0), 1, 1, transform=ax.transAxes,
        boxstyle="round,pad=0.018,rounding_size=0.04",
        facecolor=PANEL, edgecolor="#e9e3d8", linewidth=1.2,
        zorder=-10, clip_on=False)
    panel.set_path_effects([
        pe.SimplePatchShadow(offset=PANEL_SHADOW_OFFSET, alpha=0.22, shadow_rgbFace="#8d877c"),
        pe.SimplePatchShadow(offset=(-3, 3), alpha=0.75, shadow_rgbFace="#ffffff"),
        pe.Normal(),
    ])
    ax.add_patch(panel)


def add_badge(ax, text, color=GREEN, x=0.035, y=0.94, width=0.34):
    ## Add a small labeled badge that identifies the dataset/model context.
    badge = FancyBboxPatch(
        (x, y), width, 0.07, transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=color, edgecolor="#ffffff", linewidth=1.0,
        zorder=20, clip_on=False)
    badge.set_path_effects([
        pe.SimplePatchShadow(offset=(3, -3), alpha=0.35, shadow_rgbFace="#5b5245"),
        pe.Normal(),
    ])
    ax.add_patch(badge)
    ax.text(x + width / 2, y + 0.035, text, transform=ax.transAxes,
            ha="center", va="center", color="white", fontsize=10,
            fontweight="bold", zorder=21)


def neumorphic_figure(figsize=(10, 6)):
    ## Start every neumorphic-style figure with the same cream background.
    fig = plt.figure(figsize=figsize, facecolor=CREAM)
    return fig


def draw_beveled_barh(ax, labels, values, color=GREEN, value_fmt="{:.3f}", xerr=None):
    ## Custom horizontal bars with shadow/highlight styling for exports.
    ax.set_xlim(0, max(values) * 1.18 if len(values) else 1)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9, color=NAVY)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linestyle="-", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#bcb4a7")

    bar_height = 0.62
    for i, val in enumerate(values):
        shadow = FancyBboxPatch(
            (0.002, i - bar_height / 2 - 0.02), val, bar_height,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor="#6c655d", edgecolor="none", alpha=0.22, zorder=2)
        ax.add_patch(shadow)
        bar = FancyBboxPatch(
            (0, i - bar_height / 2), val, bar_height,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=color, edgecolor=GREEN_DARK, linewidth=1.1, zorder=3)
        bar.set_path_effects([
            pe.SimplePatchShadow(offset=BAR_SHADOW_OFFSET, alpha=0.35, shadow_rgbFace="#4a4a40"),
            pe.Normal(),
        ])
        ax.add_patch(bar)
        highlight = FancyBboxPatch(
            (0.001, i - bar_height / 2 + bar_height * 0.12), max(val - 0.002, 0), bar_height * 0.28,
            boxstyle="round,pad=0,rounding_size=0.01",
            facecolor="#d9efc3", edgecolor="none", alpha=0.65, zorder=4)
        ax.add_patch(highlight)
        if xerr is not None:
            ax.errorbar(val, i, xerr=xerr[i], fmt="none", ecolor="#8f8a80",
                        elinewidth=1.0, capsize=3, zorder=5)
        ax.text(val + max(values) * 0.03, i, value_fmt.format(val),
                va="center", ha="left", color="#0b5b16", fontsize=9, zorder=6)


def save_feature_selection(ranking, output_path):
    ## Save the Model 0 feature-ranking chart from averaged importances.
    top = ranking.head(14).copy()
    labels = top["feature"].tolist()
    values = top["importance_mean"].to_numpy()
    errors = top["importance_std"].to_numpy()

    fig = neumorphic_figure((10.8, 7.0))
    ax = fig.add_axes([0.24, 0.13, 0.70, 0.74])
    add_panel(ax)
    add_badge(ax, "GE-79 · Feature Selection", GREEN, x=-0.34, y=1.02, width=0.31)
    ax.set_title("Random Forest Feature Selection", loc="left",
                 color=NAVY, fontsize=17, fontweight="bold", pad=22)
    ax.text(0.0, 1.02, "Final locked biomarkers · mean importance over 20 seeds",
            transform=ax.transAxes, color=NAVY, fontsize=10, ha="left")
    draw_beveled_barh(ax, labels, values, xerr=errors)
    ax.set_xlabel("Mean importance (+/- std)", color=NAVY, fontsize=11)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_rf_importance(fi, output_path):
    ## Save the Model 3 Random Forest feature-importance chart.
    labels = fi["feature"].tolist()
    values = fi["importance"].to_numpy()

    fig = neumorphic_figure((10.8, 7.0))
    ax = fig.add_axes([0.25, 0.13, 0.69, 0.74])
    add_panel(ax)
    add_badge(ax, "GE-79 · Random Forest", GREEN, x=-0.35, y=1.02, width=0.31)
    ax.set_title("Random Forest Feature Importance", loc="left",
                 color=NAVY, fontsize=17, fontweight="bold", pad=22)
    ax.text(0.0, 1.02, "Final locked biomarkers · fit on full dataset for interpretation",
            transform=ax.transAxes, color=NAVY, fontsize=10, ha="left")
    draw_beveled_barh(ax, labels, values)
    ax.set_xlabel("Importance", color=NAVY, fontsize=11)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(cm, output_path, model_title, badge_text, footer_text):
    ## Draw a styled 2x2 confusion matrix with explanatory labels.
    fig = neumorphic_figure((6.2, 6.4))
    ax = fig.add_axes([0.12, 0.14, 0.80, 0.76])
    add_panel(ax)
    add_badge(ax, badge_text, BLUE if "Logistic" in badge_text else GREEN,
              x=0.04, y=0.94, width=0.72)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    ax.text(1.0, 2.08, model_title, ha="center", va="center",
            fontsize=15, fontweight="bold", color=NAVY)
    ax.text(1.0, 1.88, "Predicted", ha="center", va="center",
            fontsize=10, fontweight="bold", color=NAVY)
    ax.text(0.46, 1.68, "0 · No Impairment", ha="center", va="center",
            fontsize=9, color="#0755a6")
    ax.text(1.50, 1.68, "1 · Impaired", ha="center", va="center",
            fontsize=9, color="red")
    ax.text(-0.08, 0.72, "Actual", ha="center", va="center",
            rotation=90, fontsize=10, fontweight="bold", color=NAVY)
    ax.text(0.12, 1.15, "0\nNo\nImpairment", ha="center", va="center",
            fontsize=8, color="#0755a6")
    ax.text(0.12, 0.45, "1\nImpaired", ha="center", va="center",
            fontsize=8, color="red")

    colors = [[BLUE, BLUE_LIGHT], [RED, RED_LIGHT]]
    text_colors = [["white", NAVY], ["white", "#6b100b"]]
    positions = [(0.35, 0.95), (1.05, 0.95), (0.35, 0.25), (1.05, 0.25)]
    values = [cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]]
    for idx, ((x, y), val) in enumerate(zip(positions, values)):
        row, col = divmod(idx, 2)
        tile = FancyBboxPatch(
            (x, y), 0.62, 0.58,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            facecolor=colors[row][col], edgecolor="#e9f0ff", linewidth=1.6)
        tile.set_path_effects([
            pe.SimplePatchShadow(offset=TILE_SHADOW_OFFSET, alpha=0.35, shadow_rgbFace="#5f5a52"),
            pe.Normal(),
        ])
        ax.add_patch(tile)
        ax.text(x + 0.31, y + 0.29, str(val), ha="center", va="center",
                fontsize=20, fontweight="bold", color=text_colors[row][col],
                path_effects=[pe.withStroke(linewidth=2, foreground="#ffffff", alpha=0.28)])

    ax.text(1.0, 0.02, footer_text, ha="center", va="bottom",
            fontsize=9.5, color=NAVY, linespacing=1.25)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _add_metric_summary(fig, technical, takeaways):
    ## Add compact technical/takeaway bullets under ROC and PR charts.
    y = 0.225
    fig.text(0.08, y, "Technical:", ha="left", va="top",
             fontsize=11, fontweight="bold", color=NAVY)
    y -= 0.027
    for item in technical:
        wrapped = textwrap.wrap(item, width=105)
        fig.text(0.10, y, "- " + wrapped[0], ha="left", va="top",
                 fontsize=9.5, color=NAVY)
        y -= 0.023
        for line in wrapped[1:]:
            fig.text(0.118, y, line, ha="left", va="top",
                     fontsize=9.5, color=NAVY)
            y -= 0.023
    y -= 0.01
    fig.text(0.08, y, "Takeaways:", ha="left", va="top",
             fontsize=11, fontweight="bold", color=NAVY)
    y -= 0.027
    for item in takeaways:
        wrapped = textwrap.wrap(item, width=105)
        fig.text(0.10, y, "- " + wrapped[0], ha="left", va="top",
                 fontsize=9.5, color=NAVY)
        y -= 0.023
        for line in wrapped[1:]:
            fig.text(0.118, y, line, ha="left", va="top",
                     fontsize=9.5, color=NAVY)
            y -= 0.023


def _add_result_score_box(fig, label, score):
    ## Color-code the headline score box by rough performance band.
    if score < 0.50:
        facecolor = "#b91c1c"
        edgecolor = "#7f1d1d"
    elif score < 0.70:
        facecolor = "#f97316"
        edgecolor = "#c2410c"
    else:
        facecolor = "#15803d"
        edgecolor = "#166534"

    fig.text(
        0.54,
        0.295,
        f"RESULT SCORE: {label} = {score:.3f} ({score * 100:.1f}%)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="white",
        bbox={
            "boxstyle": "round,pad=0.45,rounding_size=0.08",
            "facecolor": facecolor,
            "edgecolor": edgecolor,
            "linewidth": 1.2,
        },
    )


def save_roc_curve(fpr, tpr, auc_score, output_path, model_number, model_type):
    ## Save a ROC curve with a random-classifier reference line.
    fig, ax = plt.subplots(figsize=(9.2, 10.5), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(fpr, tpr, color=BLUE, linewidth=2.6,
            label=("ROC-AUC (Receiver Operating Characteristic - "
                   f"Area Under the Curve) = {auc_score:.3f}"))
    ax.plot([0, 1], [0, 1], color="#7a7a7a", linewidth=1.5,
            linestyle="--", label="Random classifier")
    ax.set_title(f"GE-79 • Model {model_number} • {model_type} • ROC Curve",
                 fontsize=16, fontweight="bold", color=NAVY, pad=14)
    ax.set_xlabel("False Positive Rate (false alarms: no-impairment cases predicted impaired)",
                  fontsize=11, fontweight="bold", color=NAVY)
    ax.set_ylabel("True Positive Rate (impaired recall: impaired cases correctly identified)",
                  fontsize=9.5, fontweight="bold", color=NAVY, labelpad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="lower right", frameon=True, facecolor="white",
              edgecolor="#d1d5db", fontsize=10)
    for spine in ax.spines.values():
        spine.set_color("#d1d5db")
    ax.tick_params(colors=NAVY)
    technical = [
        f"Data represented: out-of-fold predicted probabilities for the impaired class from Model {model_number}, evaluated across the same GE-79 validation folds.",
        "ML impact: ROC-AUC measures how well the model ranks impaired participants above no-impairment participants across many thresholds.",
        "Bias and safety: the curve shows the tradeoff between catching impaired cases and creating false alarms.",
        f"Model result: ROC-AUC is {auc_score:.3f}; 0.500 is random ranking and 1.000 is perfect ranking.",
    ]
    takeaways = [
        "Higher is better, but the model must still be checked for missed impaired cases.",
        "A model can look acceptable overall while still being risky if impaired recall is low.",
        "For AI safety, this figure helps show whether the model is useful before choosing a decision threshold.",
    ]
    _add_result_score_box(fig, "ROC-AUC", auc_score)
    _add_metric_summary(fig, technical, takeaways)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.42)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_precision_recall_curve(recall, precision, pr_auc, output_path, model_number, model_type):
    ## Save a Precision-Recall curve focused on the minority impaired class.
    fig, ax = plt.subplots(figsize=(9.2, 10.5), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(recall, precision, color=GREEN_DARK, linewidth=2.6,
            label=("PR-AUC (Precision-Recall - Area Under the Curve) "
                   f"= {pr_auc:.3f}"))
    ax.set_title(f"GE-79 • Model {model_number} • {model_type} • Precision-Recall Curve",
                 fontsize=16, fontweight="bold", color=NAVY, pad=14)
    ax.set_xlabel("Recall (impaired cases correctly identified)",
                  fontsize=11, fontweight="bold", color=NAVY)
    ax.set_ylabel("Precision (predicted impaired cases that were truly impaired)",
                  fontsize=11, fontweight="bold", color=NAVY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="lower left", frameon=True, facecolor="white",
              edgecolor="#d1d5db", fontsize=10)
    for spine in ax.spines.values():
        spine.set_color("#d1d5db")
    ax.tick_params(colors=NAVY)
    technical = [
        f"Data represented: impaired-class probability scores from Model {model_number}, focused on precision and recall for the minority impaired class.",
        "ML impact: PR-AUC measures how reliably impaired predictions stay correct as recall changes.",
        "Bias and safety: PR-AUC is especially relevant because GE-79 has fewer impaired participants than no-impairment participants.",
        f"Model result: PR-AUC is {pr_auc:.3f}; stronger values mean better minority-class detection with fewer false impaired predictions.",
    ]
    takeaways = [
        "This is often more useful than accuracy when one class is smaller.",
        "A low PR-AUC means the model may struggle to identify impaired participants without many false alarms.",
        "For healthcare AI, this helps explain whether the model is dependable for the group we most need to catch.",
    ]
    _add_result_score_box(fig, "PR-AUC", pr_auc)
    _add_metric_summary(fig, technical, takeaways)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.42)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_target_distribution(y, output_path):
    ## Save the donut chart showing no-impairment vs impaired class balance.
    counts = pd.Series(y).value_counts().sort_index()
    no_imp = int(counts.get(0, 0))
    impaired = int(counts.get(1, 0))
    total = no_imp + impaired
    p0 = no_imp / total if total else 0

    fig = neumorphic_figure((6.2, 6.2))
    ax = fig.add_axes([0.10, 0.12, 0.82, 0.78])
    add_panel(ax)
    add_badge(ax, "GE-79 · Target", GREEN, x=0.03, y=0.93, width=0.36)
    ax.axis("off")
    ax.set_xlim(-1.3, 1.8)
    ax.set_ylim(-1.35, 1.35)
    ax.text(0.15, 1.05, "Class Distribution", ha="center", va="center",
            color=NAVY, fontsize=16, fontweight="bold")
    ax.text(0.15, 0.84, f"n = {total}", ha="center", va="center",
            color=NAVY, fontsize=11, fontweight="bold")

    start = 90
    theta0 = 360 * p0
    for wedge, color in [
        (Wedge((-0.45, -0.05), 0.78, start, start + theta0, width=0.28), BLUE),
        (Wedge((-0.45, -0.05), 0.78, start + theta0, start + 360, width=0.28), RED),
    ]:
        wedge.set_facecolor(color)
        wedge.set_edgecolor("#e9e3d8")
        wedge.set_linewidth(1.5)
        wedge.set_path_effects([
            pe.SimplePatchShadow(offset=TILE_SHADOW_OFFSET, alpha=0.30, shadow_rgbFace="#5f5a52"),
            pe.Normal(),
        ])
        ax.add_patch(wedge)
    ax.add_patch(Circle((-0.45, -0.05), 0.34, facecolor=PANEL, edgecolor="#e9e3d8", linewidth=1.2))
    ax.text(-0.45, -0.05, f"Total\n{total}", ha="center", va="center",
            color=NAVY, fontsize=14, fontweight="bold")

    ax.scatter([0.72], [0.28], s=110, color=BLUE, edgecolor="#e9e3d8", linewidth=1.2)
    ax.text(0.86, 0.28, "No Impairment (0)", va="center", color=NAVY, fontsize=9)
    ax.text(0.72, 0.03, f"{no_imp}", va="center", color=BLUE,
            fontsize=22, fontweight="bold")
    ax.text(0.72, -0.18, f"({p0 * 100:.1f}%)", va="center", color=BLUE,
            fontsize=12, fontweight="bold")
    ax.scatter([0.72], [-0.48], s=110, color=RED, edgecolor="#e9e3d8", linewidth=1.2)
    ax.text(0.86, -0.48, "Impaired (1)", va="center", color=NAVY, fontsize=9)
    ax.text(0.72, -0.73, f"{impaired}", va="center", color=RED,
            fontsize=22, fontweight="bold")
    ax.text(0.72, -0.94, f"({(1 - p0) * 100:.1f}%)", va="center", color=RED,
            fontsize=12, fontweight="bold")
    ax.text(0.15, -1.18, "Imbalanced - report macro-F1 and\nminority recall, not raw accuracy.",
            ha="center", va="center", color=NAVY, fontsize=10)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison(model1_results, model3_results, output_path):
    ## Compare baseline, Logistic Regression, and Random Forest metrics.
    baseline = model1_results.iloc[0]
    lr = model1_results[model1_results["model"] == "Logistic Regression"].iloc[0]
    rf = model3_results[model3_results["model"] == "Random Forest"].iloc[0]
    metrics = ["accuracy", "f1_macro", "recall_impaired"]
    labels = ["Accuracy", "Macro-F1", "Impaired recall"]
    values = np.array([
        [baseline[m] for m in metrics],
        [lr[m] for m in metrics],
        [rf[m] for m in metrics],
    ])

    fig = neumorphic_figure((9.5, 5.6))
    ax = fig.add_axes([0.10, 0.18, 0.84, 0.68])
    add_panel(ax)
    add_badge(ax, "GE-79 · Models", GREEN, x=-0.07, y=0.94, width=0.24)
    ax.set_title("Model Comparison vs Baseline", color=NAVY,
                 fontsize=16, fontweight="bold", pad=42)
    x = np.arange(len(metrics))
    width = 0.22
    colors = [GOLD, BLUE, GREEN]
    names = ["Majority Baseline", "Logistic Regression", "Random Forest"]
    offsets = [-width, 0, width]
    for row, name, color, offset in zip(values, names, colors, offsets):
        for xi, val in zip(x + offset, row):
            bar = FancyBboxPatch(
                (xi - width / 2, 0), width, val,
                boxstyle="round,pad=0.0,rounding_size=0.035",
                facecolor=color, edgecolor="#ffffff", linewidth=1.0)
            bar.set_path_effects([
                pe.SimplePatchShadow(offset=TILE_SHADOW_OFFSET, alpha=0.25, shadow_rgbFace="#5f5a52"),
                pe.Normal(),
            ])
            ax.add_patch(bar)
            ax.text(xi, val + 0.035, f"{val:.3f}" if val not in {0.0} else "0.00",
                    ha="center", va="bottom", color=NAVY, fontsize=9)
    ax.set_xlim(-0.55, len(metrics) - 0.45)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}\n(higher is better)" for label in labels],
                       color=NAVY, fontsize=11, fontweight="bold")
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#bcb4a7")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 1.05),
              ncol=3, frameon=False, fontsize=8)
    ax.text(0.5, -0.28, "Highest accuracy is not always most useful: Logistic Regression catches more impaired cases.",
            transform=ax.transAxes, ha="center", color=NAVY, fontsize=10)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
