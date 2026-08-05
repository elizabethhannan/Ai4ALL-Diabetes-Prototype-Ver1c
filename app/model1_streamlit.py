from pathlib import Path

import pandas as pd
import streamlit as st

## Shared dashboard components used to keep all model pages consistent.
from echarts_components import (
    bottom_export_images,
    confusion_heatmap,
    confusion_sankey,
    inject_theme,
    chart_key,
    key_panel,
    liquid_style_gauge,
    metric_bar_chart,
    model_radar_chart,
    polar_metric_bars,
    project_footer,
    project_page_header,
    section,
    stacked_outcome_bar,
    visual_notes,
)

## ---- paths to project artifacts displayed by this dashboard ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


## ---- Streamlit page setup ----
st.set_page_config(
    page_title="GE-79 Model 1 Logistic Regression",
    page_icon="GE",
    layout="wide",
)
inject_theme()
project_page_header("model1")
st.session_state["visual_takeaways"] = [
    "Model 1 is the Logistic Regression baseline tested on the locked GE-79 feature set.",
    "Its confusion matrix shows 31 correct no-impairment predictions, 10 correct impaired predictions, 24 false alarms, and 10 missed impaired cases.",
    "The model catches 10 of 20 impaired participants, so impaired recall is 0.500.",
    "Because the dataset has more no-impairment cases than impaired cases, Macro F1-score is needed with accuracy to describe performance fairly.",
]

_, key_col = st.columns([1.45, 1], gap="large")
with key_col:
    ## Define the metrics and error types used throughout the page.
    key_panel(
        {
            "Macro F1 / Macro F1-score": "Same metric name. It averages F1 across both classes, so no-impairment and impaired cases both count.",
            "Accuracy": "Percent of GE-79 validation cases classified correctly overall.",
            "Macro Recall": "Average sensitivity for no-impairment and impaired cases.",
            "Impaired Recall": "Share of truly impaired participants Model 1 correctly identifies.",
            "False Alarm": "A no-impairment case predicted as impaired.",
            "Missed Impaired": "An impaired case predicted as no impairment.",
        }
    )

## ---- required Model 1 outputs ----
results_path = OUTPUTS_DIR / "model1_results_model1_logreg.csv"
confusion_path = OUTPUTS_DIR / "model1_fig_lr_confusion.png"

if not results_path.exists():
    ## Stop early if Model 1 has not generated its metrics CSV.
    st.error("Missing Model 1 results file: outputs/model1_results_model1_logreg.csv")
    st.stop()

results = pd.read_csv(results_path)
model_row = results[results["model"].eq("Logistic Regression")]
if model_row.empty:
    st.error("Model 1 results CSV does not contain a 'Logistic Regression' row.")
    st.stop()

metrics = model_row.iloc[0]
confusion_matrix = [[31, 24], [10, 10]]  ## rows = actual, columns = predicted

## ---- headline metrics ----
metric_cols = st.columns(4)
metric_cols[0].metric("Macro F1-score", f"{metrics['f1_macro']:.3f}")
metric_cols[1].metric("Accuracy", f"{metrics['accuracy']:.3f}")
metric_cols[2].metric("Macro Recall", f"{metrics['recall_macro']:.3f}")
metric_cols[3].metric("Impaired Recall", f"{metrics['recall_impaired']:.3f}")

section("Performance Overview")

## ---- gauges summarize the most presentation-friendly scores ----
gauge_cols = st.columns(3)
with gauge_cols[0]:
    chart_key("Macro F1-score", "#2a9d8f")
    liquid_style_gauge(metrics["f1_macro"], "Macro F1", "model1_f1_gauge", "#2a9d8f")
    visual_notes(
        "Macro F1 gauge",
        [
            "Data represented: Model 1's Macro F1 summarizes balanced precision-recall performance across both outcome classes.",
            "ML impact: Macro F1 limits the effect of class imbalance by giving impaired and no-impairment cases equal importance.",
            "Model 1 accomplishment: Logistic Regression provides a baseline measure of balanced classification performance.",
        ],
        [
            "Macro F1 and Macro F1-score mean the same metric here.",
            "Higher is better when both groups matter.",
            "It helps show if Model 1 is weak on the smaller impaired group.",
        ],
    )
with gauge_cols[1]:
    chart_key("Accuracy")
    liquid_style_gauge(metrics["accuracy"], "Accuracy", "model1_accuracy_gauge")
    visual_notes(
        "Accuracy gauge",
        [
            "Data represented: Model 1 Logistic Regression correctly classified 54.7% of validation cases.",
            "ML impact: because the no-impairment class is larger, this accuracy must be interpreted with recall and Macro F1.",
            "Model 1 accomplishment: it establishes an explainable linear baseline for cognitive-status prediction.",
        ],
        [
            "This is the overall percent-correct score.",
            "It should not be the only score used.",
            "A model can have decent accuracy while still missing impaired cases.",
        ],
    )
with gauge_cols[2]:
    chart_key("Impaired Recall", "#ff6b6b")
    liquid_style_gauge(metrics["recall_impaired"], "Impaired Recall", "model1_recall_gauge", "#ff6b6b")
    visual_notes(
        "Impaired recall gauge",
        [
            "Data represented: Model 1 detected 10 of 20 impaired cases, giving impaired recall of 0.50.",
            "ML impact: this value shows how many minority-class cases the linear baseline catches instead of labeling as no impairment.",
            "Model 1 accomplishment: Logistic Regression improves minority-class detection compared with a majority-only baseline.",
        ],
        [
            "This shows how many impaired cases the model catches.",
            "Low recall means too many impaired cases are missed.",
            "This is one of the most important safety-focused metrics.",
        ],
    )

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    ## Left column focuses on raw outcome counts and validation bars.
    chart_key("Confusion Matrix")
    confusion_heatmap(confusion_matrix, "model1_confusion")
    st.markdown(
        f"""
        <div class="visual-notes">
            <span class="visual-title">What this confusion matrix means</span>
            <ul>
                <li><strong>Fact:</strong> Model 1 correctly classified 31 no-impairment participants and 10 impaired participants.</li>
                <li><strong>Fact:</strong> Model 1 produced 24 false alarms and missed 10 impaired participants.</li>
                <li><strong>Output interpretation:</strong> impaired recall is {metrics['recall_impaired']:.3f}, so the model catches half of the impaired group.</li>
                <li><strong>Defensible justification:</strong> Macro F1-score is {metrics['f1_macro']:.3f}; this is more informative than accuracy alone because the GE-79 target has more no-impairment cases than impaired cases.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    visual_notes(
        "Confusion matrix",
        [
            "Data represented: Model 1 produced 31 true no-impairment predictions, 24 false alarms, 10 missed impaired cases, and 10 detected impaired cases.",
            "ML impact: these counts show the linear model trades some no-impairment accuracy to catch more impaired participants.",
            "Model 1 accomplishment: Logistic Regression creates a transparent baseline for understanding the false-positive and false-negative tradeoff.",
        ],
        [
            "Model 1 catches half of the impaired group.",
            "It also incorrectly flags some no-impairment participants.",
            "This shows the baseline model's main tradeoff.",
        ],
    )
    chart_key("Actual-to-Predicted Flow", "#2a9d8f")
    confusion_sankey(confusion_matrix, "model1_sankey")
    visual_notes(
        "Actual-to-predicted flow",
        [
            "Data represented: the same Model 1 confusion counts are organized by actual class and predicted class.",
            "ML impact: the flow of impaired cases into predicted no-impairment highlights false negatives that reduce impaired recall.",
            "Model 1 accomplishment: Logistic Regression makes the baseline movement from true labels to predicted labels visible for error review.",
        ],
        [
            "The data shows where participants move from real group to predicted group.",
            "Missed impaired cases are the key error to watch.",
            "Correct and incorrect prediction paths explain the baseline behavior.",
        ],
    )
    chart_key("Cross-Validation Metric Bars", "#f2994a")
    metric_bar_chart(results, "model1_metric_bars")
    visual_notes(
        "Cross-validation metric bars",
        [
            "Data represented: baseline and Logistic Regression validation metrics for accuracy, precision, recall, Macro F1, and impaired recall.",
            "ML impact: these metrics show why majority-class accuracy is insufficient when impaired recall is clinically important.",
            "Model 1 accomplishment: it demonstrates whether a simple linear model improves detection beyond the majority baseline.",
        ],
        [
            "The baseline looks strong only because most cases are no impairment.",
            "Model 1 is judged by whether it catches impaired cases too.",
            "Multiple metrics give a fairer model assessment.",
        ],
    )

with right:
    ## Right column provides compact profile views of the same metrics.
    chart_key("Polar Metric Profile", "#1f3a5f")
    polar_metric_bars(metrics.to_dict(), "model1_polar")
    visual_notes(
        "Polar metric profile",
        [
            "Data represented: Model 1's accuracy, precision, recall, Macro F1, and impaired recall scores together.",
            "ML impact: the combined metric profile shows which parts of the baseline are strong or weak before comparing Models 2 and 3.",
            "Model 1 accomplishment: Logistic Regression establishes the first model-performance profile for the project.",
        ],
        [
            "This summarizes all Model 1 scores together.",
            "Weak scores show where the baseline needs improvement.",
            "It sets the comparison point for the later models.",
        ],
    )
    chart_key("Radar Metric Profile", "#2a9d8f")
    model_radar_chart(metrics.to_dict(), "Logistic Regression", "model1_radar")
    visual_notes(
        "Radar metric profile",
        [
            "Data represented: the same Model 1 evaluation metrics shown as one combined performance profile.",
            "ML impact: the profile reveals whether the linear model is balanced or dominated by only one strong metric.",
            "Model 1 accomplishment: it gives an interpretable baseline performance profile for comparing against Decision Tree and Random Forest.",
        ],
        [
            "The profile shows the baseline model's balance.",
            "Dips identify weaker performance areas.",
            "This helps compare Model 1 with the other models.",
        ],
    )
    chart_key("Predicted-Class Outcome Stack", "#ff6b6b")
    stacked_outcome_bar(confusion_matrix, "model1_stacked")
    visual_notes(
        "Predicted-class outcome stack",
        [
            "Data represented: Model 1 prediction counts separated by actual no-impairment and actual impaired groups.",
            "ML impact: the split shows how the linear model distributes errors across the two classes.",
            "Model 1 accomplishment: Logistic Regression identifies some impaired cases while showing the cost in false alarms.",
        ],
        [
            "This shows what Model 1 predicted for each real group.",
            "The impaired group still has missed cases.",
            "The no-impairment group includes some false alarms.",
        ],
    )
    section("K-Fold Validation and Performance Metrics")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.markdown(
        """
        Model 1 uses 5-fold stratified cross-validation: the GE-79 data is split
        into five folds, each fold is held out once for validation, and
        preprocessing is fit only inside the training folds. The performance
        metrics reported here are accuracy, precision macro, recall macro,
        Macro F1-score, and impaired recall.
        """
    )

section("Original Confusion-Matrix Export")
if confusion_path.exists():
    st.image(str(confusion_path), use_container_width=True)
    st.markdown(
        f"""
        <div class="visual-notes">
            <span class="visual-title">Saved confusion-matrix facts</span>
            <ul>
                <li>The saved image records the same Model 1 prediction counts shown above.</li>
                <li>Macro F1-score is {metrics['f1_macro']:.3f}; Macro F1 and Macro F1-score refer to the same score in this project.</li>
                <li>The key interpretation is that Model 1 detects 10 impaired participants but also misses 10 impaired participants.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    visual_notes(
        "Original confusion-matrix export",
        [
            "Data represented: the original saved Model 1 confusion-matrix counts from the Logistic Regression workflow.",
            "ML impact: the saved artifact documents the exact prediction outcomes used to interpret baseline performance.",
            "Model 1 accomplishment: it preserves the Logistic Regression evidence used for project reporting.",
        ],
        [
            "This is the original saved Model 1 confusion-matrix image.",
            "It records the model's correct and incorrect predictions.",
            "It helps document the model output used in the project.",
        ],
    )
else:
    st.warning("Missing confusion-matrix image: outputs/model1_fig_lr_confusion.png")

bottom_export_images(
    "Model 1 Visualization Export",
    [
        ("Model 1 Focused Visualization", str(OUTPUTS_DIR / "model_1_visualization_FOCUSED_EH_pale..png")),
    ],
)

project_footer()
