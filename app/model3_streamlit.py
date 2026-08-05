from pathlib import Path

import pandas as pd
import streamlit as st

## Shared dashboard components used to keep all model pages consistent.
from echarts_components import (
    bottom_export_images,
    confusion_heatmap,
    confusion_sankey,
    horizontal_importance_chart,
    inject_theme,
    chart_key,
    key_panel,
    liquid_style_gauge,
    metric_bar_chart,
    mixed_metric_combo,
    model_comparison_radar,
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
    page_title="GE-79 Model 3 Random Forest",
    page_icon="GE",
    layout="wide",
)
inject_theme()
project_page_header("model3")
st.session_state["visual_takeaways"] = [
    "Model 3 is the Random Forest ensemble tested on the same locked GE-79 feature set as Models 1 and 2.",
    "Its confusion matrix shows 51 correct no-impairment predictions, 5 correct impaired predictions, 4 false alarms, and 15 missed impaired cases.",
    "The model has stronger no-impairment classification but catches only 5 of 20 impaired participants, so impaired recall is 0.250.",
    "The feature-importance visuals show which selected inputs the forest used most, but they do not prove clinical causation.",
]

_, key_col = st.columns([1.45, 1], gap="large")
with key_col:
    ## Define metric and comparison terms used throughout the page.
    key_panel(
        {
            "Macro F1 / Macro F1-score": "Same metric name. It averages F1 across both classes, so no-impairment and impaired cases both count.",
            "Accuracy": "Overall percent of GE-79 validation classifications the forest gets right.",
            "Impaired Recall": "Share of truly impaired participants the forest correctly identifies.",
            "Random Forest": "An ensemble of many trees voting on cognitive-status class.",
            "Feature Importance": "Which locked Model 0 predictors most influenced the forest's decisions.",
            "Model Comparison": "Side-by-side scoring of Models 1-3 on the same locked features.",
        }
    )

## ---- required Model 3 outputs ----
results_path = OUTPUTS_DIR / "model3_results_model3_forest.csv"
confusion_path = OUTPUTS_DIR / "model3_fig_rf_confusion.png"
importance_path = OUTPUTS_DIR / "model3_fig_rf_importance.png"
importance_data_path = OUTPUTS_DIR / "model3_feature_importance_model3.csv"
comparison_path = OUTPUTS_DIR / "model1_model3_fig_model_comparison.png"

MODEL_OPTIONS = {
    "Model_1": {
        "path": OUTPUTS_DIR / "model1_results_model1_logreg.csv",
        "model": "Logistic Regression",
        "legend": "Model 1 - Logistic Regression",
    },
    "Model_2": {
        "path": OUTPUTS_DIR / "model2_results_tree.csv",
        "model": "Decision Tree",
        "legend": "Model 2 - Decision Tree",
    },
    "Model_3": {
        "path": OUTPUTS_DIR / "model3_results_model3_forest.csv",
        "model": "Random Forest",
        "legend": "Model 3 - Random Forest",
    },
}
MODEL_SELECTION_KEYS = [
    "model3_compare_selection_1",
    "model3_compare_selection_2",
    "model3_compare_selection_3",
]


def clear_model_comparison_selection() -> None:
    ## Reset optional comparison controls back to no model selected.
    for key in MODEL_SELECTION_KEYS:
        st.session_state[key] = "None"


def load_comparison_row(label: str) -> pd.DataFrame:
    ## Load one selected model's latest metrics row for comparison visuals.
    if label == "None":
        return pd.DataFrame()
    config = MODEL_OPTIONS[label]
    if not config["path"].exists():
        return pd.DataFrame()
    rows = pd.read_csv(config["path"])
    model_rows = rows[rows["model"].eq(config["model"])]
    selected = model_rows.tail(1) if not model_rows.empty else rows.tail(1)
    selected = selected.copy()
    selected["model"] = config["legend"]
    return selected

if not results_path.exists():
    ## Stop early if Model 3 has not generated its metrics CSV.
    st.error("Missing Model 3 results file: outputs/model3_results_model3_forest.csv")
    st.stop()

results = pd.read_csv(results_path)
model_row = results[results["model"].eq("Random Forest")]
if model_row.empty:
    st.error("Model 3 results CSV does not contain a 'Random Forest' row.")
    st.stop()

metrics = model_row.iloc[0]
confusion_matrix = [[51, 4], [15, 5]]  ## rows = actual, columns = predicted

## ---- headline metrics ----
metric_cols = st.columns(4)
metric_cols[0].metric("Macro F1-score", f"{metrics['f1_macro']:.3f}")
metric_cols[1].metric("Accuracy", f"{metrics['accuracy']:.3f}")
metric_cols[2].metric("Macro Recall", f"{metrics['recall_macro']:.3f}")
metric_cols[3].metric("Impaired Recall", f"{metrics['recall_impaired']:.3f}")

section("Performance Overview")

## ---- gauges summarize the most important evaluation scores ----
gauge_cols = st.columns(3)
with gauge_cols[0]:
    chart_key("Macro F1-score", "#2a9d8f")
    liquid_style_gauge(metrics["f1_macro"], "Macro F1", "model3_f1_gauge", "#2a9d8f")
    visual_notes(
        "Macro F1 gauge",
        [
            "Data represented: Model 3 Macro F1 across no-impairment and impaired classes.",
            "ML impact: Macro F1 shows whether the ensemble performs well across both classes rather than mainly fitting the larger class.",
            "Model 3 accomplishment: the Random Forest provides an ensemble balanced-performance score for comparison with linear and single-tree models.",
        ],
        [
            "Macro F1 and Macro F1-score mean the same metric here.",
            "It rewards catching cases without creating too many false alarms.",
            "A stronger Macro F1 means performance is steadier across groups.",
        ],
    )
with gauge_cols[1]:
    chart_key("Accuracy")
    liquid_style_gauge(metrics["accuracy"], "Accuracy", "model3_accuracy_gauge")
    visual_notes(
        "Accuracy gauge",
        [
            "Data represented: Model 3 Random Forest overall validation accuracy across cognitive-status predictions.",
            "ML impact: accuracy reflects total correctness but must be read with impaired recall because the impaired class is smaller.",
            "Model 3 accomplishment: the Random Forest tests whether an ensemble of trees improves stability over Models 1 and 2.",
        ],
        [
            "This is the overall percent-correct score.",
            "A higher value means the model was right more often.",
            "It should be read together with recall and F1, not by itself.",
        ],
    )
with gauge_cols[2]:
    chart_key("Impaired Recall", "#ff6b6b")
    liquid_style_gauge(metrics["recall_impaired"], "Impaired Recall", "model3_recall_gauge", "#ff6b6b")
    visual_notes(
        "Impaired recall gauge",
        [
            "Data represented: Model 3 detected 5 of 20 impaired participants, giving impaired recall of 0.25.",
            "ML impact: this lower impaired recall shows the ensemble is conservative and misses many minority-class cases despite stronger no-impairment performance.",
            "Model 3 accomplishment: the Random Forest reveals the tradeoff between ensemble accuracy and impaired-case sensitivity.",
        ],
        [
            "This shows how many impaired cases the model catches.",
            "Low impaired recall means the model misses too many at-risk people.",
            "This is one of the most important fairness and safety checks here.",
        ],
    )

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    ## Left column focuses on outcome counts and validation bars.
    chart_key("Confusion Matrix")
    confusion_heatmap(confusion_matrix, "model3_confusion")
    st.markdown(
        f"""
        <div class="visual-notes">
            <span class="visual-title">What this confusion matrix means</span>
            <ul>
                <li><strong>Fact:</strong> Model 3 correctly classified 51 no-impairment participants and 5 impaired participants.</li>
                <li><strong>Fact:</strong> Model 3 produced 4 false alarms and missed 15 impaired participants.</li>
                <li><strong>Output interpretation:</strong> impaired recall is {metrics['recall_impaired']:.3f}, so the forest misses most impaired participants even though accuracy is higher.</li>
                <li><strong>Defensible justification:</strong> Macro F1-score is {metrics['f1_macro']:.3f}; this shows why Model 3 should not be judged by accuracy alone on an imbalanced target.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    visual_notes(
        "Confusion matrix",
        [
            "Data represented: Model 3 produced 51 true no-impairment predictions, 4 false alarms, 15 missed impaired cases, and 5 detected impaired cases.",
            "ML impact: these counts show the Random Forest strongly protects the no-impairment class but misses many impaired cases.",
            "Model 3 accomplishment: the ensemble model improves specificity while exposing a sensitivity limitation for impaired detection.",
        ],
        [
            "Model 3 is strong at recognizing no impairment.",
            "It misses many impaired participants.",
            "This tradeoff matters when choosing the best model for screening.",
        ],
    )
    chart_key("Actual-to-Predicted Flow", "#2a9d8f")
    confusion_sankey(confusion_matrix, "model3_sankey")
    visual_notes(
        "Actual-to-predicted flow",
        [
            "Data represented: Model 3 actual labels flowing into Random Forest predicted labels using the same confusion counts.",
            "ML impact: the large movement from actual impaired to predicted no impairment shows why impaired recall is low.",
            "Model 3 accomplishment: the Random Forest prediction paths reveal how the ensemble distributes each outcome class.",
        ],
        [
            "The data shows where true classes are sent by the ensemble.",
            "Most no-impairment cases are classified correctly.",
            "Too many impaired cases are sent to no impairment.",
        ],
    )
    chart_key("Cross-Validation Metric Bars", "#f2994a")
    metric_bar_chart(results, "model3_metric_bars")
    visual_notes(
        "Cross-validation metric bars",
        [
            "Data represented: Model 3 validation metrics for accuracy, precision, recall, Macro F1, and impaired recall.",
            "ML impact: the metrics show that ensemble performance must be judged by both no-impairment accuracy and impaired-class sensitivity.",
            "Model 3 accomplishment: the Random Forest gives a high-specificity ensemble benchmark against the simpler models.",
        ],
        [
            "The ensemble has strengths and weaknesses.",
            "Accuracy alone does not tell the full story.",
            "Impaired recall is the key weakness to consider.",
        ],
    )

with right:
    ## Right column provides compact model-performance profile views.
    chart_key("Polar Metric Profile", "#1f3a5f")
    polar_metric_bars(metrics.to_dict(), "model3_polar")
    visual_notes(
        "Polar metric profile",
        [
            "Data represented: Model 3's combined accuracy, precision, recall, Macro F1, and impaired recall values.",
            "ML impact: this combined profile shows that the ensemble is not equally strong across every model goal.",
            "Model 3 accomplishment: the Random Forest provides a full metric profile for comparing ensemble behavior with Models 1 and 2.",
        ],
        [
            "This summarizes the ensemble's main scores together.",
            "The impaired recall score is the main caution.",
            "The profile helps compare Model 3 with the other models.",
        ],
    )
    chart_key("Radar Metric Profile", "#2a9d8f")
    model_radar_chart(metrics.to_dict(), "Random Forest", "model3_radar")
    visual_notes(
        "Radar metric profile",
        [
            "Data represented: the same Random Forest metrics combined into one performance profile.",
            "ML impact: the metric profile shows whether Model 3 is balanced or dominated by strong no-impairment classification.",
            "Model 3 accomplishment: the ensemble profile makes its strengths and impaired-detection tradeoff comparable to Models 1 and 2.",
        ],
        [
            "This shows Model 3's overall performance balance.",
            "The weaker impaired-recall area is important.",
            "It helps decide whether the ensemble is worth the tradeoff.",
        ],
    )
    chart_key("Predicted-Class Outcome Stack", "#ff6b6b")
    stacked_outcome_bar(confusion_matrix, "model3_stacked")
    visual_notes(
        "Predicted-class outcome stack",
        [
            "Data represented: Model 3 predicted-class counts inside the actual no-impairment and impaired groups.",
            "ML impact: the class split shows the ensemble predicts no impairment very often, including for many impaired participants.",
            "Model 3 accomplishment: the Random Forest outcome split reveals the ensemble's class-specific behavior.",
        ],
        [
            "Model 3 correctly handles most no-impairment cases.",
            "It misses many impaired cases.",
            "This explains why impaired recall is lower.",
        ],
    )
    if importance_data_path.exists():
        chart_key("Feature Importance", "#2a9d8f")
        importance = pd.read_csv(importance_data_path)
        horizontal_importance_chart(importance, "model3_importance", limit=12)
        visual_notes(
            "Feature importance",
            [
                "Data represented: Random Forest feature-importance values for the locked Model 0 predictors.",
                "ML impact: these values show which predictors most influenced the ensemble's tree splits and prediction behavior.",
                "Model 3 accomplishment: the Random Forest ranks which selected features drive its classification decisions.",
            ],
            [
                "The highest-ranked features are the inputs the model relied on most.",
                "This does not prove a feature causes impairment.",
                "It helps explain what the model paid attention to.",
            ],
        )
    else:
        st.warning("Missing feature-importance data: outputs/model3_feature_importance_model3.csv")

section("Model Comparison")
model_labels = ["None", *MODEL_OPTIONS.keys()]
default_selections = ["Model_1", "Model_2", "Model_3"]
for key, default in zip(MODEL_SELECTION_KEYS, default_selections):
    st.session_state.setdefault(key, default)

comparison_select_cols = st.columns([1, 1, 1, 0.8], gap="medium")
selected_labels = []
for idx, column in enumerate(comparison_select_cols[:3]):
    with column:
        selected_labels.append(
            st.selectbox(
                f"Model selection {idx + 1}",
                model_labels,
                key=MODEL_SELECTION_KEYS[idx],
            )
        )
with comparison_select_cols[3]:
    st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    st.button(
        "Clear selection",
        key="model3_clear_comparison_selection",
        on_click=clear_model_comparison_selection,
        use_container_width=True,
    )

active_labels = []
for label in selected_labels:
    if label != "None" and label not in active_labels:
        active_labels.append(label)

if len(active_labels) < len([label for label in selected_labels if label != "None"]):
    st.info("Duplicate model selections are shown once in the comparison charts.")

comparison_rows = (
    pd.concat([load_comparison_row(label) for label in active_labels], ignore_index=True)
    if active_labels
    else pd.DataFrame()
)
if len(active_labels) < 2:
    st.info("Select at least two models to compare the radar and matrix visualizations.")
elif not comparison_rows.empty:
    chart_key("Matrix Display Comparison", "#f2994a")
    mixed_metric_combo(comparison_rows, "model3_matrix_display")
    visual_notes(
        "Matrix display comparison",
        [
            "Data represented: validation metrics for the selected Models 1-3 comparison set.",
            "ML impact: selecting two or three models shows how model type changes accuracy, Macro F1, and impaired recall tradeoffs.",
            "Model comparison accomplishment: it identifies which model type best matches the project goal selected by the metric.",
        ],
        [
            "Choose two or three models from the dropdowns.",
            "The chart updates to compare their actual validation scores.",
            "Use it to decide which model is stronger for the project goal.",
        ],
    )
    chart_key("Selected Models Radar", "#2f80ed")
    model_comparison_radar(comparison_rows, "model3_comparison_radar")
    visual_notes(
        "Selected-model radar comparison",
        [
            "Data represented: the selected models' validation metrics across accuracy, precision, recall, Macro F1, and impaired recall.",
            "ML impact: differences across metrics show whether Logistic Regression, Decision Tree, or Random Forest best handles class imbalance.",
            "Model comparison accomplishment: it summarizes the tradeoffs that determine which ML model is most appropriate.",
        ],
        [
            "This compares the selected models on the same metrics.",
            "One model may win on accuracy while another catches more impaired cases.",
            "The best choice depends on the project priority.",
        ],
    )

left_static, middle_static, right_static = st.columns(3, gap="large")
with left_static:
    chart_key("Original Model Comparison Export", "#1f3a5f")
    if comparison_path.exists():
        st.image(str(comparison_path), use_container_width=True)
        visual_notes(
            "Original model comparison export",
            [
                "Data represented: the original saved comparison output for Model 1 and Model 3 metrics.",
                "ML impact: this artifact documents the earlier baseline-versus-ensemble comparison used in the project workflow.",
                "Model comparison accomplishment: it preserves the evidence used to compare Logistic Regression and Random Forest.",
            ],
            [
                "This is the original saved comparison image.",
                "It records the earlier model comparison.",
                "It helps document the project decision process.",
            ],
        )
    else:
        st.warning("Missing comparison image: outputs/model1_model3_fig_model_comparison.png")

with middle_static:
    chart_key("Original Confusion-Matrix Export", "#ff6b6b")
    if confusion_path.exists():
        st.image(str(confusion_path), use_container_width=True)
        st.markdown(
            f"""
            <div class="visual-notes">
                <span class="visual-title">Saved confusion-matrix facts</span>
                <ul>
                    <li>The saved image records the same Model 3 prediction counts shown above.</li>
                    <li>Macro F1-score is {metrics['f1_macro']:.3f}; Macro F1 and Macro F1-score refer to the same score in this project.</li>
                    <li>The key interpretation is that Model 3 correctly identifies 5 impaired participants but misses 15 impaired participants.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        visual_notes(
            "Original confusion-matrix export",
            [
                "Data represented: the original saved Model 3 confusion-matrix counts from the Random Forest workflow.",
                "ML impact: this output documents that the ensemble has high no-impairment correctness but low impaired-case detection.",
                "Model 3 accomplishment: it preserves the evidence behind the Random Forest's reported performance tradeoff.",
            ],
            [
                "This is the original saved Model 3 confusion matrix.",
                "It records the ensemble model's correct and incorrect predictions.",
                "It supports checking the dashboard values against the saved workflow output.",
            ],
        )
    else:
        st.warning("Missing confusion-matrix image: outputs/model3_fig_rf_confusion.png")

with right_static:
    chart_key("Original Feature-Importance Export", "#2a9d8f")
    if importance_path.exists():
        st.image(str(importance_path), use_container_width=True)
        visual_notes(
            "Original feature-importance export",
            [
                "Data represented: the original saved Random Forest feature-importance output.",
                "ML impact: this ranking shows which locked predictors most affected the ensemble model.",
                "Model 3 accomplishment: it documents the feature signals the Random Forest relied on for classification.",
            ],
            [
                "This is the original saved feature-ranking image.",
                "It helps confirm the feature-ranking results shown in the app.",
                "The highest features are the ones the model used most strongly.",
            ],
        )
    else:
        st.warning("Missing feature-importance image: outputs/model3_fig_rf_importance.png")

section("K-Fold Validation and Performance Metrics")
st.dataframe(results, use_container_width=True, hide_index=True)
st.markdown(
    """
    Model 3 uses 5-fold stratified cross-validation: the GE-79 data is split
    into five folds, each fold is held out once for validation, and
    preprocessing is fit only inside the training folds. The performance
    metrics reported here are accuracy, precision macro, recall macro,
    Macro F1-score, and impaired recall. The feature-importance results
    summarize which selected predictors contributed most to the fitted forest.
    """
)
bottom_export_images(
    "Model 3 Visualization Export",
    [
        ("Model 3 Focused Visualization", str(OUTPUTS_DIR / "model_3_visualization_FOCUSED_EH_pale.png")),
    ],
)

project_footer()
