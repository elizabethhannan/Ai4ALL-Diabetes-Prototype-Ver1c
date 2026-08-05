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
    page_title="GE-79 Model 2 Decision Tree",
    page_icon="GE",
    layout="wide",
)
inject_theme()
project_page_header("model2")
st.session_state["visual_takeaways"] = [
    "Model 2 is the Decision Tree model tested on the same locked GE-79 feature set as Models 1 and 3.",
    "Its confusion matrix shows 36 correct no-impairment predictions, 11 correct impaired predictions, 19 false alarms, and 9 missed impaired cases.",
    "The model catches 11 of 20 impaired participants, giving it the highest impaired recall among Models 1-3 in this dashboard.",
    "The visuals define a more interpretable rule-based model, but the results still come from a small imbalanced GE-79 dataset.",
]

_, key_col = st.columns([1.45, 1], gap="large")
with key_col:
    ## Define model metrics and tree-specific terms used on this page.
    key_panel(
        {
            "Macro F1 / Macro F1-score": "Same metric name. It averages F1 across both classes, so no-impairment and impaired cases both count.",
            "Accuracy": "Overall percent of GE-79 validation classifications the tree gets right.",
            "Impaired Recall": "How many truly impaired participants the tree catches.",
            "Tree Split": "A learned feature threshold used to route a participant toward a prediction.",
            "False Alarm": "A no-impairment case predicted as impaired.",
            "Missed Impaired": "An impaired case predicted as no impairment.",
        }
    )

## ---- required Model 2 outputs ----
results_path = OUTPUTS_DIR / "model2_results_tree.csv"
confusion_path = OUTPUTS_DIR / "model2_fig_dt_confusion.png"
tree_path = OUTPUTS_DIR / "model2_fig_dt_tree.png"

if not results_path.exists():
    ## Stop early if Model 2 has not generated its metrics CSV.
    st.error("Missing Model 2 results file: outputs/model2_results_tree.csv")
    st.stop()

results = pd.read_csv(results_path)
model_row = results[results["model"].eq("Decision Tree")]
if model_row.empty:
    st.error("Model 2 results CSV does not contain a 'Decision Tree' row.")
    st.stop()

metrics = model_row.iloc[0]
confusion_matrix = [[36, 19], [9, 11]]  ## rows = actual, columns = predicted

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
    liquid_style_gauge(metrics["f1_macro"], "Macro F1", "model2_f1_gauge", "#2a9d8f")
    visual_notes(
        "Macro F1 gauge",
        [
            "Data represented: Model 2 Macro F1 across no-impairment and impaired predictions.",
            "ML impact: this score evaluates whether tree rules perform reasonably across both classes, not just the majority group.",
            "Model 2 accomplishment: the Decision Tree provides an interpretable balanced-performance score for comparison with Models 1 and 3.",
        ],
        [
            "Macro F1 and Macro F1-score mean the same metric here.",
            "Higher values mean fewer tradeoffs between false alarms and missed cases.",
            "It helps compare Model 2 against Models 1 and 3.",
        ],
    )
with gauge_cols[1]:
    chart_key("Accuracy")
    liquid_style_gauge(metrics["accuracy"], "Accuracy", "model2_accuracy_gauge")
    visual_notes(
        "Accuracy gauge",
        [
            "Data represented: Model 2 Decision Tree overall validation accuracy across the cognitive-status classes.",
            "ML impact: accuracy summarizes total correctness but must be balanced against impaired recall because the classes are uneven.",
            "Model 2 accomplishment: the Decision Tree tests whether interpretable rule-based splits improve classification over the baseline.",
        ],
        [
            "This is the overall percent-correct score.",
            "It is useful but incomplete by itself.",
            "Use it with recall and F1 to judge Model 2 fairly.",
        ],
    )
with gauge_cols[2]:
    chart_key("Impaired Recall", "#ff6b6b")
    liquid_style_gauge(metrics["recall_impaired"], "Impaired Recall", "model2_recall_gauge", "#ff6b6b")
    visual_notes(
        "Impaired recall gauge",
        [
            "Data represented: Model 2 correctly identified 11 of 20 impaired participants, giving impaired recall of 0.55.",
            "ML impact: impaired recall shows whether the tree rules catch the smaller, clinically important class.",
            "Model 2 accomplishment: the Decision Tree improves impaired-case detection relative to the Logistic Regression baseline.",
        ],
        [
            "This shows how many impaired cases Model 2 catches.",
            "A lower value means more impaired cases are missed.",
            "This is important when screening for at-risk participants.",
        ],
    )

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    ## Left column focuses on outcome counts and validation bars.
    chart_key("Confusion Matrix")
    confusion_heatmap(confusion_matrix, "model2_confusion")
    st.markdown(
        f"""
        <div class="visual-notes">
            <span class="visual-title">What this confusion matrix means</span>
            <ul>
                <li><strong>Fact:</strong> Model 2 correctly classified 36 no-impairment participants and 11 impaired participants.</li>
                <li><strong>Fact:</strong> Model 2 produced 19 false alarms and missed 9 impaired participants.</li>
                <li><strong>Output interpretation:</strong> impaired recall is {metrics['recall_impaired']:.3f}, the highest impaired recall among Models 1-3 in this app.</li>
                <li><strong>Defensible justification:</strong> Macro F1-score is {metrics['f1_macro']:.3f}; this supports the claim that Model 2 gives a more balanced class tradeoff than accuracy alone can show.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    visual_notes(
        "Confusion matrix",
        [
            "Data represented: Model 2 produced 36 true no-impairment predictions, 19 false alarms, 9 missed impaired cases, and 11 detected impaired cases.",
            "ML impact: these counts show how the learned tree rules distribute false positives and false negatives.",
            "Model 2 accomplishment: the Decision Tree gives an inspectable error pattern for rule-based cognitive-status classification.",
        ],
        [
            "Model 2 catches slightly more impaired cases than Model 1.",
            "It still misses some impaired participants.",
            "The error counts show where the tree rules need caution.",
        ],
    )
    chart_key("Actual-to-Predicted Flow", "#2a9d8f")
    confusion_sankey(confusion_matrix, "model2_sankey")
    visual_notes(
        "Actual-to-predicted flow",
        [
            "Data represented: Model 2 prediction counts moving from actual outcome labels to predicted outcome labels.",
            "ML impact: the movement from actual impaired to predicted no impairment represents false negatives that reduce impaired recall.",
            "Model 2 accomplishment: the Decision Tree's rule-based predictions can be audited by tracking where each class is sent.",
        ],
        [
            "The data shows how real groups become predicted groups.",
            "Missed impaired cases remain the key risk.",
            "The flows explain how the tree's rules behave on each class.",
        ],
    )
    chart_key("Cross-Validation Metric Bars", "#f2994a")
    metric_bar_chart(results, "model2_metric_bars")
    visual_notes(
        "Cross-validation metric bars",
        [
            "Data represented: Model 2 validation metrics for accuracy, precision, recall, Macro F1, and impaired recall.",
            "ML impact: these metrics show whether the interpretable tree improves the project goals beyond the linear baseline.",
            "Model 2 accomplishment: the Decision Tree provides a rule-based model that can be evaluated against both accuracy and impaired detection.",
        ],
        [
            "These scores show how well the tree performs from several angles.",
            "Accuracy alone is not enough.",
            "Impaired recall and Macro F1 show whether the model is balanced.",
        ],
    )

with right:
    ## Right column provides compact model-performance profile views.
    chart_key("Polar Metric Profile", "#1f3a5f")
    polar_metric_bars(metrics.to_dict(), "model2_polar")
    visual_notes(
        "Polar metric profile",
        [
            "Data represented: Model 2's combined accuracy, precision, recall, Macro F1, and impaired recall values.",
            "ML impact: the combined profile shows which evaluation goals are strongest or weakest for the tree model.",
            "Model 2 accomplishment: the Decision Tree creates an interpretable performance profile for comparison with Models 1 and 3.",
        ],
        [
            "This summarizes Model 2's key scores together.",
            "Weak areas show where the tree is less reliable.",
            "It helps compare the rule-based model to the others.",
        ],
    )
    chart_key("Radar Metric Profile", "#2a9d8f")
    model_radar_chart(metrics.to_dict(), "Decision Tree", "model2_radar")
    visual_notes(
        "Radar metric profile",
        [
            "Data represented: the same Model 2 performance metrics viewed as one balanced-performance profile.",
            "ML impact: this profile shows whether the Decision Tree is consistently useful or strong only on selected metrics.",
            "Model 2 accomplishment: it summarizes the interpretable model's tradeoffs against Logistic Regression and Random Forest.",
        ],
        [
            "This shows the tree model's overall balance.",
            "Weak points identify metrics needing caution.",
            "It supports comparison with the other model types.",
        ],
    )
    chart_key("Predicted-Class Outcome Stack", "#ff6b6b")
    stacked_outcome_bar(confusion_matrix, "model2_stacked")
    visual_notes(
        "Predicted-class outcome stack",
        [
            "Data represented: Model 2 predicted-class counts within the actual no-impairment and impaired groups.",
            "ML impact: these counts show whether the tree over-predicts or under-predicts impairment for each real class.",
            "Model 2 accomplishment: the Decision Tree exposes class-specific prediction behavior from its learned rules.",
        ],
        [
            "This shows what the tree predicted for each real group.",
            "It highlights missed impaired cases and false alarms.",
            "It explains the class-level behavior behind the metrics.",
        ],
    )
    section("K-Fold Validation and Performance Metrics")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.markdown(
        """
        Model 2 uses 5-fold stratified cross-validation: the GE-79 data is split
        into five folds, each fold is held out once for validation, and
        preprocessing is fit only inside the training folds. The performance
        metrics reported here are accuracy, precision macro, recall macro,
        Macro F1-score, and impaired recall.
        """
    )

section("Original Model 2 Exports")
left_static, right_static = st.columns([1.25, 0.75], gap="large")
with left_static:
    chart_key("Decision Tree Diagram", "#1f3a5f")
    if tree_path.exists():
        st.image(str(tree_path), use_container_width=True)
        visual_notes(
            "Decision tree diagram",
            [
                "Data represented: the learned feature thresholds and branch rules used by Model 2.",
                "ML impact: these splits show which locked predictors the tree uses to separate cognitive-status classes.",
                "Model 2 accomplishment: the Decision Tree provides a human-readable rule structure for the classification task.",
            ],
            [
                "This shows the rules the tree learned from the data.",
                "Each split uses a selected feature to guide prediction.",
                "It helps explain how Model 2 reaches a classification.",
            ],
        )
    else:
        st.warning("Missing tree image: outputs/model2_fig_dt_tree.png")
with right_static:
    chart_key("Confusion Matrix")
    if confusion_path.exists():
        st.image(str(confusion_path), use_container_width=True)
        st.markdown(
            f"""
            <div class="visual-notes">
                <span class="visual-title">Saved confusion-matrix facts</span>
                <ul>
                    <li>The saved image records the same Model 2 prediction counts shown above.</li>
                    <li>Macro F1-score is {metrics['f1_macro']:.3f}; Macro F1 and Macro F1-score refer to the same score in this project.</li>
                    <li>The key interpretation is that Model 2 catches 11 impaired participants and misses 9 impaired participants.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        visual_notes(
            "Original confusion-matrix export",
            [
                "Data represented: the original saved Model 2 confusion-matrix counts from the Decision Tree workflow.",
                "ML impact: this output documents the exact correct and incorrect classifications used for evaluation.",
                "Model 2 accomplishment: it preserves the evidence behind the Decision Tree's reported performance.",
            ],
            [
                "This is the original saved Model 2 confusion matrix.",
                "It records the tree model's correct and incorrect predictions.",
                "It supports checking the dashboard values against the saved workflow output.",
            ],
        )
    else:
        st.warning("Missing confusion-matrix image: outputs/model2_fig_dt_confusion.png")

bottom_export_images(
    "Model 2 Visualization Export",
    [
        ("Model 2 Focused Visualization", str(OUTPUTS_DIR / "model_2_visualization_FOCUSED_EH_pale.png")),
    ],
)

project_footer()
