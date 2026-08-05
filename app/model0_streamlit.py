from pathlib import Path

import pandas as pd
import streamlit as st

## Shared dashboard components used to keep all model pages consistent.
from echarts_components import (
    bottom_export_images,
    feature_domain_sunburst,
    horizontal_importance_chart,
    inject_theme,
    key_panel,
    project_footer,
    project_page_header,
    section,
    selected_features_treemap,
    target_donut,
    visual_notes,
)


## ---- paths to project artifacts displayed by this dashboard ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


## ---- Streamlit page setup ----
st.set_page_config(
    page_title="GE-79 Model 0 Feature Selection",
    page_icon="GE",
    layout="wide",
)
inject_theme()
project_page_header("model0")
st.session_state["visual_takeaways"] = [
    "Model 0 shows which GE-79 biomarker inputs were selected before any final classifier was compared.",
    "The dashboard defines one locked feature set so Models 1, 2, and 3 all use the same predictor information.",
    "The target data shown here contains 55 no-impairment cases and 20 impaired cases, so the dataset is imbalanced.",
    "The feature visuals describe this GE-79 project dataset only; they do not prove that any biomarker causes cognitive impairment.",
]

_, key_col = st.columns([1.45, 1], gap="large")
with key_col:
    ## Explain key terms before showing the model artifacts.
    key_panel(
        {
            "Locked Features": "Final predictor list saved by Model 0 and reused by Models 1-3.",
            "Feature Importance": "How strongly a candidate predictor helped the selection forest separate cognitive-status groups.",
            "Science Anchors": "Clinically relevant glucose, blood-pressure, perfusion, vasoreactivity, and white-matter features kept for modeling.",
            "Target": "Cognitive-status label: no impairment or mild impairment.",
            "Leakage Control": "Excluding MMSE and target fields so models do not learn the answer directly.",
        }
    )

## ---- required Model 0 outputs ----
features_path = OUTPUTS_DIR / "model0_FINAL_FEATURES.csv"
importance_path = OUTPUTS_DIR / "model0_feature_importance_fullscope.csv"
feature_fig_path = OUTPUTS_DIR / "model0_fig_feature_selection.png"
target_fig_path = OUTPUTS_DIR / "model0_fig_target_distribution.png"

if not features_path.exists():
    ## Stop early if Model 0 has not generated its locked feature list.
    st.error("Missing selected-features file: outputs/model0_FINAL_FEATURES.csv")
    st.stop()

features = pd.read_csv(features_path)
importance = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()

## ---- headline metrics ----
metric_cols = st.columns(3)
metric_cols[0].metric("Locked Features", len(features))
metric_cols[1].metric("Dataset", "GE-79")
metric_cols[2].metric("Target", "Cognitive Status")

st.markdown(
    f"""
    <div class="feature-summary">
        <span class="summary-title">Summary</span>
        <strong>How the features were chosen:</strong> Model 0 used a Random Forest
        feature-selection method on the GE-79 candidate biomarker set. The Random
        Forest was run across 20 random seeds, and each feature's importance was
        averaged across those runs so the final ranking was less dependent on one
        noisy split of this small dataset.
        <ul>
            <li><strong>Selection method:</strong> the final locked set combines the top
            12 stability-ranked Random Forest features with science-based anchor
            features from glycemic, blood-pressure, vasoreactivity, perfusion, and
            white-matter domains.</li>
            <li><strong>Why these over others:</strong> features were kept when they either
            ranked highly by mean model importance or represented a clinically
            important domain that should remain available for downstream modeling.</li>
            <li><strong>Leakage control:</strong> MMSE and cognitive-status fields were
            excluded because they define the target, and preprocessing was handled
            inside the feature-selection pipeline.</li>
            <li><strong>Where these features are used:</strong> the locked list in
            <code>outputs/model0_FINAL_FEATURES.csv</code> is reused by Model 1 Logistic
            Regression, Model 2 Decision Tree, and Model 3 Random Forest so all model
            comparisons use the same patient information.</li>
            <li><strong>Current locked feature count:</strong> {len(features)} features.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

section("Feature Selection Overview")

st.markdown("#### Top Full-Scope Feature Importances")
if not importance.empty:
    ## Interactive ECharts version of the saved feature-ranking output.
    horizontal_importance_chart(importance, "model0_importance", limit=14)
    visual_notes(
        "Top full-scope feature importances",
        [
            "Data represented: candidate biomarker, imaging, and clinical predictors ranked by contribution during feature screening.",
            "ML impact: these ranked inputs determine which variables are locked before Models 1-3 are trained and compared.",
            "Model 0 accomplishment: feature selection reduces noisy inputs and creates one consistent predictor set for all downstream models.",
        ],
        [
            "These are the inputs most useful for prediction screening.",
            "The selected variables become the shared model inputs.",
            "This step supports fair comparison across the later ML models.",
        ],
    )
else:
    st.warning("Missing feature-importance file: outputs/model0_feature_importance_fullscope.csv")

st.markdown(
    '<div style="color:#000000;font-size:13px;font-weight:700;margin:0.55rem 0 0.35rem;">Feature Domains</div>',
    unsafe_allow_html=True,
)
feature_domain_sunburst(features["final_features"].tolist(), "model0_feature_domains_tree")
visual_notes(
    "Feature domains tree",
    [
        "Data represented: the locked feature set grouped into glycemic, blood-pressure, perfusion, white-matter, and body/other domains.",
        "ML impact: domain grouping shows whether the models are relying on a balanced set of biomedical signals or one narrow measurement family.",
        "Model 0 accomplishment: it organizes the selected predictors so Models 1-3 can reuse the same feature categories consistently.",
    ],
    [
        "This shows the kinds of health measurements used by the project.",
        "The exact selected features are grouped by meaning.",
        "These groups become the shared inputs for the later models.",
    ],
)

map_col, target_col = st.columns(2, gap="large")

with map_col:
    ## Show the locked features as a treemap for quick visual scanning.
    st.markdown("#### Selected Feature Map")
    selected_features_treemap(features["final_features"].tolist(), "model0_treemap")
    visual_notes(
        "Selected feature map",
        [
            "Data represented: the final locked feature names used as predictors after Model 0 feature selection.",
            "ML impact: using the same locked predictors prevents Models 1-3 from being compared with different input information.",
            "Model 0 accomplishment: it defines the fixed input list that makes downstream Logistic Regression, Decision Tree, and Random Forest results comparable.",
        ],
        [
            "This is the final feature list.",
            "Every later model uses this same input set.",
            "It helps check that no extra target-defining variable slipped in.",
        ],
    )

with target_col:
    ## Show the known class imbalance that downstream models must handle.
    st.markdown("#### Target Distribution")
    target_donut({"No Impairment": 55, "Impaired": 20}, "model0_target_donut")
    visual_notes(
        "Target distribution",
        [
            "Data represented: 55 no-impairment cases and 20 impaired cases in the cognitive-status target.",
            "ML impact: the smaller impaired class makes accuracy misleading and increases the need for impaired recall and Macro F1.",
            "Model 0 accomplishment: it documents the outcome balance that all downstream models must learn from.",
        ],
        [
            "Most participants are in the no-impairment group.",
            "The impaired group is smaller, so it is easier for models to miss.",
            "This is why we track impaired recall separately.",
        ],
    )

section("Final Locked Features")
left_table, right_static = st.columns([0.85, 1.15], gap="large")

with left_table:
    ## Raw locked feature list for checking the exact column names.
    st.dataframe(features, use_container_width=True, hide_index=True)

with right_static:
    ## Include original static exports so the app can be audited against outputs/.
    if feature_fig_path.exists():
        st.markdown("#### Original Feature-Selection Export")
        st.image(str(feature_fig_path), use_container_width=True)
        visual_notes(
            "Original feature-selection export",
            [
                "Data represented: the original saved feature-selection output from the project workflow.",
                "ML impact: this artifact documents which predictors were available before training Models 1-3.",
                "Model 0 accomplishment: it preserves the feature-selection evidence used to justify the locked predictor set.",
            ],
            [
                "This is the saved record of feature selection.",
                "It helps verify the dashboard matches the project analysis.",
                "It documents why the later models use the selected inputs.",
            ],
        )
    if target_fig_path.exists():
        st.markdown("#### Original Target-Distribution Export")
        st.image(str(target_fig_path), use_container_width=True)
        visual_notes(
            "Original target-distribution export",
            [
                "Data represented: the original saved distribution of cognitive-status labels.",
                "ML impact: this class split shapes how the models learn and why minority-class recall matters.",
                "Model 0 accomplishment: it records the class-balance condition used by the downstream classification models.",
            ],
            [
                "This is the original saved target-balance image.",
                "It records how many examples are in each outcome group.",
                "It helps document the dataset split used by the models.",
            ],
        )

    st.markdown(
        """
        Model 0 defines the fixed predictor set used by downstream Models 1, 2,
        and 3. The downstream models reuse these locked features so their results
        are comparable.
        """
    )

bottom_export_images(
    "Feature Visualization Exports",
    [
        ("Feature Visualization Lite", str(OUTPUTS_DIR / "features_visualization_LITE_EH_dark.png")),
        ("Feature Visualization Complete", str(OUTPUTS_DIR / "features_visualization_COMPLETE_EH_dark.png")),
    ],
)

project_footer()
