from __future__ import annotations

## Shared Streamlit/ECharts components used by the model dashboards.
## Keeping charts here makes the four app pages visually consistent.
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd
import streamlit as st

try:
    from streamlit_echarts import st_echarts as _streamlit_echarts
except BaseException as exc:
    _streamlit_echarts = None
    _ECHARTS_IMPORT_ERROR = exc
else:
    _ECHARTS_IMPORT_ERROR = None


def _series_name(series: dict, index: int) -> str:
    return str(series.get("name") or f"Series {index + 1}")


def _category_axis_data(axis_config) -> list[str]:
    axis = axis_config[0] if isinstance(axis_config, list) else axis_config
    if isinstance(axis, dict):
        return [str(item) for item in axis.get("data", [])]
    return []


def _series_values(series: dict) -> list[float]:
    values = []
    for item in series.get("data", []):
        if isinstance(item, dict):
            values.append(float(item.get("value", 0)))
        elif isinstance(item, (list, tuple)):
            values.append(float(item[-1]))
        else:
            values.append(float(item))
    return values


def _render_native_chart(options: dict, key: str) -> None:
    ## Refresh-safe Streamlit fallback for environments where custom components fail.
    series = options.get("series", [])
    if not series:
        st.dataframe(pd.DataFrame(options), use_container_width=True)
        return

    first = series[0]
    chart_type = first.get("type")

    if chart_type == "heatmap":
        x_labels = _category_axis_data(options.get("xAxis", {}))
        y_labels = _category_axis_data(options.get("yAxis", {}))
        matrix = pd.DataFrame(0, index=y_labels, columns=x_labels)
        for x_idx, y_idx, value in first.get("data", []):
            matrix.iloc[int(y_idx), int(x_idx)] = value
        st.dataframe(matrix.style.background_gradient(axis=None, cmap="Blues"), use_container_width=True)
        return

    if chart_type == "pie":
        data = first.get("data", [])
        frame = pd.DataFrame(
            {
                "Class": [item["name"] for item in data],
                "Count": [item["value"] for item in data],
            }
        ).set_index("Class")
        st.bar_chart(frame, use_container_width=True, height=320)
        return

    if chart_type == "gauge":
        data = first.get("data", [{"name": "Score", "value": 0}])[0]
        value = float(data.get("value", 0))
        label = str(data.get("name", "Score"))
        st.metric(label, f"{value:.1f}%")
        st.progress(min(max(value / 100, 0), 1))
        return

    if chart_type == "radar":
        labels = [item["name"] for item in options.get("radar", {}).get("indicator", [])]
        rows = {}
        for idx, item in enumerate(first.get("data", [])):
            rows[str(item.get("name", f"Series {idx + 1}"))] = item.get("value", [])
        frame = pd.DataFrame(rows, index=labels)
        st.line_chart(frame, use_container_width=True, height=320)
        return

    if chart_type == "sankey":
        links = first.get("links", [])
        frame = pd.DataFrame(links)
        if not frame.empty:
            st.dataframe(frame, use_container_width=True, hide_index=True)
            st.bar_chart(frame.set_index("target")["value"], use_container_width=True, height=320)
        return

    if chart_type == "tree":
        def flatten_tree(node: dict, parent: str = "") -> list[dict]:
            rows = [{"Parent": parent, "Feature": node.get("name", ""), "Value": node.get("value", "")}]
            for child in node.get("children", []):
                rows.extend(flatten_tree(child, node.get("name", "")))
            return rows

        tree_rows = []
        for node in first.get("data", []):
            tree_rows.extend(flatten_tree(node))
        st.dataframe(pd.DataFrame(tree_rows), use_container_width=True, hide_index=True)
        return

    if chart_type == "treemap":
        data = first.get("data", [])
        frame = pd.DataFrame(
            {
                "Feature": [item.get("name", "") for item in data],
                "Value": [item.get("value", 1) for item in data],
            }
        ).set_index("Feature")
        st.bar_chart(frame, use_container_width=True, height=320)
        return

    y_labels = _category_axis_data(options.get("yAxis", {}))
    x_labels = _category_axis_data(options.get("xAxis", {}))
    labels = y_labels or x_labels

    if labels:
        frame = pd.DataFrame(
            {
                _series_name(item, idx): _series_values(item)
                for idx, item in enumerate(series)
            },
            index=labels,
        )
    else:
        frame = pd.DataFrame(
            {
                _series_name(item, idx): _series_values(item)
                for idx, item in enumerate(series)
            }
        )

    if any(item.get("type") == "line" for item in series):
        st.line_chart(frame, use_container_width=True, height=320)
    else:
        st.bar_chart(frame, use_container_width=True, height=320)


def st_echarts(options: dict, height: str, key: str, **kwargs):
    ## Use ECharts when available; native Streamlit charts are the safety net.
    if _streamlit_echarts is not None:
        try:
            return _streamlit_echarts(options=options, height=height, key=key, **kwargs)
        except Exception as exc:
            st.caption(f"ECharts render fallback: {exc.__class__.__name__}")

    elif _ECHARTS_IMPORT_ERROR is not None:
        st.caption(f"ECharts import fallback: {_ECHARTS_IMPORT_ERROR.__class__.__name__}")

    _render_native_chart(options, key)
    return None


## ---- shared design tokens and copy used across dashboard pages ----
PALETTE = {
    "navy": "#1f3a5f",
    "blue": "#2f80ed",
    "sky": "#78c6ff",
    "teal": "#2a9d8f",
    "green": "#22c55e",
    "yellow": "#f2c94c",
    "coral": "#ff6b6b",
    "orange": "#f2994a",
    "ink": "#2b2f3a",
    "muted": "#6b7280",
    "grid": "#e7edf5",
}

APP_LINKS = {
    "model0": {
        "badge": "Model 0",
        "label": "Feature Selection",
        "url": "https://ai4all-diabetes-ml-model-0-features.streamlit.app/",
    },
    "model1": {
        "badge": "Model 1",
        "label": "Logistic Regression",
        "url": "https://ai4all-diabetes-app-ml-model-1-logistic-regression.streamlit.app/",
    },
    "model2": {
        "badge": "Model 2",
        "label": "Decision Tree",
        "url": "https://ai4all-diabetes-app-ml-model-2-decision-tree.streamlit.app/",
    },
    "model3": {
        "badge": "Model 3",
        "label": "Random Forest",
        "url": "https://ai4all-diabetes-app-ml-model-3-random-forest.streamlit.app/",
    },
    "math": {
        "badge": "Math",
        "label": "Formula Reference",
        "url": "https://ai4all-diabetes-math-formulas.streamlit.app/",
    },
    "bias": {
        "badge": "Bias Review",
        "label": "Responsible AI & Safety",
        "url": "https://i4all-diabetes-ml-bias-report.streamlit.app/",
    },
}

PAGE_HEADERS = {
    "model0": {
        "title": "Model 0 • Random Forest Feature Selection",
        "description": (
            "Model 0 evaluates all candidate demographic, clinical, cardiovascular, "
            "functional, ophthalmologic, and MRI-derived biomarkers using Random Forest "
            "feature importance. The highest-ranking predictors are retained as the final "
            "biomarker set used consistently across Models 1-3. Feature selection is "
            "entirely data-driven, reducing noise while improving model stability, "
            "reproducibility, and downstream model performance."
        ),
    },
    "model1": {
        "title": "Model 1 • Logistic Regression",
        "description": (
            "Logistic Regression serves as the project's baseline classifier for predicting "
            "Mild Cognitive Impairment. As a linear model, it provides an interpretable "
            "benchmark for evaluating the predictive value of the selected biomarkers before "
            "comparing more complex machine learning approaches."
        ),
    },
    "model2": {
        "title": "Model 2 • Decision Tree",
        "description": (
            "The Decision Tree classifier generates interpretable decision rules by "
            "recursively partitioning the selected biomarkers. This model illustrates how "
            "demographic, clinical, cardiovascular, and MRI-derived variables contribute to "
            "classification while maintaining transparent decision pathways."
        ),
    },
    "model3": {
        "title": "Model 3 • Random Forest Ensemble",
        "description": (
            "Random Forest combines multiple decision trees into an ensemble classifier to "
            "improve predictive stability and reduce overfitting. Performance is evaluated "
            "using stratified cross-validation, ROC-AUC, Precision-Recall, confusion "
            "matrices, feature importance, and standard classification metrics."
        ),
    },
    "math": {
        "title": "Mathematical Formula Reference",
        "description": (
            "This page centralizes the mathematical formulas used across the GE-79 model "
            "dashboards. It shows model equations, variable definitions, and evaluation "
            "metrics in one presentation-ready reference."
        ),
    },
    "bias": {
        "title": "Responsible AI • Bias & Safety Assessment",
        "description": (
            "This application summarizes the project's Responsible AI evaluation, including "
            "dataset auditing, exploratory data analysis, bias identification, fairness "
            "assessment, model transparency, and governance practices. Bias mitigation "
            "follows principles from the NIST AI Risk Management Framework (AI RMF), OECD "
            "AI Principles, and VerifyWise fairness analysis. The review documents known "
            "dataset limitations, including geographic, selection, demographic, survivorship, "
            "and representation bias, and demonstrates how these risks were evaluated "
            "throughout the machine learning lifecycle."
        ),
    },
}

TEXT_COLOR = "#111827"
TEXT_SIZE = 13
CHART_TITLE_SIZE = 18

CHART_BACKGROUNDS = [
    "#f2f8ff",
    "#f3fbf8",
    "#fff8ef",
    "#fff4f4",
    "#f7f4ff",
    "#f4fbff",
]

CHART_TITLES = {
    "accuracy_gauge": "Accuracy Score",
    "f1_gauge": "Macro F1 Score",
    "recall_gauge": "Impaired Recall Score",
    "confusion": "Confusion Matrix",
    "sankey": "Actual-to-Predicted Flow",
    "metric_bars": "Cross-Validation Metric Bars",
    "polar": "Polar Metric Profile",
    "radar": "Radar Metric Profile",
    "stacked": "Predicted-Class Outcome Stack",
    "importance": "Feature Importance",
    "feature_domains": "Feature Domains",
    "mixed_combo": "Mixed Line/Bar Metric Comparison",
    "matrix_display": "Comparison Matrix Display",
    "comparison_radar": "Model Comparison Radar",
}


def inject_theme() -> None:
    ## Inject dashboard CSS once per page for consistent typography and panels.
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f9fc;
        }
        [data-testid="stHeader"] {
            background: rgba(247, 249, 252, 0.84);
            backdrop-filter: blur(10px);
        }
        h1, h2, h3, h4, h5, h6,
        p, li, span, label,
        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"],
        [data-testid="stWidgetLabel"],
        [data-testid="stDataFrame"],
        [data-testid="stSelectbox"] {
            color: #111827 !important;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        h4 {
            font-size: 18px !important;
            line-height: 1.25 !important;
            margin: 0.55rem 0 0.25rem !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e4eaf3;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 8px 22px rgba(31, 58, 95, 0.06);
        }
        div[data-testid="stMetricLabel"] {
            color: #111827 !important;
            font-size: 13px !important;
        }
        div[data-testid="stMetricValue"] {
            color: #111827 !important;
        }
        .section-note {
            color: #111827;
            font-size: 13px;
            margin: -0.4rem 0 1.1rem;
        }
        .chart-key {
            color: #111827;
            font-size: 18px;
            line-height: 1.35;
            font-weight: 600;
            padding: 0.35rem 0.55rem;
            border-left: 4px solid #2f80ed;
            background: rgba(255, 255, 255, 0.72);
            border-radius: 6px;
            margin: 0.25rem 0 0.45rem;
        }
        .visual-notes {
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid #e4eaf3;
            border-radius: 8px;
            padding: 0.65rem 0.85rem;
            margin: 0.25rem 0 1rem;
            color: #111827;
            font-size: 16px;
            line-height: 1.38;
        }
        .visual-notes .visual-title {
            color: #111827;
            font-size: 18px;
            line-height: 1.25;
            font-weight: 700;
        }
        .visual-notes strong {
            color: #111827;
            font-size: 16px;
        }
        .visual-notes ul {
            margin: 0.25rem 0 0.55rem 1.1rem;
            padding: 0;
        }
        .visual-notes li {
            margin: 0.12rem 0;
        }
        div[data-testid="stIFrame"],
        div[data-testid="stVegaLiteChart"],
        div[data-testid="stPlotlyChart"],
        div[data-testid="stImage"] {
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
            overflow: hidden;
            background: #ffffff;
            border: 1px solid #d8e2ea;
            border-radius: 8px;
            padding: 0.45rem;
            box-shadow: 0 8px 22px rgba(31, 58, 95, 0.05);
        }
        div[data-testid="stIFrame"] iframe,
        div[data-testid="stVegaLiteChart"] svg,
        div[data-testid="stVegaLiteChart"] canvas,
        div[data-testid="stPlotlyChart"] svg,
        div[data-testid="stImage"] img {
            max-width: 100% !important;
            box-sizing: border-box;
        }
        div[data-testid="stIFrame"] iframe {
            display: block;
            border: 0;
            border-radius: 6px;
            overflow: hidden;
        }
        .model-context {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid #e4eaf3;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin: 0.8rem 0 1rem;
            color: #111827;
            font-size: 14px;
            line-height: 1.42;
        }
        .model-context strong {
            color: #111827;
            font-size: 14px;
        }
        .feature-summary {
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid #e4eaf3;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 0.85rem 0 1.05rem;
            color: #111827;
            font-size: 14px;
            line-height: 1.45;
        }
        .feature-summary strong {
            color: #111827;
            font-size: 14px;
        }
        .feature-summary .summary-title {
            display: block;
            color: #111827;
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .feature-summary ul {
            margin: 0.25rem 0 0 1.1rem;
            padding: 0;
        }
        .feature-summary li {
            margin: 0.16rem 0;
        }
        .model-context ul {
            margin: 0.25rem 0 0.55rem 1.1rem;
            padding: 0;
        }
        .model-context li {
            margin: 0.12rem 0;
        }
        .key-panel {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid #dbe5f1;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin: 0.8rem 0 1rem;
            color: #111827;
            font-size: 14px;
            line-height: 1.35;
        }
        .key-panel .key-title {
            display: block;
            color: #111827;
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .key-panel dl {
            margin: 0;
        }
        .key-panel dt {
            color: #111827;
            font-weight: 700;
            margin-top: 0.35rem;
        }
        .key-panel dd {
            color: #111827;
            margin: 0.05rem 0 0.15rem 0;
        }
        .formula-panel {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #dbe5f1;
            border-left: 5px solid #2a9d8f;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin: 0.8rem 0 1rem;
            color: #111827;
            font-size: 14px;
            line-height: 1.42;
        }
        .formula-panel .formula-title {
            display: block;
            color: #111827;
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }
        .formula-panel .formula-expression {
            display: block;
            background: #f7f9fc;
            border: 1px solid #e4eaf3;
            border-radius: 6px;
            color: #111827;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 15px;
            font-weight: 700;
            line-height: 1.5;
            padding: 0.55rem 0.65rem;
            margin: 0.25rem 0 0.55rem;
            overflow-wrap: anywhere;
        }
        .formula-panel .formula-purpose {
            color: #111827;
            margin: 0;
        }
        .model-context a,
        .key-panel a,
        .formula-panel a {
            color: #0f5fb8 !important;
            font-weight: 700;
            text-decoration: none;
        }
        .model-context a:hover,
        .key-panel a:hover,
        .formula-panel a:hover {
            text-decoration: underline;
        }
        .project-header {
            background: #ffffff;
            border: 1px solid #dbe5f1;
            border-radius: 8px;
            padding: 1rem 1.15rem;
            margin: 0.5rem 0 1rem;
        }
        .project-kicker {
            color: #111827;
            font-size: 13px;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .project-name {
            color: #111827;
            font-size: 24px;
            font-weight: 800;
            margin-top: 0.2rem;
        }
        .project-subtitle {
            color: #1f2937;
            font-size: 16px;
            font-weight: 650;
            line-height: 1.35;
            margin-top: 0.3rem;
        }
        .page-title {
            color: #111827;
            font-size: 28px;
            font-weight: 850;
            margin-top: 0.9rem;
        }
        .page-description {
            color: #1f2937;
            font-size: 16px;
            line-height: 1.55;
            margin-top: 0.35rem;
        }
        .project-nav {
            background: #ffffff;
            border: 1px solid #dbe5f1;
            border-radius: 8px;
            padding: 0.9rem;
            margin: 0.5rem 0 1rem;
        }
        .project-nav-title {
            display: block;
            color: #111827;
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 0.55rem;
        }
        .project-nav a,
        .project-nav .nav-current {
            display: block;
            color: #111827 !important;
            text-decoration: none;
            border: 1px solid #cfd9e6;
            border-radius: 8px;
            padding: 0.55rem 0.65rem;
            margin-bottom: 0.45rem;
            background: #f8fafc;
        }
        .project-nav a:hover {
            border-color: #2563eb;
            background: #eef6ff;
        }
        .project-nav .nav-current {
            border-color: #2563eb;
            background: #dceeff;
            box-shadow: inset 4px 0 0 #2563eb;
        }
        .nav-badge {
            display: inline-block;
            border: 1px solid #111827;
            border-radius: 5px;
            padding: 0.08rem 0.38rem;
            margin-right: 0.35rem;
            font-size: 12px;
            font-weight: 850;
            color: #111827;
            background: #ffffff;
        }
        .nav-label {
            color: #111827;
            font-size: 13px;
            font-weight: 700;
        }
        .project-nav-resources {
            border-top: 1px solid #dbe5f1;
            margin-top: 0.65rem;
            padding-top: 0.65rem;
            color: #111827;
            font-size: 13px;
            line-height: 1.45;
        }
        .project-nav-resources a {
            display: inline;
            border: 0;
            border-radius: 0;
            padding: 0;
            margin: 0;
            background: transparent;
            color: #0f5fb8 !important;
            font-weight: 700;
            text-decoration: none;
        }
        .project-nav-resources a:hover {
            background: transparent;
            border: 0;
            text-decoration: underline;
        }
        .pipeline-box {
            border: 1px solid #9ca3af;
            border-radius: 8px;
            padding: 0.65rem;
            margin: 0.45rem 0;
            background: #ffffff;
            color: #111827;
            font-size: 13px;
            line-height: 1.3;
            text-align: center;
        }
        .pipeline-arrow {
            color: #111827;
            font-size: 18px;
            font-weight: 900;
            text-align: center;
            margin: 0.1rem 0;
        }
        .project-footer {
            background: #ffffff;
            border: 1px solid #dbe5f1;
            border-radius: 8px;
            padding: 1rem 1.15rem;
            margin: 2rem 0 0.75rem;
        }
        .project-footer-title {
            color: #111827;
            font-size: 18px;
            font-weight: 850;
            margin-bottom: 0.5rem;
        }
        .project-footer p,
        .project-footer li {
            color: #111827;
            font-size: 13px;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, note: str | None = None) -> None:
    st.subheader(title)
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)


def chart_key(text: str, accent: str = "#2f80ed") -> None:
    st.markdown(
        f'<div class="chart-key" style="border-left-color:{accent}">{text}</div>',
        unsafe_allow_html=True,
    )


def visual_notes(title: str, technical: list[str], defined: list[str]) -> None:
    technical_items = "".join(f"<li>{item}</li>" for item in technical)
    defined_items = "".join(f"<li>{item}</li>" for item in defined)
    takeaways = st.session_state.get(
        "visual_takeaways",
        [
            "This visualization summarizes the saved GE-79 project output currently shown on this page.",
            "The result applies to this project dataset and should not be read as a clinical decision rule.",
            "The chart should be interpreted with the documented model metrics and class balance.",
        ],
    )
    takeaway_items = "".join(f"<li>{item}</li>" for item in takeaways)
    st.markdown(
        f"""
        <div class="visual-notes">
            <span class="visual-title">{title}</span>
            <br/><strong>Technical:</strong>
            <ul>{technical_items}</ul>
            <strong>Defined:</strong>
            <ul>{defined_items}</ul>
            <strong>Takeaways:</strong>
            <ul>{takeaway_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def formula_reference_notes(title: str, technical: list[str], described: list[str], takeaways: list[str]) -> None:
    ## Shared note card for model formula sections.
    technical_items = "".join(f"<li>{item}</li>" for item in technical)
    described_items = "".join(f"<li>{item}</li>" for item in described)
    takeaway_items = "".join(f"<li>{item}</li>" for item in takeaways)
    st.markdown(
        f"""
        <div class="visual-notes">
            <span class="visual-title">{title}</span>
            <br/><strong>Technical:</strong>
            <ul>{technical_items}</ul>
            <strong>Described:</strong>
            <ul>{described_items}</ul>
            <strong>Takeaways:</strong>
            <ul>{takeaway_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def why_formulas_matter_expander() -> None:
    ## Shared final expander for formula transparency.
    with st.expander("Why These Formulas Matter"):
        st.markdown(
            """
            These formulas make the model pipeline mathematically transparent.
            They show how predictions were generated, how errors were measured,
            and why model performance was evaluated beyond simple accuracy.
            """
        )


def display_math_section(
    title: str,
    latex_formula: Optional[Union[str, Iterable[str]]] = None,
    variable_definitions: Optional[Union[dict[str, str], str]] = None,
    explanation: Optional[str] = None,
    why_it_matters: Optional[str] = None,
) -> None:
    ## Reusable math interpretation section for model pages.
    st.header("Mathematical Formula & Model Interpretation")

    st.subheader("Model Formula")
    st.markdown(f"#### {title}")
    if latex_formula:
        formulas = [latex_formula] if isinstance(latex_formula, str) else latex_formula
        for formula in formulas:
            st.latex(formula)
    else:
        st.markdown("_Formula placeholder: add the model-specific equation here._")

    st.subheader("Variable Definitions")
    if isinstance(variable_definitions, dict):
        for variable, definition in variable_definitions.items():
            st.markdown(f"- **{variable}:** {definition}")
    elif variable_definitions:
        st.markdown(variable_definitions)
    else:
        st.markdown("_Variable definitions placeholder: define each term in the formula here._")

    st.subheader("Plain-English Explanation")
    if explanation:
        st.markdown(explanation)
    else:
        st.markdown(
            "_Explanation placeholder: describe what the formula means in accessible language._"
        )

    with st.expander("Why This Formula Matters"):
        if why_it_matters:
            st.markdown(why_it_matters)
        else:
            st.markdown(
                "_Interpretation placeholder: explain why this formula is important for understanding the model._"
            )


def display_performance_metric_formulas() -> None:
    ## Shared evaluation formula section for GE-79 binary classification pages.
    st.subheader("Performance Metric Formulas")

    st.markdown("#### Accuracy")
    st.latex(r"\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}")
    st.markdown(
        """
        Accuracy measures the overall share of correct predictions. It is useful
        for a quick performance summary, but GE-79 also requires class-sensitive
        metrics because the binary target is imbalanced.
        """
    )

    st.markdown("#### Precision")
    st.latex(r"\text{Precision}=\frac{TP}{TP+FP}")
    st.markdown(
        """
        Precision measures how often predicted impaired cases are truly impaired.
        It is important for GE-79 because false alarms can affect how model
        outputs are interpreted in a screening context.
        """
    )

    st.markdown("#### Recall")
    st.latex(r"\text{Recall}=\frac{TP}{TP+FN}")
    st.markdown(
        """
        Recall measures how many truly impaired participants the model catches.
        It is appropriate for this binary classification project because missed
        impaired cases are a key safety and interpretation concern.
        """
    )

    st.markdown("#### F1 Score")
    st.latex(r"F1=2\times\frac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}")
    st.markdown(
        """
        F1 Score balances precision and recall in one metric. It is useful for
        GE-79 because it summarizes the tradeoff between false alarms and missed
        impaired cases.
        """
    )

    st.markdown("#### Macro F1")
    st.latex(r"\text{Macro F1}=\frac{F1_{\text{class 0}}+F1_{\text{class 1}}}{2}")
    st.markdown(
        """
        Macro F1 averages F1 across both cognitive-status classes equally. This
        is important for GE-79 because the impaired class is smaller and should
        not be hidden by majority-class performance.
        """
    )

    st.markdown("#### Cross Validation")
    st.latex(r"\text{CV}=\frac{S_1+S_2+S_3+S_4+S_5}{5}")
    st.markdown(
        """
        Cross validation averages model scores across five validation folds. It
        is appropriate for GE-79 because the dataset is small, so performance
        should not depend on one train-test split.
        """
    )


def final_math_context_expander(model_choice_reason: str) -> None:
    ## Shared bottom-page expander connecting model math, metrics, and Responsible AI.
    with st.expander("Why These Mathematics Matter"):
        st.markdown("#### 1. Why this model was chosen")
        st.markdown(model_choice_reason)

        st.markdown("#### 2. Why these formulas matter")
        st.markdown(
            """
            The model equations show how GE-79 biomarker inputs are converted
            into a prediction or ranking. They make the prediction process easier
            to explain, audit, and compare across models.
            """
        )

        st.markdown("#### 3. Why these evaluation metrics matter")
        st.markdown(
            """
            Accuracy alone is insufficient for GE-79 because the medical dataset
            is imbalanced. Precision, Recall, and Macro F1 show whether the model
            performs responsibly across both cognitive-status classes.
            """
        )

        st.markdown("#### 4. Responsible AI")
        st.markdown(
            """
            Explainable AI, Feature Importance, Cross Validation, and Bias
            Analysis improve transparency, reproducibility, and trustworthiness.
            Together, they help reviewers understand both model performance and
            model limitations.
            """
        )


def key_panel(items: dict[str, str]) -> None:
    definitions = "".join(
        f"<dt>{term}</dt><dd>{definition}</dd>"
        for term, definition in items.items()
    )
    st.markdown(
        f"""
        <div class="key-panel">
            <span class="key-title">[Key]</span>
            <dl>{definitions}</dl>
        </div>
        """,
        unsafe_allow_html=True,
    )


def formula_panel(
    model_name: str,
    formula: str,
    purpose: str,
    latex_formula: str | None = None,
) -> None:
    with st.container(border=True):
        st.markdown(f"#### {model_name} Formula")
        if latex_formula:
            st.latex(latex_formula)
        else:
            st.code(formula, language="text")
        st.markdown(f"**Purpose:** {purpose}")


def metric_formula_panel(model_label: str) -> None:
    ## Shared evaluation-math panel for Models 1-3.
    with st.container(border=True):
        st.markdown(f"#### {model_label} Evaluation Formulas")
        st.markdown(
            """
            These formulas define the k-fold validation estimate and the
            performance metrics reported for this model.
            """
        )
        st.markdown("**K-fold cross-validation**")
        st.latex(r"\text{CV score}=\frac{1}{K}\sum_{k=1}^{K}M_k,\quad K=5")
        st.markdown("**Confusion-matrix terms**")
        st.latex(r"TP=\text{true positives},\quad TN=\text{true negatives},\quad FP=\text{false positives},\quad FN=\text{false negatives}")
        st.markdown("**Performance metrics**")
        st.latex(r"\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}")
        st.latex(r"\text{Precision}=\frac{TP}{TP+FP}")
        st.latex(r"\text{Recall}=\frac{TP}{TP+FN}")
        st.latex(r"F1=\frac{2\cdot\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}")
        st.latex(r"\text{Macro F1}=\frac{F1_{\text{No Impairment}}+F1_{\text{Impaired}}}{2}")
        st.caption("Macro F1 and Macro F1-score mean the same metric in these dashboards.")


def _navigation_markup(current_page: str) -> str:
    links = []
    for page_key, page in APP_LINKS.items():
        content = (
            f'<span class="nav-badge">{page["badge"]}</span>'
            f'<span class="nav-label">{page["label"]}</span>'
        )
        if page_key == current_page:
            links.append(f'<div class="nav-current">{content}</div>')
        else:
            links.append(f'<a href="{page["url"]}">{content}</a>')
    return "\n".join(links)


def project_navigation(current_page: str) -> None:
    st.markdown(
        f"""
        <div class="project-nav">
            <span class="project-nav-title">Project Navigation</span>
            {_navigation_markup(current_page)}
            <div class="project-nav-resources">
                <strong>Resources:</strong> [insert GDrive link] and [insert GitHub link]<br/>
                <strong>Dataset Sources:</strong> [insert GE-79 link] and [insert GE-75 link]<br/>
                <strong>LinkedIn:</strong>
                <a href="http://www.linkedin.com/in/elizabethhannan">/in/ElizabethHannan</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def project_page_header(current_page: str) -> None:
    page = PAGE_HEADERS[current_page]
    header_col, nav_col = st.columns([1.55, 0.85], gap="large")
    with header_col:
        st.markdown(
            f"""
            <div class="project-header">
                <div class="project-kicker">AI4ALL Ignite 2026</div>
                <div class="project-name">GE-79 Machine Learning Project</div>
                <div class="project-subtitle">
                    Predicting Mild Cognitive Impairment in Older Adults with Type 2 Diabetes<br/>
                    Using Clinical and MRI-Derived Biomarkers
                </div>
                <div class="page-title">{page["title"]}</div>
                <div class="page-description">{page["description"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nav_col:
        project_navigation(current_page)


def pipeline_sidebar(current_page: str) -> None:
    steps = [
        ("model0", "Model 0", "Random Forest", "Feature Selection"),
        ("model1", "Model 1", "Logistic Regression", ""),
        ("model2", "Model 2", "Decision Tree", ""),
        ("model3", "Model 3", "Random Forest Ensemble", ""),
        ("math", "Math", "Formula Reference", ""),
        ("bias", "Responsible AI", "Bias & Safety Review", ""),
    ]
    st.sidebar.markdown("### GE-79 Machine Learning Pipeline")
    for index, (page_key, line_1, line_2, line_3) in enumerate(steps):
        current_style = "border-color:#2563eb; background:#dceeff;" if page_key == current_page else ""
        third_line = f"<br/>{line_3}" if line_3 else ""
        st.sidebar.markdown(
            f"""
            <div class="pipeline-box" style="{current_style}">
                <strong>{line_1}</strong><br/>
                {line_2}{third_line}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if index < len(steps) - 1:
            st.sidebar.markdown('<div class="pipeline-arrow">↓</div>', unsafe_allow_html=True)


def project_footer() -> None:
    st.markdown(
        """
        <div class="project-footer">
            <div class="project-footer-title">GE-79 Machine Learning Research Pipeline</div>
            <p><strong>Dataset:</strong><br/>PhysioNet CDED (GE-79)</p>
            <p><strong>Evaluation:</strong></p>
            <ul>
                <li>Accuracy</li>
                <li>Precision</li>
                <li>Recall</li>
                <li>Macro F1-Score</li>
                <li>ROC-AUC</li>
                <li>Precision-Recall AUC</li>
                <li>Confusion Matrix</li>
                <li>5-Fold Stratified Cross-Validation</li>
            </ul>
            <p><strong>Responsible AI:</strong></p>
            <ul>
                <li>Dataset Auditing</li>
                <li>Exploratory Data Analysis (EDA)</li>
                <li>Bias Assessment</li>
                <li>Leakage Prevention</li>
                <li>Feature Selection</li>
                <li>VerifyWise Fairness Review</li>
                <li>NIST AI Risk Management Framework (AI RMF)</li>
                <li>OECD AI Principles</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bottom_export_images(title: str, image_paths: list[tuple[str, str]]) -> None:
    existing = [(label, path) for label, path in image_paths if Path(path).exists()]
    if not existing:
        return
    st.markdown("---")
    st.subheader(title)
    for label, path in existing:
        st.markdown(f"#### {label}")
        st.image(path, use_container_width=True)


def _chart_title(key: str, fallback: str) -> str:
    for token, title in CHART_TITLES.items():
        if token in key:
            return title
    return fallback


def _chart_background(key: str) -> str:
    return CHART_BACKGROUNDS[sum(ord(char) for char in key) % len(CHART_BACKGROUNDS)]


def _normalize_text_styles(value):
    if isinstance(value, dict):
        normalized = {}
        for item_key, item_value in value.items():
            normalized[item_key] = _normalize_text_styles(item_value)
        if any(
            token in normalized
            for token in ["formatter", "fontSize", "fontWeight", "color", "show", "position", "align", "verticalAlign"]
        ):
            normalized.setdefault("color", TEXT_COLOR)
            normalized.setdefault("fontSize", TEXT_SIZE)
        return normalized
    if isinstance(value, list):
        return [_normalize_text_styles(item) for item in value]
    return value


def _decorate_options(options: dict, key: str, fallback_title: str) -> dict:
    options = _normalize_text_styles(dict(options))
    options.setdefault("backgroundColor", _chart_background(key))
    options.setdefault("textStyle", {"color": TEXT_COLOR, "fontSize": TEXT_SIZE})
    options.setdefault(
        "title",
        {
            "text": _chart_title(key, fallback_title),
            "left": 12,
            "top": 8,
            "textStyle": {"color": TEXT_COLOR, "fontSize": CHART_TITLE_SIZE, "fontWeight": 700},
            "subtextStyle": {"color": TEXT_COLOR, "fontSize": TEXT_SIZE},
        },
    )
    titles = options["title"] if isinstance(options["title"], list) else [options["title"]]
    for title in titles:
        if isinstance(title, dict):
            text_style = title.setdefault("textStyle", {})
            text_style["color"] = TEXT_COLOR
            text_style["fontSize"] = CHART_TITLE_SIZE
            title.setdefault("subtextStyle", {"color": TEXT_COLOR, "fontSize": TEXT_SIZE})
    if "grid" in options and isinstance(options["grid"], dict):
        top = options["grid"].get("top", 20)
        options["grid"].setdefault("containLabel", True)
        if isinstance(top, int) and top < 50:
            options["grid"] = {**options["grid"], "top": 52}
    elif "grid" in options and isinstance(options["grid"], list):
        for grid in options["grid"]:
            if isinstance(grid, dict):
                grid.setdefault("containLabel", True)
    return options


def _axis_common() -> dict:
    return {
        "axisLine": {"lineStyle": {"color": "#d4dce8"}},
        "axisLabel": {"color": TEXT_COLOR, "fontSize": TEXT_SIZE},
        "nameTextStyle": {"color": TEXT_COLOR, "fontSize": TEXT_SIZE},
        "splitLine": {"lineStyle": {"color": PALETTE["grid"]}},
    }


def metric_bar_chart(results: pd.DataFrame, key: str) -> None:
    metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "recall_impaired"]
    labels = ["Accuracy", "Precision", "Recall", "Macro F1", "Impaired Recall"]
    series = []
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["coral"]]
    for idx, row in results.iterrows():
        series.append(
            {
                "name": row["model"],
                "type": "bar",
                "barMaxWidth": 18,
                "itemStyle": {"borderRadius": [5, 5, 0, 0], "color": colors[idx % len(colors)]},
                "emphasis": {"focus": "series"},
                "data": [round(float(row[col]), 3) for col in metric_cols if col in row],
            }
        )
    options = {
        "color": colors,
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"bottom": 0, "textStyle": {"color": TEXT_COLOR, "fontSize": TEXT_SIZE}},
        "grid": {"left": 42, "right": 28, "top": 24, "bottom": 64, "containLabel": True},
        "xAxis": {"type": "category", "data": labels, **_axis_common()},
        "yAxis": {"type": "value", "min": 0, "max": 1, **_axis_common()},
        "series": series,
        "animationDuration": 900,
        "animationEasing": "cubicOut",
    }
    st_echarts(options=_decorate_options(options, key, "Metric Chart"), height="360px", key=key)


def model_radar_chart(metrics: dict[str, float], model_name: str, key: str) -> None:
    labels = [
        ("Accuracy", metrics["accuracy"]),
        ("Precision", metrics["precision_macro"]),
        ("Recall", metrics["recall_macro"]),
        ("Macro F1", metrics["f1_macro"]),
        ("Impaired Recall", metrics["recall_impaired"]),
    ]
    options = {
        "tooltip": {},
        "radar": {
            "radius": "68%",
            "indicator": [{"name": name, "max": 1} for name, _ in labels],
            "axisName": {"color": TEXT_COLOR, "fontSize": TEXT_SIZE},
            "splitLine": {"lineStyle": {"color": ["#dbe5f1"]}},
            "splitArea": {"areaStyle": {"color": ["rgba(47,128,237,0.04)", "rgba(42,157,143,0.05)"]}},
        },
        "series": [
            {
                "name": model_name,
                "type": "radar",
                "areaStyle": {"color": "rgba(47,128,237,0.18)"},
                "lineStyle": {"width": 3, "color": PALETTE["blue"]},
                "symbolSize": 7,
                "data": [{"value": [round(v, 3) for _, v in labels], "name": model_name}],
            }
        ],
        "animationDuration": 1000,
        "animationEasing": "elasticOut",
    }
    st_echarts(options=_decorate_options(options, key, "Metric Chart"), height="360px", key=key)


def confusion_heatmap(matrix: list[list[int]], key: str) -> None:
    labels = ["No Impairment", "Impaired"]
    data = [[j, i, matrix[i][j]] for i in range(2) for j in range(2)]
    options = {
        "tooltip": {
            "position": "top",
            "formatter": "{b0}: {c}",
        },
        "grid": {"left": 96, "right": 42, "top": 36, "bottom": 62, "containLabel": True},
        "xAxis": {"type": "category", "data": labels, "name": "Predicted", **_axis_common()},
        "yAxis": {"type": "category", "data": labels, "name": "Actual", **_axis_common()},
        "visualMap": {
            "min": 0,
            "max": max(max(row) for row in matrix),
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 0,
            "inRange": {"color": ["#edf7ff", "#78c6ff", "#1f75cb"]},
        },
        "series": [
            {
                "type": "heatmap",
                "data": data,
                "label": {"show": True, "fontSize": TEXT_SIZE, "fontWeight": "bold", "color": TEXT_COLOR},
                "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(31,58,95,0.28)"}},
            }
        ],
        "animationDuration": 900,
    }
    st_echarts(options=_decorate_options(options, key, "Metric Chart"), height="360px", key=key)


def horizontal_importance_chart(df: pd.DataFrame, key: str, limit: int = 12) -> None:
    chart = df.head(limit).copy()
    value_col = "importance_mean" if "importance_mean" in chart.columns else "importance"
    chart = chart.iloc[::-1]
    options = {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 180, "right": 36, "top": 24, "bottom": 36, "containLabel": True},
        "xAxis": {"type": "value", **_axis_common()},
        "yAxis": {
            "type": "category",
            "data": chart["feature"].tolist(),
            "axisLabel": {"color": PALETTE["ink"], "width": 150, "overflow": "truncate"},
            "axisLine": {"show": False},
        },
        "series": [
            {
                "type": "bar",
                "data": [round(float(v), 4) for v in chart[value_col]],
                "barWidth": 14,
                "itemStyle": {"borderRadius": [0, 6, 6, 0], "color": PALETTE["teal"]},
            }
        ],
        "animationDuration": 1000,
        "animationEasing": "cubicOut",
    }
    st_echarts(options=_decorate_options(options, key, "Feature Importance"), height="430px", key=key)


def target_donut(values: dict[str, int], key: str) -> None:
    options = {
        "tooltip": {"trigger": "item"},
        "legend": {"bottom": 0, "textStyle": {"color": TEXT_COLOR, "fontSize": TEXT_SIZE}},
        "series": [
            {
                "type": "pie",
                "radius": ["48%", "72%"],
                "avoidLabelOverlap": True,
                "itemStyle": {"borderRadius": 8, "borderColor": "#fff", "borderWidth": 3},
                "label": {"formatter": "{b}\\n{d}%", "color": PALETTE["ink"]},
                "data": [
                    {"name": name, "value": value}
                    for name, value in values.items()
                ],
            }
        ],
        "color": [PALETTE["blue"], PALETTE["coral"], PALETTE["teal"]],
        "animationType": "scale",
        "animationEasing": "elasticOut",
        "animationDelay": 120,
    }
    st_echarts(options=_decorate_options(options, key, "Metric Chart"), height="360px", key=key)


def selected_features_treemap(features: Iterable[str], key: str) -> None:
    data = [{"name": feature, "value": 1} for feature in features]
    options = {
        "tooltip": {"formatter": "{b}"},
        "series": [
            {
                "type": "treemap",
                "roam": False,
                "nodeClick": False,
                "breadcrumb": {"show": False},
                "data": data,
                "label": {"show": True, "fontSize": TEXT_SIZE, "color": TEXT_COLOR},
                "upperLabel": {"show": False},
                "itemStyle": {"borderColor": "#ffffff", "borderWidth": 2, "gapWidth": 2},
                "levels": [
                    {
                        "color": [PALETTE["blue"], PALETTE["sky"], PALETTE["teal"], PALETTE["green"], PALETTE["yellow"]],
                        "colorSaturation": [0.35, 0.65],
                    }
                ],
            }
        ],
        "animationDuration": 900,
    }
    st_echarts(options=_decorate_options(options, key, "Selected Feature Map"), height="380px", key=key)


def model_comparison_radar(rows: pd.DataFrame, key: str) -> None:
    metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "recall_impaired"]
    labels = ["Accuracy", "Precision", "Recall", "Macro F1", "Impaired Recall"]
    options = {
        "tooltip": {},
        "legend": {"bottom": 0, "textStyle": {"color": TEXT_COLOR, "fontSize": TEXT_SIZE}},
        "radar": {
            "radius": "66%",
            "indicator": [{"name": label, "max": 1} for label in labels],
            "axisName": {"color": TEXT_COLOR, "fontSize": TEXT_SIZE},
        },
        "series": [
            {
                "type": "radar",
                "data": [
                    {"name": row["model"], "value": [round(float(row[col]), 3) for col in metric_cols]}
                    for _, row in rows.iterrows()
                ],
                "areaStyle": {"opacity": 0.14},
                "symbolSize": 6,
            }
        ],
        "color": [PALETTE["blue"], PALETTE["teal"], PALETTE["coral"]],
        "animationDuration": 1000,
    }
    st_echarts(options=_decorate_options(options, key, "Radar Chart"), height="390px", key=key)


def liquid_style_gauge(value: float, label: str, key: str, color: str | None = None) -> None:
    percent = max(0.0, min(1.0, float(value)))
    chart_color = color or PALETTE["blue"]
    options = {
        "series": [
            {
                "type": "gauge",
                "startAngle": 210,
                "endAngle": -30,
                "min": 0,
                "max": 1,
                "radius": "92%",
                "progress": {
                    "show": True,
                    "roundCap": True,
                    "width": 16,
                    "itemStyle": {"color": chart_color},
                },
                "axisLine": {"lineStyle": {"width": 16, "color": [[1, "#e8eef7"]]}},
                "axisTick": {"show": False},
                "splitLine": {"show": False},
                "axisLabel": {"show": False},
                "pointer": {"show": False},
                "anchor": {"show": False},
                "detail": {
                    "valueAnimation": True,
                    "formatter": "{value}%",
                    "fontSize": 24,
                    "fontWeight": "bold",
                    "color": PALETTE["ink"],
                    "offsetCenter": [0, "-2%"],
                },
                "title": {"offsetCenter": [0, "34%"], "fontSize": TEXT_SIZE, "color": TEXT_COLOR},
                "data": [{"value": round(percent * 100, 1), "name": label}],
            }
        ],
        "animationDuration": 1200,
        "animationEasing": "elasticOut",
    }
    st_echarts(options=_decorate_options(options, key, "Score Gauge"), height="260px", key=key)


def polar_metric_bars(metrics: dict[str, float], key: str) -> None:
    labels = ["Accuracy", "Precision", "Recall", "Macro F1", "Impaired Recall"]
    values = [
        float(metrics["accuracy"]),
        float(metrics["precision_macro"]),
        float(metrics["recall_macro"]),
        float(metrics["f1_macro"]),
        float(metrics["recall_impaired"]),
    ]
    options = {
        "tooltip": {"trigger": "item", "formatter": "{b}: {c}"},
        "polar": {"radius": [22, "78%"]},
        "angleAxis": {"max": 1, "startAngle": 75, "axisLabel": {"show": False}},
        "radiusAxis": {
            "type": "category",
            "data": labels,
            "axisLabel": {"color": TEXT_COLOR, "fontSize": TEXT_SIZE},
        },
        "series": [
            {
                "type": "bar",
                "data": [round(v, 3) for v in values],
                "coordinateSystem": "polar",
                "roundCap": True,
                "label": {"show": True, "position": "middle", "formatter": "{c}"},
                "itemStyle": {
                    "borderRadius": 8,
                    "color": {
                        "type": "linear",
                        "x": 0,
                        "y": 0,
                        "x2": 1,
                        "y2": 0,
                        "colorStops": [
                            {"offset": 0, "color": PALETTE["sky"]},
                            {"offset": 1, "color": PALETTE["blue"]},
                        ],
                    },
                },
            }
        ],
        "animationDuration": 1100,
        "animationEasing": "cubicOut",
    }
    st_echarts(options=_decorate_options(options, key, "Radar Chart"), height="390px", key=key)


def confusion_sankey(matrix: list[list[int]], key: str) -> None:
    nodes = [
        {"name": "Actual: No Impairment"},
        {"name": "Actual: Impaired"},
        {"name": "Predicted: No Impairment"},
        {"name": "Predicted: Impaired"},
    ]
    links = [
        {"source": "Actual: No Impairment", "target": "Predicted: No Impairment", "value": matrix[0][0]},
        {"source": "Actual: No Impairment", "target": "Predicted: Impaired", "value": matrix[0][1]},
        {"source": "Actual: Impaired", "target": "Predicted: No Impairment", "value": matrix[1][0]},
        {"source": "Actual: Impaired", "target": "Predicted: Impaired", "value": matrix[1][1]},
    ]
    options = {
        "tooltip": {"trigger": "item", "triggerOn": "mousemove"},
        "series": [
            {
                "type": "sankey",
                "data": nodes,
                "links": links,
                "nodeAlign": "justify",
                "emphasis": {"focus": "adjacency"},
                "lineStyle": {"color": "gradient", "curveness": 0.5, "opacity": 0.38},
                "itemStyle": {"borderRadius": 4, "borderColor": "#fff", "borderWidth": 1},
                "label": {"color": TEXT_COLOR, "fontSize": TEXT_SIZE},
                "levels": [
                    {"depth": 0, "itemStyle": {"color": PALETTE["blue"]}},
                    {"depth": 1, "itemStyle": {"color": PALETTE["teal"]}},
                ],
            }
        ],
        "animationDuration": 1000,
    }
    st_echarts(options=_decorate_options(options, key, "Metric Chart"), height="360px", key=key)


def stacked_outcome_bar(matrix: list[list[int]], key: str) -> None:
    labels = ["Actual No Impairment", "Actual Impaired"]
    options = {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"top": 0, "textStyle": {"color": TEXT_COLOR, "fontSize": TEXT_SIZE}},
        "grid": {"left": 138, "right": 36, "top": 52, "bottom": 30, "containLabel": True},
        "xAxis": {"type": "value", **_axis_common()},
        "yAxis": {"type": "category", "data": labels, "axisLabel": {"color": PALETTE["ink"]}},
        "series": [
            {
                "name": "Predicted No Impairment",
                "type": "bar",
                "stack": "total",
                "label": {"show": True},
                "emphasis": {"focus": "series"},
                "itemStyle": {"color": PALETTE["blue"], "borderRadius": [0, 0, 0, 0]},
                "data": [matrix[0][0], matrix[1][0]],
            },
            {
                "name": "Predicted Impaired",
                "type": "bar",
                "stack": "total",
                "label": {"show": True},
                "emphasis": {"focus": "series"},
                "itemStyle": {"color": PALETTE["coral"], "borderRadius": [0, 6, 6, 0]},
                "data": [matrix[0][1], matrix[1][1]],
            },
        ],
        "animationDuration": 1000,
        "animationEasing": "cubicOut",
    }
    st_echarts(options=_decorate_options(options, key, "Outcome Stack"), height="300px", key=key)


def feature_domain_sunburst(features: Iterable[str], key: str) -> None:
    domains: dict[str, list[str]] = {
        "Glycemic": [],
        "Blood Pressure": [],
        "Perfusion": [],
        "White Matter": [],
        "Body / Other": [],
    }
    for feature in features:
        if any(token in feature for token in ["glucose", "hba1c", "diabetes"]):
            domains["Glycemic"].append(feature)
        elif "sbp" in feature or "dbp" in feature:
            domains["Blood Pressure"].append(feature)
        elif "perfusion" in feature or "vasoreactivity" in feature:
            domains["Perfusion"].append(feature)
        elif "wmh" in feature or "wm_" in feature:
            domains["White Matter"].append(feature)
        else:
            domains["Body / Other"].append(feature)

    data = {
        "name": "Feature Domains",
        "children": [
            {
                "name": f"{domain} ({len(items)})",
                "children": [{"name": item, "value": 1} for item in items],
            }
            for domain, items in domains.items()
            if items
        ],
    }
    options = {
        "title": {
            "text": "Feature Domains",
            "left": 12,
            "top": 8,
            "textStyle": {"color": TEXT_COLOR, "fontSize": TEXT_SIZE, "fontWeight": 700},
            "subtextStyle": {"color": TEXT_COLOR, "fontSize": TEXT_SIZE},
        },
        "tooltip": {"trigger": "item", "triggerOn": "mousemove"},
        "series": [
            {
                "type": "tree",
                "data": [data],
                "top": "4%",
                "left": "8%",
                "bottom": "4%",
                "right": "26%",
                "orient": "LR",
                "symbol": "circle",
                "symbolSize": 9,
                "edgeShape": "polyline",
                "edgeForkPosition": "48%",
                "initialTreeDepth": -1,
                "lineStyle": {"color": PALETTE["grid"], "width": 2},
                "itemStyle": {"color": PALETTE["blue"], "borderColor": "#ffffff", "borderWidth": 2},
                "label": {
                    "position": "left",
                    "verticalAlign": "middle",
                    "align": "right",
                    "color": "#000000",
                    "fontSize": 13,
                    "fontWeight": 600,
                },
                "leaves": {
                    "label": {
                        "position": "right",
                        "verticalAlign": "middle",
                        "align": "left",
                        "color": "#000000",
                        "fontSize": 13,
                        "fontWeight": 500,
                    }
                },
                "emphasis": {"focus": "descendant"},
                "expandAndCollapse": True,
                "animationDuration": 550,
                "animationDurationUpdate": 750,
            }
        ],
        "color": [PALETTE["blue"], PALETTE["teal"], PALETTE["green"], PALETTE["yellow"], PALETTE["coral"]],
        "textStyle": {"color": "#000000", "fontSize": 13},
    }
    st_echarts(options=_decorate_options(options, key, "Feature Domains"), height="520px", key=key)


def mixed_metric_combo(rows: pd.DataFrame, key: str) -> None:
    metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "recall_impaired"]
    labels = ["Accuracy", "Precision", "Recall", "Macro F1", "Impaired Recall"]
    model_rows = rows
    series = []
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["coral"]]
    for idx, (_, row) in enumerate(model_rows.iterrows()):
        series.append(
            {
                "name": row["model"],
                "type": "bar" if idx == 0 else "line",
                "smooth": True,
                "barMaxWidth": 22,
                "itemStyle": {"borderRadius": [6, 6, 0, 0], "color": colors[idx % len(colors)]},
                "lineStyle": {"width": 3, "color": colors[idx % len(colors)]},
                "data": [round(float(row[col]), 3) for col in metric_cols],
            }
        )
    options = {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross", "crossStyle": {"color": "#999"}}},
        "legend": {"top": 0, "textStyle": {"color": TEXT_COLOR, "fontSize": TEXT_SIZE}},
        "grid": {"left": 46, "right": 30, "top": 54, "bottom": 48, "containLabel": True},
        "xAxis": [{"type": "category", "data": labels, "axisPointer": {"type": "shadow"}, **_axis_common()}],
        "yAxis": [{"type": "value", "min": 0, "max": 1, **_axis_common()}],
        "series": series,
        "animationDuration": 1000,
    }
    st_echarts(options=_decorate_options(options, key, "Metric Chart"), height="360px", key=key)
