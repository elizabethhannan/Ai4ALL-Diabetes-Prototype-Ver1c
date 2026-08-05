## Streamlit dashboard for the Responsible AI / bias-report portfolio.
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


## ---- project paths and app import setup ----
REPORTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPORTS_DIR.parents[1]
APP_DIR = PROJECT_ROOT / "app"
DATA_PATH = PROJECT_ROOT / "data" / "GE79_MASTER_DATASET_V1.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

## Shared dashboard helpers are imported after APP_DIR is added to sys.path.
from echarts_components import (  # noqa: E402
    bottom_export_images,
    inject_theme,
    project_footer,
    project_page_header,
)

## ---- report inventory shown in the dashboard selector ----
REPORTS = [
    {
        "title": "NIST AI RMF Report",
        "file": "NIST_AI_RMF_Report.md",
        "summary": "Responsible AI governance report organized around Govern, Map, Measure, and Manage.",
    },
    {
        "title": "TRIPOD-AI Checklist",
        "file": "TRIPOD_AI_Checklist.md",
        "summary": "Reporting-completeness checklist for the GE-79 machine learning project.",
    },
    {
        "title": "PROBAST-AI Report",
        "file": "PROBAST_AI_Report.md",
        "summary": "Risk-of-bias and applicability review for the clinical prediction workflow.",
    },
    {
        "title": "GE-79 Model Card",
        "file": "GE79_MODEL_CARD.md",
        "summary": "Model documentation covering intended use, metrics, limitations, and Responsible AI constraints.",
    },
    {
        "title": "GE-79 Dataset Card",
        "file": "GE79_DATASET_CARD.md",
        "summary": "Dataset documentation covering source, variables, cleaning, missing data, bias, licensing, and use limits.",
    },
    {
        "title": "AI4ALL Rubric Alignment",
        "file": "AI4ALL_RUBRIC_ALIGNMENT.md",
        "summary": "Direct checklist mapping project artifacts to AI4ALL final presentation, GitHub page, and repository expectations.",
    },
    {
        "title": "SHAP Explainability Report",
        "file": "SHAP_Report.md",
        "summary": "Random Forest explainability report using global and individual SHAP outputs.",
    },
    {
        "title": "Bias Mitigation Notes",
        "file": "model1_model2_model3_bias_mitigation.txt",
        "summary": "Bias and mitigation notes for Models 1-3.",
    },
]

REPORT_LOOKUP = {report["title"]: report for report in REPORTS}

## ---- fixed summary tables used by the Responsible AI visuals ----
VERIFYWISE_RESULTS = pd.DataFrame(
    [
        {
            "Protected Attribute": "Race",
            "Target": "Cognitive Status",
            "Overall Bias": "Low",
            "Statistical Parity Difference": 0.000,
            "Disparate Impact Ratio": 1.000,
            "Demographic Parity Ratio": 1.000,
        },
        {
            "Protected Attribute": "Study Group",
            "Target": "Cognitive Status",
            "Overall Bias": "Low",
            "Statistical Parity Difference": 0.000,
            "Disparate Impact Ratio": 1.000,
            "Demographic Parity Ratio": 1.000,
        },
        {
            "Protected Attribute": "Diabetes Status",
            "Target": "Cognitive Status",
            "Overall Bias": "Low",
            "Statistical Parity Difference": 0.000,
            "Disparate Impact Ratio": 1.000,
            "Demographic Parity Ratio": 1.000,
        },
        {
            "Protected Attribute": "Hypertension Status",
            "Target": "Cognitive Status",
            "Overall Bias": "Low",
            "Statistical Parity Difference": 0.000,
            "Disparate Impact Ratio": 1.000,
            "Demographic Parity Ratio": 1.000,
        },
        {
            "Protected Attribute": "Insulin Use",
            "Target": "Cognitive Status",
            "Overall Bias": "Low",
            "Statistical Parity Difference": 0.000,
            "Disparate Impact Ratio": 1.000,
            "Demographic Parity Ratio": 1.000,
        },
    ]
)

MODEL_AUC_RESULTS = pd.DataFrame(
    [
        {"Model": "Model 1", "ROC-AUC": 0.534, "PR-AUC": 0.369},
        {"Model": "Model 2", "ROC-AUC": 0.639, "PR-AUC": 0.370},
        {"Model": "Model 3", "ROC-AUC": 0.648, "PR-AUC": 0.441},
    ]
)

PROBAST_RISK_RESULTS = pd.DataFrame(
    [
        {"Domain": "Participants", "Risk Score": 3, "Risk of Bias": "High"},
        {"Domain": "Predictors", "Risk Score": 2, "Risk of Bias": "Moderate"},
        {"Domain": "Outcome", "Risk Score": 2, "Risk of Bias": "Moderate"},
        {"Domain": "Analysis", "Risk Score": 2, "Risk of Bias": "Moderate"},
    ]
)

FAIRLEARN_READINESS = pd.DataFrame(
    [
        {"Item": "Protected attributes identified", "Status": "Available", "Ready": 1},
        {"Item": "Ground-truth outcome label", "Status": "Available", "Ready": 1},
        {"Item": "Model predictions by participant", "Status": "Not saved", "Ready": 0},
        {"Item": "Subgroup performance metrics", "Status": "Future work", "Ready": 0},
        {"Item": "Fairlearn dashboard/MetricFrame", "Status": "Future work", "Ready": 0},
    ]
)

OECD_PRINCIPLES = pd.DataFrame(
    [
        {
            "Principle": "Inclusive Growth, Sustainable Development & Well-being",
            "What it means": "AI should benefit people and society.",
            "How this project addresses it": "Investigates biomarkers associated with mild cognitive impairment in older adults with Type 2 Diabetes to support future research into earlier identification of cognitive decline.",
            "Covered": 1,
        },
        {
            "Principle": "Human-Centered Values & Fairness",
            "What it means": "Respect fairness and avoid harmful bias.",
            "How this project addresses it": "Documents geographic, selection, education, racial representation, survivorship, and label bias; evaluates fairness using VerifyWise and interprets results conservatively.",
            "Covered": 1,
        },
        {
            "Principle": "Transparency & Explainability",
            "What it means": "AI decisions should be understandable.",
            "How this project addresses it": "Documents preprocessing, feature selection, model comparisons, confusion matrices, ROC-AUC, PR-AUC, and feature importance; Decision Tree and Random Forest feature importance support interpretability.",
            "Covered": 1,
        },
        {
            "Principle": "Robustness, Security & Safety",
            "What it means": "AI should be reliable and appropriately tested.",
            "How this project addresses it": "Uses documented preprocessing, leakage prevention, 5-fold stratified cross-validation, and multiple evaluation metrics rather than relying on accuracy alone.",
            "Covered": 1,
        },
        {
            "Principle": "Accountability",
            "What it means": "Developers should document limitations and be accountable.",
            "How this project addresses it": "Includes documented preprocessing, version-controlled code, dataset limitations, bias assessment, and states that results are research findings rather than clinical decision support.",
            "Covered": 1,
        },
    ]
)

## ---- narrative sections: each item powers one report panel in the app ----
REPORT_SECTIONS = [
    {
        "title": "VerifyWise Fairness Assessment",
        "subtitle": "Independent fairness-metric screen across protected attributes",
        "file": "model1_model2_model3_bias_mitigation.txt",
        "accent": "#2563eb",
        "technical": [
            "VerifyWise evaluated the cleaned GE-79 dataset across race, study group, diabetes status, hypertension status, and insulin use.",
            "The target evaluated was `cognitive_status_label`, which is a ground-truth clinical label rather than model-generated predictions.",
            "Statistical parity difference was 0.000, disparate impact ratio was 1.000, and demographic parity ratio was 1.000 for the tested comparisons.",
            "The result supports a narrow fairness-screening statement only; it does not prove that the dataset or trained models are unbiased.",
        ],
        "takeaways": [
            "The fairness tool did not flag measurable parity differences for the attributes tested.",
            "This is not the same as proving the project is bias-free.",
            "The dataset still has important limits, including small sample size and limited representation.",
            "Model fairness should also be checked using model predictions by subgroup when sample sizes allow.",
        ],
    },
    {
        "title": "Fairlearn Fairness Extension",
        "subtitle": "Recommended next-step subgroup fairness analysis for model predictions",
        "file": "",
        "accent": "#0891b2",
        "technical": [
            "Fairlearn would extend the current bias review by evaluating fairness metrics on model-generated predictions, not only the ground-truth target label.",
            "Relevant sensitive features for future Fairlearn analysis include race, study group, diabetes status, hypertension status, and insulin use.",
            "The current repository does not save per-participant cross-validated predictions for Models 1-3, so Fairlearn MetricFrame results should not be invented in this report.",
            "A defensible Fairlearn update would save out-of-fold predictions and then compute subgroup accuracy, recall, selection rate, false positive rate, and false negative rate.",
        ],
        "takeaways": [
            "Fairlearn is the next tool to use when the project is ready to test model fairness directly.",
            "VerifyWise checked the dataset label; Fairlearn would check the model predictions.",
            "We should not claim Fairlearn results until prediction-level files are created.",
            "Adding Fairlearn later would make the Responsible AI section stronger and more review-ready.",
        ],
    },
    {
        "title": "OECD AI Principles",
        "subtitle": "Responsible AI principles mapped to the GE-79 project",
        "file": "",
        "accent": "#0f766e",
        "technical": [
            "The project is mapped to the five OECD AI Principles: inclusive growth and well-being, human-centered values and fairness, transparency and explainability, robustness and safety, and accountability.",
            "The mapping documents how the GE-79 workflow supports research benefit, conservative fairness interpretation, reproducible model evaluation, and limitation reporting.",
            "The OECD section complements NIST AI RMF by translating governance into a principle-based checklist for responsible project communication.",
            "The project remains research-only; OECD alignment documents responsible practice but does not imply clinical validation or deployment readiness.",
        ],
        "takeaways": [
            "The OECD principles explain what responsible AI should look like in plain terms.",
            "This project supports social benefit by studying cognitive decline risk in an older Type 2 Diabetes cohort.",
            "The project is transparent about methods, model limits, and bias risks.",
            "Being accountable means clearly saying what the model can and cannot prove.",
        ],
    },
    {
        "title": "NIST AI RMF Report",
        "subtitle": "Govern, Map, Measure, and Manage review",
        "file": "NIST_AI_RMF_Report.md",
        "accent": "#2a9d8f",
        "technical": [
            "The NIST AI RMF report organizes the project around governance, risk mapping, measurement, and risk management.",
            "The report documents leakage prevention, model evaluation metrics, fairness review, SHAP interpretability, and conservative interpretation.",
            "ROC-AUC and PR-AUC are included because accuracy alone is insufficient for an imbalanced healthcare classification task.",
            "The report keeps the project framed as research-only and not clinically deployment-ready.",
        ],
        "takeaways": [
            "This report explains how the project identifies and manages AI risk.",
            "It shows that the project uses more than accuracy to judge model performance.",
            "It documents why the results should be interpreted carefully.",
            "It supports responsible presentation for the AI4ALL Ignite Symposium.",
        ],
    },
    {
        "title": "TRIPOD-AI Checklist",
        "subtitle": "Clinical ML reporting completeness check",
        "file": "TRIPOD_AI_Checklist.md",
        "accent": "#f2994a",
        "technical": [
            "The TRIPOD-AI checklist reports 15 complete sections, 2 partial sections, and 0 missing sections.",
            "The weighted completeness score is 16.0 out of 17.0, corresponding to 94.1%.",
            "Partial areas are documented instead of inferred, preserving defensibility and transparency.",
            "The checklist confirms that model metrics, validation strategy, Responsible AI review, and reproducibility artifacts are documented.",
        ],
        "takeaways": [
            "Most reporting requirements are complete.",
            "The remaining gaps are clearly identified instead of hidden.",
            "This helps reviewers understand what was done and what still needs improvement.",
            "The report makes the project look more like a professional clinical ML study.",
        ],
    },
    {
        "title": "PROBAST-AI Report",
        "subtitle": "Risk-of-bias and applicability assessment",
        "file": "PROBAST_AI_Report.md",
        "accent": "#dc2626",
        "technical": [
            "The overall PROBAST-AI risk-of-bias rating is High.",
            "Participant risk is High because the cleaned cohort is small, single-site, and not externally validated.",
            "Predictor, outcome, and analysis domains are rated Moderate, with limitations documented rather than overstated.",
            "The report separates research value from clinical readiness and recommends future external validation.",
        ],
        "takeaways": [
            "The project is useful for research, but it is not ready for clinical use.",
            "The biggest concern is the small, limited dataset.",
            "The report is honest about risk instead of overselling the model.",
            "Future testing on outside data would make the evidence stronger.",
        ],
    },
    {
        "title": "GE-79 Model Card",
        "subtitle": "Model documentation, metrics, limitations, and responsible use",
        "file": "GE79_MODEL_CARD.md",
        "accent": "#7c3aed",
        "technical": [
            "The model card documents Models 0-3, the finalized biomarker feature set, intended use, metrics, ROC-AUC, PR-AUC, limitations, and Responsible AI constraints.",
            "Model 3 has the highest accuracy and ROC-AUC/PR-AUC, but its impaired recall remains low at 0.25.",
            "Model 2 has the strongest impaired recall among the three supervised models at 0.55.",
            "The card recommends presenting the models as research comparisons rather than choosing a deployment-ready clinical model.",
        ],
        "takeaways": [
            "The model card is the main plain-language documentation for the ML system.",
            "It explains what each model can and cannot be used for.",
            "A higher accuracy score does not automatically mean the safest model.",
            "The safest interpretation is that these models need more validation before real-world use.",
        ],
    },
    {
        "title": "GE-79 Dataset Card",
        "subtitle": "Dataset documentation for source, variables, cleaning, missingness, bias, and use limits",
        "file": "GE79_DATASET_CARD.md",
        "accent": "#4f46e5",
        "technical": [
            "The dataset card documents the cleaned `GE79_MASTER_DATASET_V1.csv` file with 75 rows and 46 columns.",
            "It identifies GE-79 / PhysioNet CDED 1.0.1 as the primary modeling source and GE-75 as a supporting reference dataset excluded from current modeling.",
            "It summarizes participant population, target definition, variable domains, preprocessing, missing data, known biases, licensing caveats, and appropriate versus inappropriate use.",
            "The card explicitly states that the dataset supports research and education, not clinical diagnosis, triage, treatment, or deployment claims.",
        ],
        "takeaways": [
            "The dataset card explains what data the project used and where it came from.",
            "It makes the dataset limits visible instead of hiding them.",
            "It helps reviewers understand what the model was allowed to learn from.",
            "It clearly says the data is for research, not medical decision-making.",
        ],
    },
    {
        "title": "AI4ALL Rubric Alignment",
        "subtitle": "Direct evidence map from project assets to AI4ALL grading criteria",
        "file": "AI4ALL_RUBRIC_ALIGNMENT.md",
        "accent": "#111827",
        "technical": [
            "The alignment report maps project evidence to the AI4ALL final presentation, GitHub page, and repository rubric categories.",
            "The report identifies completed evidence for project description, visualizations, algorithm explanation, essential question, next steps, citations, and documentation.",
            "The README now exceeds the citation threshold with 11 citations and includes direct data-source links.",
            "Remaining recommendations focus on final presentation logistics rather than missing technical artifacts.",
        ],
        "takeaways": [
            "This section shows graders exactly where to find the required project evidence.",
            "The project now has a clear checklist for presentation, GitHub page, and repository review.",
            "Most remaining work is polishing delivery and confirming team details.",
            "The report helps make the project easier to grade at the highest rubric level.",
        ],
    },
]


def read_report(filename: str) -> str:
    path = REPORTS_DIR / filename
    if not path.exists():
        return f"Report file not found: `{filename}`"
    return path.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def safe_bar_chart(data, height: int = 330) -> None:
    ## Keep native Streamlit charts inside the same bordered, clipped frame as ECharts.
    st.bar_chart(data, use_container_width=True, height=height)


def inject_report_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7fafc;
            color: #111827;
        }
        h1, h2, h3, h4, h5, h6, p, li, span, div {
            color: #111827;
        }
        .report-header {
            background: #ffffff;
            border: 1px solid #d8e2ea;
            border-radius: 8px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
        }
        .report-card {
            background: #ffffff;
            border: 1px solid #d8e2ea;
            border-left: 6px solid #2563eb;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 0.5rem 0 1rem 0;
        }
        .report-card strong {
            color: #111827;
            font-size: 1rem;
        }
        .report-card p {
            color: #374151;
            font-size: 0.95rem;
            margin-bottom: 0;
        }
        .report-body {
            background: #ffffff;
            border: 1px solid #d8e2ea;
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
        }
        .report-body p,
        .report-body li {
            font-size: 16px;
            line-height: 1.55;
        }
        .report-body h1 {
            font-size: 1.75rem;
        }
        .report-body h2 {
            font-size: 1.35rem;
            margin-top: 1.25rem;
        }
        .report-body h3 {
            font-size: 1.1rem;
        }
        .audit-note {
            background: #ffffff;
            border: 1px solid #d8e2ea;
            border-left: 6px solid #2563eb;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 0.75rem 0 1rem 0;
            color: #111827;
            font-size: 15px;
            line-height: 1.5;
        }
        .executive-summary {
            background: #ffffff;
            border: 1px solid #d8e2ea;
            border-radius: 8px;
            padding: 1.15rem 1.25rem;
            margin: 0.75rem 0 1rem 0;
        }
        .executive-summary h3 {
            margin: 0 0 0.4rem 0;
            color: #111827;
            font-size: 1.15rem;
        }
        .executive-summary p,
        .executive-summary li {
            color: #111827;
            font-size: 15px;
            line-height: 1.5;
        }
        .model-results-summary {
            background: #ffffff;
            border: 1px solid #d8e2ea;
            border-left: 7px solid #4f46e5;
            border-radius: 8px;
            padding: 1rem 1.15rem;
            margin: 0.75rem 0 1.1rem 0;
        }
        .model-results-summary h3 {
            margin: 0 0 0.35rem 0;
            color: #111827;
            font-size: 1.2rem;
        }
        .model-results-summary p {
            color: #111827;
            font-size: 15px;
            line-height: 1.5;
            margin-bottom: 0;
        }
        .section-box {
            background: #ffffff;
            border: 1px solid #d8e2ea;
            border-radius: 8px;
            padding: 1.1rem 1.2rem;
            margin: 1rem 0 1.25rem 0;
        }
        .section-title {
            display: block;
            color: #111827;
            font-size: 1.35rem;
            font-weight: 850;
            margin-bottom: 0.1rem;
        }
        .section-subtitle {
            color: #374151;
            font-size: 15px;
            line-height: 1.45;
        }
        .bullet-panel {
            background: #f8fafc;
            border: 1px solid #dbe5f1;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-top: 0.75rem;
        }
        .bullet-panel strong {
            color: #111827;
            font-size: 16px;
        }
        .bullet-panel li {
            color: #111827;
            font-size: 15px;
            line-height: 1.45;
            margin-bottom: 0.25rem;
        }
        div[data-testid="stVegaLiteChart"],
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
        div[data-testid="stVegaLiteChart"] svg,
        div[data-testid="stVegaLiteChart"] canvas,
        div[data-testid="stImage"] img {
            max-width: 100% !important;
            box-sizing: border-box;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def report_card(title: str, summary: str, filename: str) -> None:
    st.markdown(
        f"""
        <div class="report-card">
            <strong>{title}</strong>
            <p>{summary}<br><code>{filename}</code></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_report(report: dict[str, str]) -> None:
    report_card(report["title"], report["summary"], report["file"])
    content = read_report(report["file"])
    st.markdown('<div class="report-body">', unsafe_allow_html=True)
    st.markdown(content)
    st.markdown("</div>", unsafe_allow_html=True)


def render_model_results_summary() -> None:
    image_name = "model_results_comparison_summary.png"
    summary_image_candidates = [
        OUTPUTS_DIR / image_name,
        PROJECT_ROOT / "outputs" / image_name,
        Path.cwd() / "outputs" / image_name,
        REPORTS_DIR / image_name,
    ]
    summary_image = next(
        (path for path in summary_image_candidates if path.exists()),
        summary_image_candidates[0],
    )
    st.markdown(
        """
        <div class="model-results-summary">
            <h3>Model Results Comparison Summary</h3>
            <p>
                This comparison highlights why the GE-79 project evaluates multiple
                metrics instead of relying on accuracy alone. Random Forest shows the
                strongest overall predictive performance across accuracy, Macro F1,
                ROC-AUC, and PR-AUC, while Decision Tree detects the largest share of
                Mild Cognitive Impairment cases using impaired recall.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if summary_image.exists():
        st.image(str(summary_image), use_container_width=True)
    else:
        st.warning("Missing summary image: outputs/model_results_comparison_summary.png")


def bullet_panel(title: str, items: list[str]) -> None:
    bullet_items = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(
        f"""
        <div class="bullet-panel">
            <strong>{title}</strong>
            <ul>{bullet_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_executive_summary() -> None:
    st.markdown(
        """
        <div class="executive-summary">
            <h3>Initial Summary</h3>
            <p>
                This Bias & Responsible AI dashboard separates the GE-79 project reports into
                distinct review sections instead of displaying them as one continuous document.
                The sections summarize VerifyWise fairness screening, the planned Fairlearn
                fairness extension, OECD AI Principles alignment, NIST AI RMF governance,
                TRIPOD-AI reporting completeness, PROBAST-AI risk of bias, the GE-79 model
                card, the GE-79 dataset card, and AI4ALL rubric alignment. The purpose is
                documentation and review only: no model training, preprocessing, feature
                selection, or evaluation outputs are changed.
            </p>
            <ul>
                <li><strong>Dataset:</strong> GE-79 cleaned modeling dataset with 75 rows and 46 columns.</li>
                <li><strong>Target:</strong> cognitive-status label used for binary classification.</li>
                <li><strong>Fairness note:</strong> VerifyWise screened the dataset label; Fairlearn is documented as the next step for model-prediction fairness.</li>
                <li><strong>Clinical note:</strong> reports support research interpretation, not clinical deployment.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Report Sections", "9")
    metric_cols[1].metric("Dataset Rows", "75")
    metric_cols[2].metric("Dataset Columns", "46")
    metric_cols[3].metric("TRIPOD Complete", "94.1%")
    metric_cols[4].metric("PROBAST Overall", "High Risk")


def render_section_visual(title: str) -> None:
    if title == "VerifyWise Fairness Assessment":
        st.dataframe(VERIFYWISE_RESULTS, use_container_width=True, hide_index=True)
        parity_chart = VERIFYWISE_RESULTS.set_index("Protected Attribute")[
            ["Disparate Impact Ratio", "Demographic Parity Ratio"]
        ]
        safe_bar_chart(parity_chart)
    elif title == "Fairlearn Fairness Extension":
        st.metric("Current Fairlearn Status", "Planned")
        st.dataframe(FAIRLEARN_READINESS, use_container_width=True, hide_index=True)
        safe_bar_chart(FAIRLEARN_READINESS.set_index("Item")["Ready"])
    elif title == "OECD AI Principles":
        st.metric("OECD Principles Covered", "5 / 5")
        st.dataframe(
            OECD_PRINCIPLES[
                ["Principle", "What it means", "How this project addresses it"]
            ],
            use_container_width=True,
            hide_index=True,
        )
        safe_bar_chart(OECD_PRINCIPLES.set_index("Principle")["Covered"])
    elif title == "NIST AI RMF Report":
        nist = pd.DataFrame(
            {
                "Function": ["Govern", "Map", "Measure", "Manage"],
                "Documented": [1, 1, 1, 1],
            }
        )
        st.dataframe(nist, use_container_width=True, hide_index=True)
        safe_bar_chart(nist.set_index("Function")["Documented"])
    elif title == "TRIPOD-AI Checklist":
        tripod = pd.DataFrame(
            {
                "Status": ["Complete", "Partial", "Missing"],
                "Sections": [15, 2, 0],
            }
        )
        st.metric("Overall TRIPOD-AI Completeness", "94.1%")
        st.dataframe(tripod, use_container_width=True, hide_index=True)
        safe_bar_chart(tripod.set_index("Status")["Sections"])
    elif title == "PROBAST-AI Report":
        st.metric("Overall Risk of Bias", "High")
        st.dataframe(PROBAST_RISK_RESULTS, use_container_width=True, hide_index=True)
        safe_bar_chart(PROBAST_RISK_RESULTS.set_index("Domain")["Risk Score"])
    elif title == "GE-79 Model Card":
        st.dataframe(MODEL_AUC_RESULTS, use_container_width=True, hide_index=True)
        safe_bar_chart(MODEL_AUC_RESULTS.set_index("Model")[["ROC-AUC", "PR-AUC"]])
    elif title == "GE-79 Dataset Card":
        data = load_dataset()
        st.metric("Dataset Shape", f"{data.shape[0]} x {data.shape[1]}")
        dataset_summary = pd.DataFrame(
            [
                {"Item": "Rows", "Value": data.shape[0]},
                {"Item": "Columns", "Value": data.shape[1]},
                {"Item": "Normal / No Impairment", "Value": int((data["cognitive_status_label"] == "Normal").sum())},
                {"Item": "Mild Impairment", "Value": int((data["cognitive_status_label"] == "Mild Impairment").sum())},
            ]
        )
        st.dataframe(dataset_summary, use_container_width=True, hide_index=True)
        safe_bar_chart(
            data["cognitive_status_label"].value_counts().rename_axis("Class").reset_index(name="Count").set_index("Class")["Count"]
        )
    elif title == "AI4ALL Rubric Alignment":
        rubric = pd.DataFrame(
            [
                {"Criterion": "Project Description", "Status": 1},
                {"Criterion": "Visualizations", "Status": 1},
                {"Criterion": "Algorithm Explanation", "Status": 1},
                {"Criterion": "Essential Question", "Status": 1},
                {"Criterion": "Next Steps", "Status": 1},
                {"Criterion": "Citations", "Status": 1},
                {"Criterion": "GitHub Documentation", "Status": 1},
            ]
        )
        st.metric("Rubric Evidence Coverage", "7 / 7")
        st.dataframe(rubric, use_container_width=True, hide_index=True)
        safe_bar_chart(rubric.set_index("Criterion")["Status"])


def render_report_section(section: dict[str, object]) -> None:
    st.markdown(
        f"""
        <div class="section-box" style="border-left: 7px solid {section['accent']}">
            <span class="section-title">{section['title']}</span>
            <span class="section-subtitle">{section['subtitle']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    visual_col, text_col = st.columns([1, 1.05], gap="large")
    with visual_col:
        render_section_visual(str(section["title"]))
    with text_col:
        bullet_panel("Technical:", section["technical"])  # type: ignore[arg-type]
        bullet_panel("Takeaways:", section["takeaways"])  # type: ignore[arg-type]

    report_file = str(section["file"])
    if report_file and (REPORTS_DIR / report_file).exists():
        with st.expander(f"Open full source report: {report_file}", expanded=False):
            st.markdown(read_report(report_file))


def render_report_sections() -> None:
    render_executive_summary()
    for section in REPORT_SECTIONS:
        render_report_section(section)


def render_dataset_audit() -> None:
    st.markdown("## GE-79 Dataset Audit")
    if not DATA_PATH.exists():
        st.error(f"Dataset file not found: `{DATA_PATH.relative_to(PROJECT_ROOT)}`")
        return

    data = load_dataset()
    numeric_data = data.select_dtypes(include="number")
    categorical_data = data.select_dtypes(exclude="number")
    target_col = "cognitive_status_label"

    st.markdown(
        """
        <div class="audit-note">
            This audit reads <code>data/GE79_MASTER_DATASET_V1.csv</code> directly from
            the repository. It is read-only and does not modify preprocessing,
            feature selection, model training, or evaluation outputs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", f"{data.shape[0]}")
    metric_cols[1].metric("Columns", f"{data.shape[1]}")
    metric_cols[2].metric("Numeric Columns", f"{numeric_data.shape[1]}")
    metric_cols[3].metric("Categorical Columns", f"{categorical_data.shape[1]}")

    audit_tabs = st.tabs(
        [
            "Complete Table",
            "Feature Dictionary",
            "Missing Values",
            "Summary Statistics",
            "Correlations",
            "Class Distribution",
            "Categorical Counts",
            "Outlier Check",
        ]
    )

    with audit_tabs[0]:
        st.markdown("### Complete Data Table")
        st.dataframe(data, use_container_width=True, hide_index=True)

    with audit_tabs[1]:
        dictionary = pd.DataFrame(
            {
                "feature": data.columns,
                "dtype": [str(dtype) for dtype in data.dtypes],
                "missing_count": data.isna().sum().values,
                "missing_percent": (data.isna().mean().values * 100).round(2),
                "unique_values": data.nunique(dropna=True).values,
            }
        )
        st.markdown("### Feature Dictionary")
        st.dataframe(dictionary, use_container_width=True, hide_index=True)

    with audit_tabs[2]:
        missing = (
            pd.DataFrame(
                {
                    "feature": data.columns,
                    "missing_count": data.isna().sum().values,
                    "missing_percent": (data.isna().mean().values * 100).round(2),
                }
            )
            .sort_values(["missing_count", "feature"], ascending=[False, True])
            .reset_index(drop=True)
        )
        st.markdown("### Missing-Value Report")
        st.dataframe(missing, use_container_width=True, hide_index=True)

    with audit_tabs[3]:
        st.markdown("### Numeric Summary Statistics")
        if numeric_data.empty:
            st.info("No numeric columns found.")
        else:
            summary = numeric_data.describe().T.round(3)
            summary.insert(0, "missing_count", numeric_data.isna().sum())
            st.dataframe(summary, use_container_width=True)

    with audit_tabs[4]:
        st.markdown("### Correlation Matrix")
        if numeric_data.shape[1] < 2:
            st.info("At least two numeric columns are required for a correlation matrix.")
        else:
            corr = numeric_data.corr(numeric_only=True).round(3)
            st.dataframe(corr, use_container_width=True)

    with audit_tabs[5]:
        st.markdown("### Class Distribution")
        if target_col not in data.columns:
            st.warning(f"Target column not found: `{target_col}`")
        else:
            class_counts = (
                data[target_col]
                .value_counts(dropna=False)
                .rename_axis(target_col)
                .reset_index(name="count")
            )
            class_counts["percent"] = (
                class_counts["count"] / class_counts["count"].sum() * 100
            ).round(2)
            st.dataframe(class_counts, use_container_width=True, hide_index=True)
            safe_bar_chart(class_counts.set_index(target_col)["count"])

    with audit_tabs[6]:
        st.markdown("### Categorical Value Counts")
        if categorical_data.empty:
            st.info("No categorical columns found.")
        else:
            selected_column = st.selectbox(
                "Select categorical column",
                list(categorical_data.columns),
                key="categorical_count_column",
            )
            counts = (
                data[selected_column]
                .value_counts(dropna=False)
                .rename_axis(selected_column)
                .reset_index(name="count")
            )
            counts["percent"] = (counts["count"] / counts["count"].sum() * 100).round(2)
            st.dataframe(counts, use_container_width=True, hide_index=True)
            safe_bar_chart(counts.set_index(selected_column)["count"])

    with audit_tabs[7]:
        st.markdown("### IQR Outlier Report")
        if numeric_data.empty:
            st.info("No numeric columns found.")
        else:
            rows = []
            for column in numeric_data.columns:
                series = numeric_data[column].dropna()
                if series.empty:
                    continue
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = series[(series < lower) | (series > upper)]
                rows.append(
                    {
                        "feature": column,
                        "lower_bound": round(lower, 3),
                        "upper_bound": round(upper, 3),
                        "outlier_count": int(outliers.shape[0]),
                        "outlier_percent": round(outliers.shape[0] / series.shape[0] * 100, 2),
                    }
                )
            outlier_report = (
                pd.DataFrame(rows)
                .sort_values(["outlier_count", "feature"], ascending=[False, True])
                .reset_index(drop=True)
            )
            st.dataframe(outlier_report, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="GE-79 Bias Reports",
        page_icon="GE",
        layout="wide",
    )
    inject_theme()
    inject_report_theme()
    project_page_header("bias")

    st.markdown(
        """
        <div class="report-header">
            <strong>Purpose:</strong> This page displays the completed GE-79 documentation
            reports from the local <code>bias_reports</code> folder. It does not retrain
            models, change outputs, or generate new results.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_model_results_summary()

    available = [
        report for report in REPORTS if (REPORTS_DIR / report["file"]).exists()
    ]
    missing = [
        report["file"] for report in REPORTS if not (REPORTS_DIR / report["file"]).exists()
    ]

    metric_cols = st.columns(3)
    metric_cols[0].metric("Reports Found", len(available))
    metric_cols[1].metric("Folder", "bias_reports")
    metric_cols[2].metric("Mode", "Read-only")

    if missing:
        st.warning("Missing expected report files: " + ", ".join(missing))

    tabs = st.tabs(["Report Sections", "Dataset Audit", "Full Report Files"])
    with tabs[0]:
        render_report_sections()

    with tabs[1]:
        render_dataset_audit()

    with tabs[2]:
        for report in available:
            with st.expander(report["title"], expanded=False):
                render_report(report)

    bottom_export_images(
        "Overall Dashboard Visualization Export",
        [
            (
                "All Models Overall Dashboard",
                str(OUTPUTS_DIR / "all_models_Overall_dashboard _visualizations_EH_pale.png"),
            ),
        ],
    )

    project_footer()


if __name__ == "__main__":
    main()
