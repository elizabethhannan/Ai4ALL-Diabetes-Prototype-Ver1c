import streamlit as st

from echarts_components import (
    final_math_context_expander,
    inject_theme,
    project_footer,
    project_page_header,
    section,
)


st.set_page_config(
    page_title="GE-79 Mathematical Formula Reference",
    page_icon="GE",
    layout="wide",
)
inject_theme()
project_page_header("math")

st.markdown(
    """
    This page centralizes the formulas used to explain the GE-79 machine
    learning pipeline. Model pages focus on results and visuals; this reference
    page keeps the mathematics in one place for presentation and review.
    """
)


def formula_card(title: str, formulas: list[str], explanation: str, definitions: str | None = None) -> None:
    with st.container(border=True):
        st.subheader(title)
        for formula in formulas:
            st.latex(formula)
        st.markdown(explanation)
        if definitions:
            st.markdown(definitions)


section("Preprocessing and Shared Setup")

prep_left, prep_right = st.columns(2, gap="large")

with prep_left:
    formula_card(
        "Feature Standardization",
        [r"z=\frac{x-\mu}{\sigma}"],
        """
        Standardization puts biomarkers on comparable scales before modeling.
        This is especially useful when variables use different clinical units,
        such as glucose, blood pressure, and MRI-derived measurements.
        """,
        """
        - **x:** original biomarker value
        - **mu:** mean of the feature
        - **sigma:** standard deviation of the feature
        - **z:** standardized feature value
        """,
    )

with prep_right:
    formula_card(
        "Binary Target Encoding",
        [r"y \in \{0,1\}"],
        """
        The GE-79 target is encoded as a binary cognitive-status label. This
        setup supports Logistic Regression, Decision Tree, and Random Forest as
        supervised binary classification models.
        """,
        """
        - **0:** no impairment class
        - **1:** impaired cognitive class
        """,
    )

section("Model 0: Feature Selection")

model0_left, model0_right = st.columns(2, gap="large")

with model0_left:
    formula_card(
        "Random Forest Feature Importance",
        [r"FI_j = \sum \Delta I_j", r"\sum FI = 1"],
        """
        Random Forest ranks biomarkers by their contribution to reducing
        impurity across many decision trees. The normalized feature importance
        values show which biomarkers were most useful for the downstream GE-79
        models.
        """,
        """
        - **FI_j:** feature importance for biomarker j
        - **Delta I_j:** impurity decrease contributed by biomarker j
        - **Sum FI:** normalized feature-importance total
        """,
    )

with model0_right:
    formula_card(
        "Feature Importance Normalization",
        [r"FI_j^{norm}=\frac{FI_j}{\sum_{k=1}^{p}FI_k}"],
        """
        Normalization rescales feature importance values so the full ranking is
        easier to compare. This supports a stable locked feature set for Models
        1, 2, and 3.
        """,
        """
        - **p:** number of ranked candidate biomarkers
        - **FI_j norm:** normalized importance for biomarker j
        """,
    )

section("Model 1: Logistic Regression")

model1_left, model1_right = st.columns(2, gap="large")

with model1_left:
    formula_card(
        "Prediction Probability",
        [
            r"P(y=1)=\frac{1}{1+e^{-z}}",
            r"z=\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n",
        ],
        """
        Logistic Regression estimates the probability that a participant belongs
        to the impaired cognitive class. In this project, it is a binary
        classification model, not linear regression.
        """,
        """
        - **P(y=1):** probability of the impaired class
        - **z:** weighted biomarker score
        - **beta_0:** model intercept
        - **beta_i:** learned biomarker coefficient
        - **x_i:** selected biomarker value
        """,
    )

with model1_right:
    formula_card(
        "Logistic Regression Loss",
        [r"\text{Log Loss}=-(y\log(p)+(1-y)\log(1-p))"],
        """
        Log Loss penalizes incorrect probability estimates. It is useful for
        Logistic Regression because the model outputs a probability for the
        impaired class, not only a hard class label.
        """,
        """
        - **y:** true class label
        - **p:** predicted probability for the impaired class
        """,
    )

section("Model 2: Decision Tree")

model2_left, model2_right = st.columns(2, gap="large")

with model2_left:
    formula_card(
        "Gini Impurity and Information Gain",
        [
            r"\text{Gini}=1-\sum p^2",
            r"IG=H(\text{parent})-\sum_i \frac{N_i}{N}H(\text{child}_i)",
        ],
        """
        A Decision Tree selects biomarker thresholds that split participants
        into increasingly pure cognitive-status groups. Gini impurity and
        information gain describe how each split improves class separation.
        """,
        """
        - **Gini:** class-mixing score inside a node
        - **p:** class proportion in the node
        - **IG:** information gained from a candidate split
        - **N_i/N:** share of samples routed to child node i
        """,
    )

with model2_right:
    formula_card(
        "Entropy",
        [r"H=-\sum_i p_i\log_2(p_i)"],
        """
        Entropy measures how mixed the outcome classes are inside a node. Lower
        entropy means a node is more class-specific, which helps explain why a
        tree chooses one biomarker split over another.
        """,
        """
        - **H:** entropy
        - **p_i:** proportion of class i in the node
        """,
    )

section("Model 3: Random Forest")

with st.container(border=True):
    st.subheader("Mathematical Formula & Model Interpretation")
    st.markdown("#### Model Formula")
    st.markdown("**Random Forest Majority Vote**")
    st.latex(r"\hat{y}=\operatorname{mode}(T_1,T_2,\ldots,T_n)")

    st.markdown("#### Variable Definitions")
    st.markdown(
        """
        - **$\\hat{y}$:** Final predicted cognitive-status class.
        - **T_i:** Prediction from decision tree i in the forest.
        - **mode:** The class receiving the most tree votes.
        - **n:** Number of decision trees in the forest.
        """,
    )

    st.markdown("#### Plain-English Explanation")
    st.markdown(
        """
        Random Forest combines many decision trees and predicts using majority
        vote. Each tree votes for a cognitive-status class, and the forest
        returns the class selected most often.
        """,
    )

    with st.expander("Why This Formula Matters"):
        st.markdown(
            """
            This formula shows how Model 3 combines many individual tree
            predictions into one final classification. It makes the ensemble
            voting process easier to explain and compare with the single
            Decision Tree model.
            """
        )

section("Performance Metric Formulas")

metric_left, metric_right = st.columns(2, gap="large")

with metric_left:
    formula_card(
        "Accuracy",
        [r"\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}"],
        """
        Accuracy measures the overall share of correct predictions. It is useful
        for a quick performance summary, but GE-79 also requires class-sensitive
        metrics because the binary target is imbalanced.
        """,
    )

    formula_card(
        "Recall",
        [r"\text{Recall}=\frac{TP}{TP+FN}"],
        """
        Recall measures how many truly impaired participants the model catches.
        It is appropriate for GE-79 because missed impaired cases are a key
        safety and interpretation concern.
        """,
    )

    formula_card(
        "F1 Score",
        [r"F1=2\times\frac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}"],
        """
        F1 Score balances precision and recall in one metric. It is useful for
        GE-79 because it summarizes the tradeoff between false alarms and missed
        impaired cases.
        """,
    )

    formula_card(
        "ROC-AUC Concept",
        [r"AUC=\int_0^1 TPR(FPR)\,dFPR"],
        """
        ROC-AUC summarizes performance across classification thresholds. It is
        useful for comparing models beyond one fixed cutoff.
        """,
    )

with metric_right:
    formula_card(
        "Precision",
        [r"\text{Precision}=\frac{TP}{TP+FP}"],
        """
        Precision measures how often predicted impaired cases are truly
        impaired. It matters for GE-79 because false alarms affect how
        screening-style model outputs are interpreted.
        """,
    )

    formula_card(
        "Macro F1",
        [r"\text{Macro F1}=\frac{F1_{\text{class 0}}+F1_{\text{class 1}}}{2}"],
        """
        Macro F1 averages F1 across both cognitive-status classes equally. This
        is important for GE-79 because the impaired class is smaller and should
        not be hidden by majority-class performance.
        """,
    )

    formula_card(
        "Cross Validation",
        [r"\text{CV}=\frac{S_1+S_2+S_3+S_4+S_5}{5}"],
        """
        Cross validation averages model scores across five validation folds. It
        is appropriate for GE-79 because the dataset is small, so performance
        should not depend on one train-test split.
        """,
    )

    formula_card(
        "Balanced Accuracy",
        [r"\text{Balanced Accuracy}=\frac{\text{Recall}_{class0}+\text{Recall}_{class1}}{2}"],
        """
        Balanced Accuracy averages recall across both classes. It is useful for
        imbalanced medical datasets because each cognitive-status class receives
        equal weight.
        """,
    )

    formula_card(
        "Precision-Recall AUC Concept",
        [r"PR\text{-}AUC=\int_0^1 Precision(Recall)\,dRecall"],
        """
        Precision-Recall AUC summarizes the precision-recall tradeoff across
        thresholds. It is especially relevant when the impaired class is smaller.
        """,
    )

section("Confusion Matrix Terms")

confusion_left, confusion_right = st.columns(2, gap="large")

with confusion_left:
    formula_card(
        "Positive and Negative Prediction Counts",
        [
            r"TP=\text{true positives}",
            r"TN=\text{true negatives}",
            r"FP=\text{false positives}",
            r"FN=\text{false negatives}",
        ],
        """
        These four counts define the confusion matrix. They are the foundation
        for Accuracy, Precision, Recall, F1, and Macro F1.
        """,
    )

with confusion_right:
    formula_card(
        "Specificity",
        [r"\text{Specificity}=\frac{TN}{TN+FP}"],
        """
        Specificity measures how well the model identifies no-impairment cases.
        It is useful in medical classification because false alarms and missed
        cases both matter for interpretation.
        """,
    )

section("Bias Checkers and Responsible AI")

bias_left, bias_right = st.columns(2, gap="large")

with bias_left:
    formula_card(
        "Group Metric Gap",
        [r"\text{Gap}=\left|M_{groupA}-M_{groupB}\right|"],
        """
        A group metric gap compares performance across patient subgroups. It
        helps identify whether a model performs unevenly across demographic or
        clinical groups.
        """,
        """
        - **M:** selected metric such as Recall, Precision, or F1
        - **groupA/groupB:** comparison groups in a bias review
        """,
    )

    formula_card(
        "Demographic Parity Difference",
        [r"\Delta_{DP}=\left|P(\hat{Y}=1|A=a)-P(\hat{Y}=1|A=b)\right|"],
        """
        Demographic parity difference compares positive prediction rates across
        groups. It is a bias-checking formula, not a clinical performance metric,
        and helps flag possible representation or selection concerns.
        """,
        """
        - **A:** subgroup attribute
        - **a, b:** groups being compared
        - **hat Y = 1:** predicted impaired class
        """,
    )

with bias_right:
    formula_card(
        "SHAP Additive Explanation",
        [r"f(x)=\phi_0+\sum_{j=1}^{p}\phi_j"],
        """
        SHAP explains a prediction as a baseline value plus feature
        contributions. It supports Explainable AI by showing which biomarkers
        pushed a prediction higher or lower.
        """,
        """
        - **phi_0:** baseline model output
        - **phi_j:** contribution from feature j
        - **p:** number of features in the explanation
        """,
    )

    formula_card(
        "Equal Opportunity Difference",
        [r"\Delta_{EO}=\left|\text{Recall}_{groupA}-\text{Recall}_{groupB}\right|"],
        """
        Equal opportunity difference compares recall across groups. For GE-79,
        this matters because catching impaired participants should not depend
        unfairly on subgroup membership.
        """,
    )

final_math_context_expander(
    """
    The GE-79 project compares an interpretable baseline, a readable tree model,
    and an ensemble model after Random Forest feature selection. These algorithms
    fit the project because they support binary classification while preserving
    different levels of interpretability and stability.
    """
)

project_footer()
