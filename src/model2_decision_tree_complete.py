"""
============================================================================
 GE-79 / CDED  -  Group 6C  |  AI4ALL Ignite 2026
 MODEL 2 of 3:  DECISION TREE  (Interpretable)   *** COMPLETE / FUNCTIONAL ***
 Label stamped on all figures:  GE-79 · Decision Tree · Interpretable
============================================================================
 WHERE THIS FILE GOES (PyCharm):
     AI4ALL_ML-Diabetes_-Ver_1_A/
     └── src/
         └── Model2_decision_tree_complete.py      <-- THIS FILE

 IT READS:
     ../data/GE79_MASTER_DATASET_V1.csv    (input dataset)
     ../outputs/model0_FINAL_FEATURES.csv  (the 14 features, made by Model 0)

 IT WRITES (to ../outputs/):
     model2_fig_dt_tree.png       - the decision-tree diagram (headline visual)
     model2_fig_dt_confusion.png  - confusion matrix
     model2_fig_roc_auc.png       - ROC curve with ROC-AUC
     model2_fig_pr_auc.png        - Precision-Recall curve with PR-AUC
     model2_results_tree.csv      - metrics row for the Phase-8 comparison table

 ----------------------------------------------------------------------------
 RUN ORDER:  Run Model 0 FIRST (it creates FINAL_FEATURES.csv). Then this file.
             Models 1, 2 and 3 can then run in any order.
 ----------------------------------------------------------------------------

 WHAT THIS MODEL IS
   Decision Tree = a flowchart of yes/no questions on the features. Each split
   sends a patient left or right until a leaf assigns a class. It is the most
   INTERPRETABLE model: you can literally read the rules off the diagram, which
   is why it anchors the project's explainability story.

 TARGET ENCODING
   0 = No Impairment   (MMSE >= 28, "Normal")
   1 = Impaired        (MMSE 25-27)

 EVALUATION METRICS
   - Accuracy
   - Precision
   - Recall
   - Macro F1-score
   - Confusion Matrix
   - ROC Curve
   - ROC-AUC

 LEAKAGE CONTROL
   - MMSE (defines the label) is NOT a feature  -> no target leakage.
   - One row per patient (Visit 2)              -> no repeated-measures leakage.
   - Imputation + scaling happen INSIDE the CV folds via a Pipeline.

 NOTE ON COMMENTS: Python uses '#' for comments (not '//'). Every step below
 is commented inline so each line's purpose is clear to a first-time reader.
============================================================================
"""

## --- Standard library ------------------------------------------------------
import runpy                          # lets Streamlit Cloud delegate to the lightweight app
import sys                            # lets the fallback import app-local helpers
import warnings                       # used to silence non-critical warnings
from pathlib import Path              # safe, OS-independent file paths


def _run_lightweight_streamlit_app_if_needed():
    ## If Streamlit runs this script, show the dashboard instead of console output.
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return

    if get_script_run_ctx(suppress_warning=True) is not None:
        app_dir = Path(__file__).resolve().parents[1] / "app"
        app_dir_str = str(app_dir)
        if app_dir_str in sys.path:
            sys.path.remove(app_dir_str)
        sys.path.insert(0, app_dir_str)
        loaded_helper = sys.modules.get("echarts_components")
        if loaded_helper is not None and not str(getattr(loaded_helper, "__file__", "")).startswith(app_dir_str):
            del sys.modules["echarts_components"]
        app_path = app_dir / "model2_streamlit.py"
        runpy.run_path(str(app_path), run_name="__main__")
        st.stop()


_run_lightweight_streamlit_app_if_needed()

## --- Third-party data + math -----------------------------------------------
import numpy as np                    # numeric helpers (used for dtype checks)
import pandas as pd                   # tables / CSV reading + writing

## --- Plotting (headless-safe) ----------------------------------------------
import matplotlib                     # base plotting library
matplotlib.use("Agg")                 # "Agg" = render to file, no pop-up window
import matplotlib.pyplot as plt       # the plotting interface we actually call

## --- scikit-learn machine-learning pieces ----------------------------------
from sklearn.compose import ColumnTransformer        # apply different prep to num/cat cols
from sklearn.impute import SimpleImputer             # fill in missing values
from sklearn.metrics import (accuracy_score,         # overall % correct
                             average_precision_score, # area under precision-recall curve
                             classification_report,  # per-class precision/recall/F1
                             confusion_matrix,       # TP/FP/FN/TN grid
                             f1_score,               # balance of precision & recall
                             precision_recall_curve, # PR curve points from probabilities
                             precision_score,        # of predicted-positive, how many right
                             recall_score,           # of actual-positive, how many caught
                             roc_auc_score,          # ROC-AUC from predicted probabilities
                             roc_curve)              # ROC curve points from probabilities
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # CV tools
from sklearn.pipeline import Pipeline                # chain prep + model as one unit
from sklearn.preprocessing import OneHotEncoder, StandardScaler         # encode + scale
from sklearn.tree import DecisionTreeClassifier, plot_tree              # the model + its diagram
from neumorphic_visualizations import save_precision_recall_curve, save_roc_curve

warnings.filterwarnings("ignore")     # hide harmless convergence/version warnings
RANDOM_STATE = 42                     # fixed seed -> same result every run (reproducible)

# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]       # repo root, no matter where script runs
DATA_PATH = PROJECT_ROOT / "data" / "GE79_MASTER_DATASET_V1.csv"   # the cleaned input dataset
FEATURES_PATH = PROJECT_ROOT / "outputs" / "model0_FINAL_FEATURES.csv"  # written by Model 0
OUT_DIR = PROJECT_ROOT / "outputs"                       # where figures + CSVs are saved
OUT_DIR.mkdir(parents=True, exist_ok=True)               # create outputs/ if missing

# Figure-label standard:  DATASET · MODEL · description (encoding)
DATASET = "GE-79"                     # dataset name stamped on every figure
MODEL_NAME = "Decision Tree"          # model name stamped on every figure
MODEL_ROLE = "Interpretable"          # this model's role in the project
CLASS_0_LABEL = "No Impairment"       # human-readable name for code 0
CLASS_1_LABEL = "Impaired"            # human-readable name for code 1
ENCODING = f"0 = {CLASS_0_LABEL}, 1 = {CLASS_1_LABEL}"   # printed under each title

TARGET_COL = "cognitive_status_label"                    # the column we predict
TARGET_MAP = {"Normal": 0, "Mild Impairment": 1}         # text label -> numeric code

# Columns that must NEVER be used as predictors (ids, target copies, junk cols)
DROP_COLS = ["patient_id", "visit", "cognitive_status_code",
             "cognitive_status_label", "gait_walk1_distance_m", "dm_status"]
ADD_MISSING_FLAG_FOR = ["diabetes_duration"]   # 45% missing -> impute + add a 0/1 flag

NAVY, TEAL = "#1f3a5f", "#2a9d8f"     # brand colors for the figures


def caption(description):
    """Build the standard two-line figure title used on every model's figures."""
    # Line 1: "GE-79 · Decision Tree — <description>"
    # Line 2: "Interpretable model  (0 = No Impairment, 1 = Impaired)"
    return f"{DATASET} · {MODEL_NAME} — {description}\n{MODEL_ROLE} model  ({ENCODING})"


# ---------------------------------------------------------------------------
# 1. LOAD DATA + THE LOCKED FEATURE LIST
# ---------------------------------------------------------------------------
def load_data_and_features():
    df = pd.read_csv(DATA_PATH)                 # read the 75-patient dataset into a table

    # For each high-missing column, add a 0/1 "was this value missing?" flag.
    # This lets the model use missingness itself as a signal, instead of hiding it.
    for col in ADD_MISSING_FLAG_FOR:
        df[f"{col}_missing"] = df[col].isnull().astype(int)   # 1 if missing, else 0

    y = df[TARGET_COL].map(TARGET_MAP)          # convert text labels -> 0/1 target vector

    # Reuse the EXACT 14 features Model 0 selected, so all three models match.
    if FEATURES_PATH.exists():                  # normal path: Model 0 already ran
        final_features = pd.read_csv(FEATURES_PATH)["final_features"].tolist()
        # If we kept a column's missing-flag, make sure that flag is included too.
        for col in ADD_MISSING_FLAG_FOR:
            flag = f"{col}_missing"
            if col in final_features and flag in df.columns and flag not in final_features:
                final_features.append(flag)
        print(f"Loaded {len(final_features)} locked features from {FEATURES_PATH.name}")
    else:                                        # safety net: stop early with a clear message
        raise FileNotFoundError(
            "model0_FINAL_FEATURES.csv not found. Run Model 0 "
            "(model0_feature_selection.py) FIRST.")

    X = df[final_features]                       # X = just the predictor columns
    print(f"Dataset: {len(df)} patients | features used: {X.shape[1]}")
    print("Target:", df[TARGET_COL].value_counts().to_dict())   # show class balance
    return X, y                                  # hand X and y back to the caller


# ---------------------------------------------------------------------------
# 2. PREPROCESSING (fit inside CV folds only -> no leakage)
# ---------------------------------------------------------------------------
def build_preprocessor(X):
    # Split columns by type so each gets the right treatment.
    numeric = X.select_dtypes(include=np.number).columns.tolist()      # number columns
    categorical = X.select_dtypes(include="object").columns.tolist()   # text columns

    # Numeric pipeline: fill blanks with the median, then standardize the scale.
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])

    # Categorical pipeline: fill blanks with the most common value, then one-hot encode.
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("encode", OneHotEncoder(drop="first", handle_unknown="ignore"))])

    # Bundle both pipelines so they run on the correct columns automatically.
    return ColumnTransformer([("num", num_pipe, numeric),
                              ("cat", cat_pipe, categorical)])


# ---------------------------------------------------------------------------
# 3. THE DECISION TREE
#    max_depth=4        : keep the tree shallow so it stays readable
#    min_samples_leaf=5 : each leaf needs >=5 patients (avoids tiny over-fit leaves)
#    class_weight=balanced: up-weights the minority "Impaired" class
# ---------------------------------------------------------------------------
def get_decision_tree():
    # Return the model with the EXACT settings fixed by the project contract.
    return DecisionTreeClassifier(
        max_depth=4,                  # at most 4 questions deep -> human-readable
        min_samples_leaf=5,           # no leaf may rely on fewer than 5 patients
        class_weight="balanced",      # counteract the 73/27 class imbalance
        random_state=RANDOM_STATE)    # reproducible tree every run


# ---------------------------------------------------------------------------
# 4. EVALUATE WITH 5-FOLD STRATIFIED CROSS-VALIDATION
# ---------------------------------------------------------------------------
def evaluate(X, y):
    pre = build_preprocessor(X)                 # build the prep step (Section 2)

    # Stratified 5-fold = split into 5 parts, each keeping the 73/27 class ratio.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # One pipeline = preprocessing + the tree, so prep is fit on training folds only.
    tree = Pipeline([("pre", pre), ("clf", get_decision_tree())])

    # cross_val_predict gives each patient a prediction from the fold where they were held out.
    y_pred = cross_val_predict(tree, X, y, cv=cv)
    y_prob = cross_val_predict(tree, X, y, cv=cv, method="predict_proba")[:, 1]

    # Assemble the one-row results table (EXACT column names required by the contract).
    results = pd.DataFrame([{
        "model": "Decision Tree",
        "accuracy": round(accuracy_score(y, y_pred), 3),                                 # overall correct
        "precision_macro": round(precision_score(y, y_pred, average="macro", zero_division=0), 3),  # class-balanced precision
        "recall_macro": round(recall_score(y, y_pred, average="macro", zero_division=0), 3),        # class-balanced recall
        "f1_macro": round(f1_score(y, y_pred, average="macro", zero_division=0), 3),                # class-balanced F1 (primary metric)
        "recall_impaired": round(recall_score(y, y_pred, pos_label=1, zero_division=0), 3),         # how many Impaired we caught
    }])
    results.to_csv(OUT_DIR / "model2_results_tree.csv", index=False)   # save for the comparison table

    # Print results so the run is self-documenting in the console.
    print("\n=== MODEL 2: Decision Tree — 5-fold CV results ===")
    print(results.to_string(index=False))
    print("\nMajority-class baseline accuracy = 0.733 (always predicts 'No Impairment').")
    return y_pred, y_prob                       # later figures need class predictions + probabilities


# ---------------------------------------------------------------------------
# 5. FIGURE A — THE DECISION-TREE DIAGRAM (headline visual)
# ---------------------------------------------------------------------------
def plot_tree_diagram(X, y):
    pre = build_preprocessor(X)                 # same prep as evaluation
    tree = Pipeline([("pre", pre), ("clf", get_decision_tree())])
    tree.fit(X, y)                              # fit on ALL data so we can draw the final rules

    fig, ax = plt.subplots(figsize=(17, 8.5))   # wide canvas so the tree is legible
    plot_tree(tree.named_steps["clf"],          # the fitted tree object
              feature_names=tree.named_steps["pre"].get_feature_names_out(),  # readable feature labels
              class_names=[f"0:{CLASS_0_LABEL}", f"1:{CLASS_1_LABEL}"],       # leaf class labels
              filled=True,                      # color nodes by majority class
              rounded=True,                     # rounded boxes (nicer look)
              fontsize=8, ax=ax)                # small font so text fits in nodes
    ax.set_title(caption("Full Diagram"), fontweight="bold", fontsize=13)     # standard title
    plt.tight_layout()                          # trim extra whitespace
    plt.savefig(OUT_DIR / "model2_fig_dt_tree.png", dpi=130)   # save the figure
    plt.close()                                 # free the figure from memory
    print("Saved model2_fig_dt_tree.png")


# ---------------------------------------------------------------------------
# 6. FIGURE B — CONFUSION MATRIX
# ---------------------------------------------------------------------------
def plot_confusion(y, y_pred):
    cm = confusion_matrix(y, y_pred)            # 2x2 grid: rows = actual, cols = predicted
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    ax.imshow(cm, cmap="Blues")                 # shade cells by count

    lab = [f"0 · {CLASS_0_LABEL}", f"1 · {CLASS_1_LABEL}"]   # axis tick labels
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(lab); ax.set_yticklabels(lab)

    # Write the count inside each cell; flip text color on dark cells for contrast.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=18,
                    fontweight="bold", color="white" if cm[i, j] > cm.max() / 2 else NAVY)

    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Actual", fontweight="bold")
    ax.set_title(caption("Confusion Matrix"), fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model2_fig_dt_confusion.png", dpi=150)   # save the figure
    plt.close()
    print("Saved model2_fig_dt_confusion.png")


def plot_roc_and_pr(y, y_prob):
    ## Convert held-out probabilities into threshold curves for model evaluation.
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = roc_auc_score(y, y_prob)
    precision, recall, _ = precision_recall_curve(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    ## Save standardized ROC-AUC and PR-AUC figures used across all models.
    save_roc_curve(
        fpr,
        tpr,
        roc_auc,
        OUT_DIR / "model2_fig_roc_auc.png",
        2,
        "Decision Tree",
    )
    save_precision_recall_curve(
        recall,
        precision,
        pr_auc,
        OUT_DIR / "model2_fig_pr_auc.png",
        2,
        "Decision Tree",
    )
    print("Saved model2_fig_roc_auc.png")
    print("Saved model2_fig_pr_auc.png")
    print("\n" + "=" * 50)
    print("ROC-AUC RESULTS")
    print("=" * 50)
    print("\nModel 2 (Decision Tree)")
    print(f"\nAUC = {roc_auc:.3f}")
    print(f"PR-AUC = {pr_auc:.3f}")


# ---------------------------------------------------------------------------
# 7. MAIN  (runs the whole model in order when you click Run in PyCharm)
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print(" MODEL 2 of 3 — DECISION TREE  (GE-79 · Decision Tree · Interpretable)")
    print("=" * 64)
    X, y = load_data_and_features()             # 1) load data + the 14 features
    y_pred, y_prob = evaluate(X, y)              # 2) cross-validate + save metrics
    plot_tree_diagram(X, y)                      # 3) save the tree diagram
    plot_confusion(y, y_pred)                    # 4) save the confusion matrix
    plot_roc_and_pr(y, y_prob)                   # 5) save ROC-AUC and PR-AUC figures

    # Final per-class breakdown printed to the console.
    print("\nPer-class report (Decision Tree):")
    print(classification_report(y, y_pred,
          target_names=[CLASS_0_LABEL, CLASS_1_LABEL], zero_division=0))
    print("\nDone. Outputs written to ../outputs/")


# Standard Python entry point: only run main() when this file is executed directly
# (not when it is imported by another script).
if __name__ == "__main__":
    main()


# ===========================================================================
#  [Dictated by E. Hannan]
# ===========================================================================
