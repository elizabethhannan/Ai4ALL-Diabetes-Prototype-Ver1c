# Streamlit Cloud Deployment

Use the same GitHub repository and branch for all four model apps:

- Repository: `elizabethhannan/AI4ALL-Diabetes-PRIVATE-ML`
- Branch: `main`
- Python runtime: `python-3.12` from `runtime.txt`

Model app entrypoints:

| App | Main module |
| --- | --- |
| Model 0 Feature Selection | `src/model0_feature_selection.py` |
| Model 1 Logistic Regression | `src/model1_logistic_regression.py` |
| Model 2 Decision Tree | `src/model2_decision_tree_complete.py` |
| Model 3 Random Forest | `src/model3_random_forest.py` |

Each `src/` module delegates to the lightweight dashboard in `app/` when it is
run by Streamlit Cloud, so the apps show the precomputed outputs instead of
rerunning the full model-training workflow.

If Streamlit Cloud logs say `Failed to download the sources`, the app has not
reached Python yet. Reconnect the app to GitHub or reauthorize Streamlit Cloud
for the private repository, then reboot the app. After the clone succeeds, the
root `requirements.txt` and `runtime.txt` are used to install the dashboard
dependencies.
