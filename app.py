# ==============================================================================
# M.Tech (AIML) - Machine Learning Assignment 2
# Interactive Credit Risk Classification & Model Evaluation Suite
# Student Name: Sresth Anand | BITS ID: 2026AJ05039 | Seed: 5039
# Dataset: Statlog (German Credit Data) - UCI Machine Learning Repository
# ==============================================================================

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.special import expit

st.set_page_config(
    page_title="Credit Risk ML Evaluation Suite | 2026AJ05039",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Credit Risk Classification — 5-Model Evaluation Suite")
st.caption("Student: Sresth Anand | BITS ID: 2026AJ05039 | M.Tech (AIML) Machine Learning Assignment 2 | Dataset: UCI Statlog German Credit (1,000 instances, 20 features)")

MODEL_REGISTRY = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/svm.pkl",
}

REQUIRED_FEATURES = [
    "checking_balance", "months_loan_duration", "credit_history", "purpose",
    "amount", "savings_balance", "employment_length", "installment_rate",
    "personal_status", "other_debtors", "residence_history", "property",
    "age", "installment_plan", "housing", "existing_credits",
    "dependents", "telephone", "foreign_worker", "job"
]

def transform_features(df_X, prep):
    cols_out = []
    for col in prep["num_cols"]:
        vals = df_X[col].astype(float).values
        cols_out.append(((vals - prep["num_mean"][col]) / prep["num_std"][col]).reshape(-1, 1))
    for col in prep["cat_cols"]:
        s_vals = df_X[col].astype(str).values
        for c in prep["cat_categories"][col]:
            cols_out.append((s_vals == c).astype(float).reshape(-1, 1))
    return np.hstack(cols_out)

def _predict_tree_one(x_row, node):
    while not node["is_leaf"]:
        node = node["left"] if x_row[node["feat"]] <= node["thr"] else node["right"]
    return node["prob1"]

def run_saved_model_inference(bundle, df_features):
    """Evaluate pre-trained model artifact on uploaded test CSV without retraining."""
    if hasattr(bundle, "predict") and hasattr(bundle, "predict_proba"):
        return np.asarray(bundle.predict(df_features), dtype=int), np.asarray(bundle.predict_proba(df_features)[:, 1], dtype=float)

    X = transform_features(df_features, bundle["preprocessor"])
    mtype = bundle["model_type"]

    if mtype == "logistic_regression":
        p1 = expit(X @ bundle["coef"] + bundle["intercept"])
    elif mtype == "decision_tree":
        p1 = np.array([_predict_tree_one(r, bundle["tree"]) for r in X], dtype=float)
    elif mtype == "knn":
        dists = cdist(X, bundle["X_train"], metric="euclidean")
        idx = np.argsort(dists, axis=1)[:, :int(bundle["n_neighbors"])]
        w = 1.0 / (np.take_along_axis(dists, idx, axis=1) + 1e-8)
        p1 = np.sum(w * bundle["y_train"][idx], axis=1) / np.sum(w, axis=1)
    elif mtype == "gaussian_nb":
        priors, theta, var = bundle["priors"], bundle["theta"], bundle["var"]
        lps = [np.log(priors[c]) - 0.5 * np.sum(np.log(2.0 * np.pi * var[c]) + ((X - theta[c]) ** 2) / var[c], axis=1) for c in (0, 1)]
        lp_mat = np.column_stack(lps)
        lp_mat -= np.max(lp_mat, axis=1, keepdims=True)
        probs = np.exp(lp_mat) / np.sum(np.exp(lp_mat), axis=1, keepdims=True)
        p1 = probs[:, 1]
    elif mtype == "svm_rbf":
        K_test = np.exp(-bundle["gamma_val"] * cdist(X, bundle["X_sv"], metric="sqeuclidean"))
        scores = K_test @ bundle["alpha"] + bundle["b"]
        p1 = expit(bundle["platt_a"] * scores + bundle["platt_b"])
        return (scores >= 0.0).astype(int), p1
    else:
        raise ValueError(f"Unsupported model_type: {mtype}")
    return (p1 >= 0.5).astype(int), p1

def compute_all_metrics(y_true, y_pred, y_prob):
    y_t, y_p, y_s = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), np.asarray(y_prob, dtype=float)
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    acc = (tp + tn) / max(len(y_t), 1)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    pos_s, neg_s = y_s[y_t == 1], y_s[y_t == 0]
    diff = pos_s[:, None] - neg_s[None, :]
    auc = float(np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / max(float(len(pos_s) * len(neg_s)), 1.0)
    return {
        "Accuracy": round(float(acc), 4), "AUC": round(float(auc), 4),
        "Precision": round(float(prec), 4), "Recall": round(float(rec), 4),
        "F1": round(float(f1), 4), "MCC": round(float(mcc), 4),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }

def build_classification_report_df(y_true, y_pred):
    y_t, y_p = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    rows, n_tot = [], max(len(y_t), 1)
    for cls, name in [(0, "Class 0 (Good Credit)"), (1, "Class 1 (Default Risk)")]:
        tp = int(np.sum((y_t == cls) & (y_p == cls)))
        fp = int(np.sum((y_t != cls) & (y_p == cls)))
        fn = int(np.sum((y_t == cls) & (y_p != cls)))
        sup = int(np.sum(y_t == cls))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        rows.append({"Class": name, "Precision": round(p, 4), "Recall": round(r, 4), "F1-Score": round(f, 4), "Support": sup})
    rows.append({
        "Class": "Macro Avg",
        "Precision": round(float(np.mean([r["Precision"] for r in rows[:2]])), 4),
        "Recall": round(float(np.mean([r["Recall"] for r in rows[:2]])), 4),
        "F1-Score": round(float(np.mean([r["F1-Score"] for r in rows[:2]])), 4),
        "Support": len(y_t),
    })
    rows.append({
        "Class": "Weighted Avg",
        "Precision": round(float(sum(r["Precision"] * r["Support"] for r in rows[:2]) / n_tot), 4),
        "Recall": round(float(sum(r["Recall"] * r["Support"] for r in rows[:2]) / n_tot), 4),
        "F1-Score": round(float(sum(r["F1-Score"] * r["Support"] for r in rows[:2]) / n_tot), 4),
        "Support": len(y_t),
    })
    return pd.DataFrame(rows)

@st.cache_resource
def load_all_saved_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return {k: pickle.load(open(os.path.join(base_dir, v), "rb")) for k, v in MODEL_REGISTRY.items()}

models_dict = load_all_saved_models()

# Sidebar: (a) Dataset Upload Option (CSV) & (b) Model Selection Dropdown
st.sidebar.header("⚙️ Step 6 App Controls")
uploaded_file = st.sidebar.file_uploader("a. Upload Test Data CSV (20 features + target)", type=["csv"])

base_dir = os.path.dirname(os.path.abspath(__file__))
default_test_path = os.path.join(base_dir, "test_data.csv")
if os.path.exists(default_test_path):
    st.sidebar.download_button("📥 Download Sample test_data.csv", data=open(default_test_path, "rb").read(), file_name="test_data.csv", mime="text/csv")

selected_model_name = st.sidebar.selectbox("b. Select Classification Model", list(MODEL_REGISTRY.keys()), index=3)

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.info(f"Loaded uploaded test file: `{uploaded_file.name}` ({len(df_test)} instances). Evaluating pre-trained models without retraining.")
else:
    df_test = pd.read_csv(default_test_path)
    st.info(f"Using bundled held-out `test_data.csv` ({len(df_test)} instances, Seed=5039). Upload any test CSV in the sidebar to evaluate dynamically.")

if "default" in df_test.columns and "target" in df_test.columns:
    df_test = df_test.drop(columns=["default"])

X_test_df = df_test[REQUIRED_FEATURES]
y_test_arr = df_test["target"].astype(int).values

all_eval_rows, model_outputs = [], {}
for m_name, m_bundle in models_dict.items():
    y_p, y_s = run_saved_model_inference(m_bundle, X_test_df)
    m_res = compute_all_metrics(y_test_arr, y_p, y_s)
    model_outputs[m_name] = {"y_pred": y_p, "y_prob": y_s, "metrics": m_res}
    all_eval_rows.append({"ML Model Name": m_name, **{k: m_res[k] for k in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}})

# (c) Display of Evaluation Metrics for Selected Model
sel_out = model_outputs[selected_model_name]
sel_met = sel_out["metrics"]
st.subheader(f"c. Evaluation Metrics — {selected_model_name}")
cols = st.columns(6)
for idx, (label, key) in enumerate([("Accuracy", "Accuracy"), ("AUC Score", "AUC"), ("Precision", "Precision"), ("Recall", "Recall"), ("F1 Score", "F1"), ("MCC Score", "MCC")]):
    cols[idx].metric(label, f"{sel_met[key]:.4f}")

# (d) Confusion Matrix & Classification Report
st.subheader(f"d. Confusion Matrix & Classification Report — {selected_model_name}")
col_cm, col_rep = st.columns([1.0, 1.3])
with col_cm:
    cm_mat = np.array([[sel_met["TN"], sel_met["FP"]], [sel_met["FN"], sel_met["TP"]]])
    fig_cm, ax_cm = plt.subplots(figsize=(4.8, 3.6), dpi=130)
    sns.heatmap(cm_mat, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred 0 (Good)", "Pred 1 (Default)"],
                yticklabels=["True 0 (Good)", "True 1 (Default)"], ax=ax_cm)
    ax_cm.set_title(f"Confusion Matrix: {selected_model_name}")
    st.pyplot(fig_cm)

with col_rep:
    st.dataframe(build_classification_report_df(y_test_arr, sel_out["y_pred"]), use_container_width=True, hide_index=True)

st.subheader("🏆 All 5 Models — Comparison Table on Active Test Data")
st.dataframe(pd.DataFrame(all_eval_rows), use_container_width=True, hide_index=True)
