import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit

BITS_ID = "2026AJ05039"
SEED = 5039
np.random.seed(SEED)

raw_path = "/working_dir/c_6139fa6580675f9d/user_uploaded_files/ttl=63d/user_uploaded/german_credit.csv"
df = pd.read_csv(raw_path)

# Remove redundant 1/2 encoded 'default' column so only the 20 true UCI features + binary 'target' (0=Good, 1=Default) remain
if "default" in df.columns and "target" in df.columns:
    df = df.drop(columns=["default"])

feature_cols = [c for c in df.columns if c != "target"]
num_cols = df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()

# Stratified 80/20 train-test split with SEED = 5039
def stratified_split(df_in, target_col="target", test_size=0.20, seed=5039):
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    y_vals = df_in[target_col].values
    for cls in np.unique(y_vals):
        cls_idx = np.where(y_vals == cls)[0]
        rng.shuffle(cls_idx)
        n_test = int(round(len(cls_idx) * test_size))
        test_idx.extend(cls_idx[:n_test])
        train_idx.extend(cls_idx[n_test:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return df_in.iloc[train_idx].reset_index(drop=True), df_in.iloc[test_idx].reset_index(drop=True)

train_df, test_df = stratified_split(df, "target", test_size=0.20, seed=SEED)

# Fit preprocessing pipeline (StandardScaler on numerical + OneHotEncoder drop_first on categorical)
class TabularPreprocessor:
    def __init__(self, num_cols, cat_cols):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.num_mean_ = {}
        self.num_std_ = {}
        self.cat_categories_ = {}
        self.encoded_feature_names_ = []

    def fit(self, df_X):
        self.encoded_feature_names_ = []
        for col in self.num_cols:
            vals = df_X[col].astype(float).values
            self.num_mean_[col] = float(np.mean(vals))
            std_val = float(np.std(vals))
            self.num_std_[col] = std_val if std_val > 1e-12 else 1.0
            self.encoded_feature_names_.append(col)
        for col in self.cat_cols:
            cats = sorted(df_X[col].astype(str).unique().tolist())
            # drop first category to avoid collinearity
            kept_cats = cats[1:] if len(cats) > 1 else cats
            self.cat_categories_[col] = kept_cats
            for c in kept_cats:
                self.encoded_feature_names_.append(f"{col}_{c}")
        return self

    def transform(self, df_X):
        cols_out = []
        for col in self.num_cols:
            vals = df_X[col].astype(float).values
            cols_out.append(((vals - self.num_mean_[col]) / self.num_std_[col]).reshape(-1, 1))
        for col in self.cat_cols:
            s_vals = df_X[col].astype(str).values
            for c in self.cat_categories_[col]:
                cols_out.append((s_vals == c).astype(float).reshape(-1, 1))
        return np.hstack(cols_out)

    def fit_transform(self, df_X):
        return self.fit(df_X).transform(df_X)

preprocessor = TabularPreprocessor(num_cols, cat_cols)
X_train = preprocessor.fit_transform(train_df[feature_cols])
y_train = train_df["target"].astype(int).values
X_test = preprocessor.transform(test_df[feature_cols])
y_test = test_df["target"].astype(int).values

print(f"Encoded X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"Train class counts: 0={(y_train==0).sum()}, 1={(y_train==1).sum()}")
print(f"Test class counts : 0={(y_test==0).sum()}, 1={(y_test==1).sum()}")

# 1. Logistic Regression (L2 regularized L-BFGS)
class LogisticRegressionModel:
    def __init__(self, C=1.0, seed=5039):
        self.C = float(C)
        self.seed = int(seed)
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        n, d = X.shape
        reg = 1.0 / (self.C * n)
        def obj_and_grad(params):
            w = params[:-1]
            b = params[-1]
            z = X @ w + b
            p = expit(z)
            eps = 1e-15
            p_clip = np.clip(p, eps, 1.0 - eps)
            loss = -np.mean(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip)) + 0.5 * reg * np.dot(w, w)
            diff = (p - y) / n
            gw = X.T @ diff + reg * w
            gb = float(np.sum(diff))
            return loss, np.append(gw, gb)
        res = minimize(obj_and_grad, np.zeros(d + 1), method="L-BFGS-B", jac=True)
        self.coef_ = res.x[:-1]
        self.intercept_ = float(res.x[-1])
        return self

    def predict_proba(self, X):
        p1 = expit(X @ self.coef_ + self.intercept_)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# 2. Decision Tree Classifier (CART with Gini impurity)
class DecisionTreeModel:
    def __init__(self, max_depth=5, min_samples_leaf=10, seed=5039):
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.seed = int(seed)
        self.tree_ = None

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        p1 = np.mean(y == 1)
        return float(2.0 * p1 * (1.0 - p1))

    def _build(self, X, y, depth):
        p1 = float(np.mean(y == 1)) if len(y) > 0 else 0.0
        node = {"prob1": p1, "pred": int(p1 >= 0.5), "is_leaf": True}
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or len(np.unique(y)) == 1:
            return node

        best_gain = 1e-7
        parent_gini = self._gini(y)
        best_feat, best_thr = None, None
        n, d = X.shape

        for j in range(d):
            col = X[:, j]
            uniq = np.unique(col)
            if len(uniq) <= 1:
                continue
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0
            for thr in thresholds:
                left_mask = col <= thr
                n_left = int(np.sum(left_mask))
                n_right = n - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                gain = parent_gini - (n_left / n) * self._gini(y[left_mask]) - (n_right / n) * self._gini(y[~left_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thr = float(thr)

        if best_feat is None:
            return node

        left_mask = X[:, best_feat] <= best_thr
        node["is_leaf"] = False
        node["feat"] = int(best_feat)
        node["thr"] = best_thr
        node["left"] = self._build(X[left_mask], y[left_mask], depth + 1)
        node["right"] = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def fit(self, X, y):
        self.tree_ = self._build(X, y, 0)
        return self

    def _pred_one(self, x, node):
        while not node["is_leaf"]:
            if x[node["feat"]] <= node["thr"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["prob1"]

    def predict_proba(self, X):
        p1 = np.array([self._pred_one(row, self.tree_) for row in X], dtype=float)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# 3. K-Nearest Neighbor Classifier (Distance-weighted Euclidean KNN)
class KNNModel:
    def __init__(self, n_neighbors=13, weights="distance"):
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y):
        self.X_train_ = np.array(X, dtype=float, copy=True)
        self.y_train_ = np.array(y, dtype=int, copy=True)
        return self

    def predict_proba(self, X):
        dists = cdist(X, self.X_train_, metric="euclidean")
        idx = np.argsort(dists, axis=1)[:, :self.n_neighbors]
        k_dists = np.take_along_axis(dists, idx, axis=1)
        k_labels = self.y_train_[idx]
        if self.weights == "distance":
            w = 1.0 / (k_dists + 1e-8)
            p1 = np.sum(w * k_labels, axis=1) / np.sum(w, axis=1)
        else:
            p1 = np.mean(k_labels, axis=1)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# 4. Gaussian Naive Bayes Classifier
class GaussianNBModel:
    def __init__(self, var_smoothing=1e-2):
        self.var_smoothing = float(var_smoothing)
        self.classes_ = np.array([0, 1])
        self.priors_ = None
        self.theta_ = None
        self.var_ = None

    def fit(self, X, y):
        eps = self.var_smoothing * np.max(np.var(X, axis=0))
        self.priors_ = np.array([np.mean(y == 0), np.mean(y == 1)], dtype=float)
        self.theta_ = np.vstack([np.mean(X[y == 0], axis=0), np.mean(X[y == 1], axis=0)])
        self.var_ = np.vstack([np.var(X[y == 0], axis=0) + eps, np.var(X[y == 1], axis=0) + eps])
        return self

    def predict_proba(self, X):
        log_probs = []
        for c in (0, 1):
            lp = np.log(self.priors_[c]) - 0.5 * np.sum(
                np.log(2.0 * np.pi * self.var_[c]) + ((X - self.theta_[c]) ** 2) / self.var_[c],
                axis=1
            )
            log_probs.append(lp)
        lp_mat = np.column_stack(log_probs)
        lp_mat -= np.max(lp_mat, axis=1, keepdims=True)
        probs = np.exp(lp_mat)
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# 5. Support Vector Machine (RBF Kernel SVM via L2-regularized Dual Kernel Ridge / Hinge-Squared + Platt Calibration)
class SVMModel:
    def __init__(self, C=1.5, gamma=None, seed=5039):
        self.C = float(C)
        self.gamma = gamma
        self.seed = int(seed)
        self.X_sv_ = None
        self.alpha_ = None
        self.b_ = 0.0
        self.gamma_val_ = None
        self.platt_a_ = 1.0
        self.platt_b_ = 0.0

    def _rbf(self, X1, X2):
        sq_dists = cdist(X1, X2, metric="sqeuclidean")
        return np.exp(-self.gamma_val_ * sq_dists)

    def fit(self, X, y):
        n, d = X.shape
        self.X_sv_ = np.array(X, dtype=float, copy=True)
        self.gamma_val_ = float(self.gamma) if self.gamma is not None else 1.0 / (d * float(np.var(X)))
        y_signed = np.where(y == 1, 1.0, -1.0)
        K = self._rbf(X, X)

        # Optimize L2-regularized squared hinge objective in dual/Representer form:
        # 0.5 * alpha^T K alpha + C * sum(max(0, 1 - y_i (K_i alpha + b))^2)
        def svm_obj(params):
            al = params[:-1]
            b = params[-1]
            f = K @ al + b
            margin = 1.0 - y_signed * f
            active = np.maximum(0.0, margin)
            Ka = K @ al
            loss = 0.5 * float(al @ Ka) + self.C * float(np.sum(active ** 2))
            d_f = -2.0 * self.C * y_signed * active
            grad_al = Ka + K @ d_f
            grad_b = float(np.sum(d_f))
            return loss, np.append(grad_al, grad_b)

        res = minimize(svm_obj, np.zeros(n + 1), method="L-BFGS-B", jac=True, options={"maxiter": 400})
        self.alpha_ = res.x[:-1]
        self.b_ = float(res.x[-1])

        # Platt scaling on decision function f(X)
        scores = K @ self.alpha_ + self.b_
        def platt_obj(ab):
            a, b0 = ab
            p = expit(a * scores + b0)
            p_c = np.clip(p, 1e-15, 1.0 - 1e-15)
            return -np.mean(y * np.log(p_c) + (1.0 - y) * np.log(1.0 - p_c))
        p_res = minimize(platt_obj, np.array([1.0, 0.0]), method="BFGS")
        self.platt_a_, self.platt_b_ = float(p_res.x[0]), float(p_res.x[1])
        return self

    def decision_function(self, X):
        K_test = self._rbf(X, self.X_sv_)
        return K_test @ self.alpha_ + self.b_

    def predict_proba(self, X):
        scores = self.decision_function(X)
        p1 = expit(self.platt_a_ * scores + self.platt_b_)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.decision_function(X) >= 0.0).astype(int)

# Evaluation metric functions (exact formulas matching sklearn.metrics)
def calc_metrics(y_true, y_pred, y_prob):
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    y_s = np.asarray(y_prob, dtype=float)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))

    acc = (tp + tn) / len(y_t)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    # Exact ROC AUC via Mann-Whitney U / Wilcoxon rank-sum statistic
    pos_scores = y_s[y_t == 1]
    neg_scores = y_s[y_t == 0]
    diff = pos_scores[:, None] - neg_scores[None, :]
    auc = float(np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / float(len(pos_scores) * len(neg_scores))

    return {
        "Accuracy": round(float(acc), 4),
        "AUC": round(float(auc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1": round(float(f1), 4),
        "MCC": round(float(mcc), 4),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }

models = [
    ("Logistic Regression", LogisticRegressionModel(C=0.8, seed=SEED)),
    ("Decision Tree", DecisionTreeModel(max_depth=5, min_samples_leaf=10, seed=SEED)),
    ("KNN", KNNModel(n_neighbors=11, weights="distance")),
    ("Naive Bayes", GaussianNBModel(var_smoothing=1e-2)),
    ("Support Vector Machine", SVMModel(C=0.5, gamma=0.025, seed=SEED)),
]

print(f"\n{'ML Model Name':<24} | {'Accuracy':>8} | {'AUC':>8} | {'Precision':>9} | {'Recall':>8} | {'F1':>8} | {'MCC':>8}")
print("-" * 88)
for name, m in models:
    m.fit(X_train, y_train)
    yp = m.predict(X_test)
    ys = m.predict_proba(X_test)[:, 1]
    met = calc_metrics(y_test, yp, ys)
    print(f"{name:<24} | {met['Accuracy']:8.4f} | {met['AUC']:8.4f} | {met['Precision']:9.4f} | {met['Recall']:8.4f} | {met['F1']:8.4f} | {met['MCC']:8.4f}")
