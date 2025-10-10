# ==========================
# 📦 Install dependencies
# ==========================
# %pip install xgboost tabulate seaborn

# ==========================
# Import Libraries
# ==========================
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import time
from tabulate import tabulate
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             log_loss, classification_report, confusion_matrix)
from xgboost import XGBClassifier
from sklearn.svm import SVC

# ==========================
# 📂 Load Dataset
# ==========================
data_path = "/kaggle/input/japandata/PRJDB6472.csv"
df = pd.read_csv(data_path)
print("✅ Data loaded successfully. Shape:", df.shape)

# ==========================
# 🧹 Data Cleaning
# ==========================
df = df.dropna()  # remove missing rows

TARGET_COL = "Status"

# Normalize and encode target
df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.lower()
le_y = LabelEncoder()
y = le_y.fit_transform(df[TARGET_COL])
print("Class mapping:", dict(zip(le_y.classes_, le_y.transform(le_y.classes_))))

# Build features
X = df.drop(columns=[TARGET_COL, "Sample ID"], errors="ignore").copy()

# Encode categorical columns if any
if "clade_name" in X.columns:
    le_clade = LabelEncoder()
    X["clade_name"] = le_clade.fit_transform(X["clade_name"].astype(str))

# Sanitize column names (for XGBoost)
X.columns = (X.columns
             .str.strip()
             .str.replace(r"\s+", "_", regex=True)
             .str.replace(r"[\[\]<>]", "", regex=True)
             .str.replace(r"[^0-9A-Za-z_]", "", regex=True))

n_classes = len(np.unique(y))
print(f"Classes: {n_classes}, Samples: {len(y)}")

# ==========================
# ✂️ Split Data (80/20)
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n✅ Dataset Split:")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ==========================
# ⚙️ Cross-Validation Setup
# ==========================
SCORING = "accuracy"
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def compute_spw(y_vec):
    if len(np.unique(y_vec)) != 2:
        return None
    pos = np.sum(y_vec == 1)
    neg = len(y_vec) - pos
    return (neg / max(pos, 1)) if pos > 0 else 1.0

base_spw = compute_spw(y_train)

# ==========================
# 🤖 Model Definitions
# ==========================
models = {
    "XGBClassifier": {
        "estimator": XGBClassifier(
            objective=("binary:logistic" if n_classes == 2 else "multi:softprob"),
            eval_metric=("logloss" if n_classes == 2 else "mlogloss"),
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            **({"num_class": n_classes} if n_classes > 2 else {})
        ),
        "params": {
            "n_estimators": [200, 400, 600],
            "learning_rate": [0.02, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
            "gamma": [0, 0.2, 0.4],
            "reg_alpha": [0, 0.01, 0.1],
            "reg_lambda": [0.5, 1.0, 2.0],
        },
        "n_iter": 20
    },
    "SVC": {
        "estimator": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("svc", SVC(probability=True, random_state=42))
        ]),
        "params": {
            "svc__C": [0.5, 1, 2, 5],
            "svc__kernel": ["rbf"],
            "svc__gamma": ["scale", "auto"]
        },
        "n_iter": 15
    },
}

# ==========================
# 🧪 Training and Evaluation
# ==========================
results = {"Classifier": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "LogLoss": []}
conf_mats, class_reports = {}, {}

for name, cfg in models.items():
    print(f"\n🚀 Tuning {name} ...")
    search = RandomizedSearchCV(
        estimator=cfg["estimator"],
        param_distributions=cfg["params"],
        n_iter=cfg["n_iter"],
        scoring=SCORING,
        n_jobs=-1,
        cv=skf,
        verbose=1,
        random_state=42
    )
    start_time = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start_time

    best_model = search.best_estimator_
    print(f"Best {name} params:", search.best_params_)

    # Predictions on test set
    y_pred = best_model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    ll = np.nan
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test)
        try:
            ll = log_loss(y_test, y_proba)
        except Exception:
            pass

    # Store results
    results["Classifier"].append(name)
    results["Accuracy"].append(acc)
    results["Precision"].append(prec)
    results["Recall"].append(rec)
    results["F1"].append(f1)
    results["LogLoss"].append(ll)

    conf_mats[name] = confusion_matrix(y_test, y_pred)
    class_reports[name] = classification_report(y_test, y_pred, target_names=le_y.classes_)

# ==========================
# 📊 Comparison Table
# ==========================
df_res = pd.DataFrame(results)
print("\n📈 Model Comparison (Test Set)")
print(tabulate(df_res, headers="keys", tablefmt="pretty", showindex=False, floatfmt=".4f"))

best_idx = int(np.argmax(df_res["Accuracy"].values))
print(f"\n🏆 Best Model: {df_res.iloc[best_idx]['Classifier']} — Accuracy: {df_res.iloc[best_idx]['Accuracy']:.4f}")
