"""
Stress Level Detection from Lifestyle Data
==========================================
Machine Learning Classification Mini Project

Problem : Classify a person's stress level (Low / Medium / High)
          from everyday lifestyle habits.
Run     : python stress_detection.py
Data    : Uses 'stress_data.csv' if present (e.g. from your own Google Form
          survey). Otherwise a synthetic dataset is generated so the project
          runs end to end.
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
os.makedirs("outputs", exist_ok=True)

TARGET = "stress_level"
CLASSES = ["Low", "Medium", "High"]
NUMERIC = ["age", "sleep_hours", "screen_time_hours", "caffeine_cups",
           "exercise_min_per_week", "study_work_hours", "social_hours_per_week"]
CATEGORICAL = ["gender", "diet_quality"]


# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------
def generate_synthetic_data(n=1500):
    """Synthetic lifestyle data. Stress is driven by habits plus random noise,
    so the model has a real (but not perfect) pattern to learn."""
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame({
        "age": rng.integers(17, 40, n),
        "gender": rng.choice(["Male", "Female", "Other"], n, p=[0.48, 0.48, 0.04]),
        "sleep_hours": np.clip(rng.normal(6.8, 1.3, n), 3, 10).round(1),
        "screen_time_hours": np.clip(rng.normal(6.5, 2.5, n), 1, 16).round(1),
        "caffeine_cups": rng.poisson(2.0, n).clip(0, 8),
        "exercise_min_per_week": np.clip(rng.normal(120, 80, n), 0, 420).round(0),
        "study_work_hours": np.clip(rng.normal(7, 2.5, n), 1, 14).round(1),
        "social_hours_per_week": np.clip(rng.normal(8, 5, n), 0, 30).round(1),
        "diet_quality": rng.choice(["Poor", "Average", "Good"], n, p=[0.25, 0.5, 0.25]),
    })

    diet_pts = df["diet_quality"].map({"Poor": 1.0, "Average": 0.0, "Good": -1.0})
    score = (
        -1.1 * (df["sleep_hours"] - 7)
        + 0.25 * (df["screen_time_hours"] - 6)
        + 0.35 * (df["caffeine_cups"] - 2)
        - 0.012 * (df["exercise_min_per_week"] - 120)
        + 0.30 * (df["study_work_hours"] - 7)
        - 0.06 * (df["social_hours_per_week"] - 8)
        + 0.5 * diet_pts
        + rng.normal(0, 1.0, n)  # human randomness
    )
    df[TARGET] = pd.cut(score, bins=[-np.inf, np.quantile(score, 0.35),
                                     np.quantile(score, 0.75), np.inf],
                        labels=CLASSES).astype(str)

    # Make the data messy like a real survey: ~4% missing values
    for col in ["sleep_hours", "screen_time_hours", "caffeine_cups", "diet_quality"]:
        df.loc[rng.random(n) < 0.04, col] = np.nan
    return df


def load_data():
    if os.path.exists("stress_data.csv"):
        print("Loaded your own dataset: stress_data.csv")
        return pd.read_csv("stress_data.csv")
    print("stress_data.csv not found -> generating synthetic dataset")
    df = generate_synthetic_data()
    df.to_csv("stress_data.csv", index=False)
    return df


df = load_data()
print("\nShape:", df.shape)
print(df.head())
print("\nMissing values per column:\n", df.isnull().sum())
print("\nClass distribution:\n", df[TARGET].value_counts())

# ----------------------------------------------------------------------
# 2. EDA (quick plots)
# ----------------------------------------------------------------------
plt.figure(figsize=(5, 4))
sns.countplot(x=TARGET, data=df, order=CLASSES, palette="viridis")
plt.title("Class distribution")
plt.tight_layout()
plt.savefig("outputs/class_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df[NUMERIC].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation of numeric features")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 3. PREPROCESSING
#    - impute missing values (median for numbers, most frequent for categories)
#    - one-hot encode categoricals
#    - scale numeric features
#    All wrapped in a Pipeline so there is NO data leakage from the test set.
# ----------------------------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), NUMERIC),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
])

# ----------------------------------------------------------------------
# 4. MODELS
# ----------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVM": SVC(kernel="rbf", probability=True, random_state=SEED),
}

results, fitted = [], {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1_macro").mean()
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="macro"),
        "Recall": recall_score(y_test, pred, average="macro"),
        "F1": f1_score(y_test, pred, average="macro"),
        "CV F1 (5-fold)": cv,
    })
    fitted[name] = (pipe, pred)

res_df = pd.DataFrame(results).set_index("Model").round(3)
print("\n=== MODEL COMPARISON ===")
print(res_df)
res_df.to_csv("outputs/model_comparison.csv")

# ----------------------------------------------------------------------
# 5. EVALUATION OF THE BEST MODEL
# ----------------------------------------------------------------------
best_name = res_df["F1"].idxmax()
best_pipe, best_pred = fitted[best_name]
print(f"\nBest model: {best_name}")
print(classification_report(y_test, best_pred, labels=CLASSES))

cm = confusion_matrix(y_test, best_pred, labels=CLASSES)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 6. FEATURE IMPORTANCE (interpretation)
# ----------------------------------------------------------------------
rf_pipe = fitted["Random Forest"][0]
feat_names = rf_pipe.named_steps["prep"].get_feature_names_out()
feat_names = [f.split("__")[1] for f in feat_names]
importances = rf_pipe.named_steps["model"].feature_importances_
imp = pd.Series(importances, index=feat_names).sort_values()

plt.figure(figsize=(7, 5))
imp.plot(kind="barh", color="teal")
plt.title("What drives stress? (Random Forest feature importance)")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.close()
print("\nTop factors:\n", imp.sort_values(ascending=False).head(5).round(3))

# ----------------------------------------------------------------------
# 7. SAVE MODEL + DEMO PREDICTION
# ----------------------------------------------------------------------
joblib.dump(best_pipe, "stress_model.pkl")
print("\nSaved model -> stress_model.pkl")

sample = pd.DataFrame([{
    "age": 21, "gender": "Female", "sleep_hours": 4.5, "screen_time_hours": 10,
    "caffeine_cups": 5, "exercise_min_per_week": 0, "study_work_hours": 11,
    "social_hours_per_week": 1, "diet_quality": "Poor"}])
print("\nDEMO - sample person:", sample.iloc[0].to_dict())
print("Predicted stress level:", best_pipe.predict(sample)[0])
print("Probabilities:", dict(zip(best_pipe.classes_,
                                 best_pipe.predict_proba(sample)[0].round(2))))
