import re
import math
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from feature_utils import TextExtractor, HandcraftedTransformer

# -----------------------------
# Load dataset (JSONL)
df = pd.read_json("data/problems.jsonl", lines=True)

text_cols = ["title", "description", "input_description", "output_description"]
for col in text_cols:
    df[col] = df[col].fillna("")

df["problem_score"] = pd.to_numeric(df["problem_score"], errors="coerce")
df = df.dropna(subset=["problem_class", "problem_score"])

print(df.head())
print(df.columns)

# -----------------------------
# Feature pipeline
# -----------------------------
tfidf = Pipeline(
    [
        ("text", TextExtractor()),
        (
            "tfidf",
            TfidfVectorizer(
                max_features=20000, ngram_range=(1, 2), stop_words="english"
            ),
        ),
    ]
)

handcrafted = Pipeline([("hc", HandcraftedTransformer()), ("scale", StandardScaler())])

features = FeatureUnion([("tfidf", tfidf), ("handcrafted", handcrafted)])

# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = (
    train_test_split(
        df,
        df["problem_class"],
        df["problem_score"],
        test_size=0.2,
        random_state=42,
        stratify=df["problem_class"],
    )
)

# -----------------------------
# Classification model
# -----------------------------
clf = Pipeline(
    [
        ("features", features),
        (
            "model",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=25,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

clf.fit(X_train, y_class_train)
y_pred_class = clf.predict(X_test)

print("Classification Accuracy:", accuracy_score(y_class_test, y_pred_class))

# -----------------------------
# Regression model
# -----------------------------
reg = Pipeline(
    [
        ("features", features),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ]
)

reg.fit(X_train, y_reg_train)
y_pred_reg = reg.predict(X_test)

mae = mean_absolute_error(y_reg_test, y_pred_reg)
rmse = math.sqrt(mean_squared_error(y_reg_test, y_pred_reg))

print(f"Regression MAE: {mae:.3f}")
print(f"Regression RMSE: {rmse:.3f}")

# -----------------------------
# Save models
# -----------------------------
joblib.dump(clf, "models/autojudge_classifier.joblib")
joblib.dump(reg, "models/autojudge_regressor.joblib")

print("Models saved successfully.")
