import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Model N2: Nationality prediction using text + predicted political leaning
def build_n2_model(max_features: int = 20000) -> Pipeline:

    preproc = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(lowercase=True, max_features=max_features, ngram_range=(1, 2)), "post"),
            ("pol", OneHotEncoder(handle_unknown="ignore"), ["pred_political_leaning"]),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("features", preproc),
        ("clf", LogisticRegression(
            solver="saga",
            max_iter=1000,
            verbose=1
        )),
    ])


def main():

    df = pd.read_csv("outputs/data/nationality_with_pred_political.csv")

    required = {"post", "nationality", "pred_political_leaning"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in outputs/data/nationality_with_pred_political.csv: {required - set(df.columns)}")
    # Basic cleaning
    df["post"] = df["post"].astype(str).fillna("")
    df["nationality"] = df["nationality"].astype(str).fillna("")
    df["pred_political_leaning"] = df["pred_political_leaning"].astype(str).fillna("unknown")

    # Split
    X = df[["post", "pred_political_leaning"]]
    y = df["nationality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    model = build_n2_model(max_features=20000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("=== Nationality N2 (raw text + predicted political leaning) --- POLLUTED ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(model, "outputs/models/nationality_N2.joblib")


if __name__ == "__main__":
    main()
