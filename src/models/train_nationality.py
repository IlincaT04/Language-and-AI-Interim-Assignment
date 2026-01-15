import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data.load_data import load_nationality

# Model N1: Nationality prediction using posts only - polluted version
def build_nationality_model(max_features: int = 20000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            max_features=max_features,
            ngram_range=(1, 2),
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="saga",
            verbose=1
        )),
    ])


def main():

    df = load_nationality("src/data/nationality.csv")


    # X = text, y = label
    X = df["post"].astype(str).tolist()
    y = df["nationality"].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(set(y)) > 1 else None
    )


    model = build_nationality_model(max_features=20000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("=== Nationality N1 (text-only) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=4))

    # Save model
    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(model, "outputs/models/nationality_N1.joblib")



if __name__ == "__main__":
    main()
