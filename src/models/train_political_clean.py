import os
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data.load_data import load_political


# Build the political model
def build_political_model(max_features: int = 20000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            max_features=max_features,
            ngram_range=(1, 2),
        )),
        ("clf", LogisticRegression(
            solver="saga",
            max_iter=1000,
            verbose=1
        )),
    ])


# Build a rule that finds leaky political words
def build_political_mask_regex() -> re.Pattern:

    leaky_terms = [
        "left", "right", "center", "centre",
        "liberal", "conservative", "progressive",
        "democrat", "republican",
    ]

    # Make sure longer words match first
    leaky_terms = sorted(set(leaky_terms), key=len, reverse=True)

    escaped = [re.escape(t) for t in leaky_terms]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


def mask_political_terms(text: str, pol_re: re.Pattern) -> str:
    return pol_re.sub("<POL>", str(text))


def main():

    # Load dataset (expects: author_ID, post, political_leaning)
    df = load_political("src/data/political_leaning.csv")

    # Build the masking rule (our "cleaning" step)
    pol_re = build_political_mask_regex()

    # Clean the text by masking obvious label words
    X = (
        df["post"]
        .astype(str)
        .apply(lambda t: mask_political_terms(t, pol_re))
        .tolist()
    )

    # Labels (should be left/center/right)
    y = df["political_leaning"].astype(str).tolist()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(set(y)) > 1 else None
    )


    model = build_political_model(max_features=20000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("=== Political Leaning P (clean: label words masked) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(model, "outputs/models/political_P_clean.joblib")


if __name__ == "__main__":
    main()
