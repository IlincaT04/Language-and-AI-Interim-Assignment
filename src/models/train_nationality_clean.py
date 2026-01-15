import os
import re
import joblib

# ML / NLP imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# data loading function
from src.data.load_data import load_nationality


# Build the nationality model with clean data
def build_nationality_model(max_features: int = 20000) -> Pipeline:
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                max_features=max_features,
                ngram_range=(1, 2),     
            ),
        ),
        (
            "clf",
            LogisticRegression(
                solver="saga",         
                max_iter=1000,          
                verbose=1               
            ),
        ),
    ])

# Build a rule that finds country names in the text
def build_country_mask_regex(countries: list[str]) -> re.Pattern:
    # Clean the list
    countries = sorted(
        {c.strip() for c in countries if isinstance(c, str) and c.strip()},
        key=len,
        reverse=True  # longest names first to avoid partial matches
    )

    # Escape special characters
    escaped = [re.escape(c) for c in countries]

    # OR rule
    pattern = r"\b(" + "|".join(escaped) + r")\b"

    # Make it case-insensitive
    return re.compile(pattern, flags=re.IGNORECASE)

# Mask country names in a post
def mask_countries(text: str, country_re: re.Pattern) -> str:
    return country_re.sub("<COUNTRY>", str(text))


# Main training script
def main():

    # Load the nationality dataset
    df = load_nationality("src/data/nationality.csv")

    # Get all nationality labels
    countries = df["nationality"].astype(str).unique().tolist()

    # Build the tue for finding country names
    country_re = build_country_mask_regex(countries)

    # Clean the text by masking country names
    X = (
        df["post"]
        .astype(str)
        .apply(lambda t: mask_countries(t, country_re))
        .tolist()
    )

    y = df["nationality"].astype(str).tolist()

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(set(y)) > 1 else None
    )


    # Build and train the model
    model = build_nationality_model(max_features=20000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("=== Nationality N1 (clean: country names masked) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # Save trained model
    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(model, "outputs/models/nationality_N1_clean.joblib")


if __name__ == "__main__":
    main()