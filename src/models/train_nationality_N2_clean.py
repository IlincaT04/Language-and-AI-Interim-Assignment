import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


SEED = 42
TEST_SIZE = 0.2
MAX_FEATURES = 20000

DATA_NAT = "src/data/nationality.csv"  # for the country label list
DATA_N2 = "outputs/data/nationality_with_pred_political.csv"
MODEL_OUT = "outputs/models/nationality_N2_clean.joblib"

# Cleaning (same idea as N1-clean)
def build_country_mask_regex(countries: list[str]) -> re.Pattern:
    countries = sorted(
        {c.strip() for c in countries if isinstance(c, str) and c.strip()},
        key=len,
        reverse=True,
    )
    escaped = [re.escape(c) for c in countries]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


def mask_countries(text: str, country_re: re.Pattern) -> str:
    return country_re.sub("<COUNTRY>", str(text))


# Model N2-clean: Nationality prediction using masked posts + predicted political leaning
def build_n2_model(max_features: int = MAX_FEATURES) -> Pipeline:
    preproc = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(lowercase=True, max_features=max_features, ngram_range=(1, 2)), "post"),
            ("pol", OneHotEncoder(handle_unknown="ignore"), ["pred_political_leaning"]),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("features", preproc),
        ("clf", LogisticRegression(solver="saga", max_iter=1000, verbose=1)),
    ])


def main():
    # Load base nationality labels to build country list
    df_nat = pd.read_csv(DATA_NAT)
    if "auhtor_ID" in df_nat.columns and "author_ID" not in df_nat.columns:
        df_nat = df_nat.rename(columns={"auhtor_ID": "author_ID"})

    if "nationality" not in df_nat.columns:
        raise ValueError(f"{DATA_NAT} must contain a 'nationality' column")

    countries = df_nat["nationality"].fillna("").astype(str).unique().tolist()
    country_re = build_country_mask_regex(countries)

    # Load N2 dataset (already contains pred_political_leaning)
    df = pd.read_csv(DATA_N2)

    required = {"post", "nationality", "pred_political_leaning"}
    if not required.issubset(df.columns):
        raise ValueError(f"{DATA_N2} missing columns: {required - set(df.columns)}")

    # Basic cleaning
    df["post"] = df["post"].fillna("").astype(str)
    df["nationality"] = df["nationality"].fillna("").astype(str)
    df["pred_political_leaning"] = df["pred_political_leaning"].fillna("unknown").astype(str)

    # N2-clean: mask country names in the post text
    df["post"] = df["post"].apply(lambda t: mask_countries(t, country_re))

    X = df[["post", "pred_political_leaning"]]
    y = df["nationality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y if y.nunique() > 1 else None,
    )

    model = build_n2_model(max_features=MAX_FEATURES)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("=== Nationality N2-clean (masked countries + predicted political leaning) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nSaved model to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
