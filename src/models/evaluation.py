import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


SEED = 42
TEST_SIZE = 0.2

# Paths to your saved models
MODEL_N1 = "outputs/models/nationality_N1.joblib"
MODEL_N1_CLEAN = "outputs/models/nationality_N1_clean.joblib"
MODEL_N2 = "outputs/models/nationality_N2.joblib"
MODEL_N2_CLEAN = "outputs/models/nationality_N2_clean.joblib"

DATA_NAT = "src/data/nationality.csv"
DATA_NAT_WITH_POL = "outputs/data/nationality_with_pred_political.csv"

OUT_CSV = "outputs/results/nationality_results.csv"


# Cleaning utilities (same as training)
def build_country_mask_regex(countries: list[str]) -> re.Pattern:
    countries = sorted(
        {c.strip() for c in countries if isinstance(c, str) and c.strip()},
        key=len,
        reverse=True,
    )
    escaped = [re.escape(c) for c in countries]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


def mask_countries_series(posts: pd.Series, countries: list[str]) -> pd.Series:
    country_re = build_country_mask_regex(countries)
    return posts.fillna("").astype(str).apply(lambda t: country_re.sub("<COUNTRY>", t))


# Eval helpers
def eval_model_text_only(model_path: str, df: pd.DataFrame, label_col: str, text_col: str, model_name: str):
    X = df[text_col].fillna("").astype(str).tolist()
    y = df[label_col].fillna("").astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y if len(set(y)) > 1 else None,
    )

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    return {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
        "n_test": len(y_test),
    }


def eval_model_text_plus_pol(model_path: str, df: pd.DataFrame, model_name: str):
    # Model expects both columns
    X = df[["post", "pred_political_leaning"]].copy()
    y = df["nationality"].fillna("").astype(str)

    # clean NaNs
    X["post"] = X["post"].fillna("").astype(str)
    X["pred_political_leaning"] = X["pred_political_leaning"].fillna("unknown").astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y if y.nunique() > 1 else None,
    )

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    return {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
        "n_test": len(y_test),
    }


def main():
    rows = []

    # Load base nationality dataset
    df_nat = pd.read_csv(DATA_NAT)
    if "auhtor_ID" in df_nat.columns and "author_ID" not in df_nat.columns:
        df_nat = df_nat.rename(columns={"auhtor_ID": "author_ID"})
    if not {"post", "nationality"}.issubset(df_nat.columns):
        raise ValueError(f"{DATA_NAT} must contain columns: post, nationality")

    # For clean models we need the country list
    countries = df_nat["nationality"].fillna("").astype(str).unique().tolist()

    # Evaluate N1 polluted
    if os.path.exists(MODEL_N1):
        rows.append(eval_model_text_only(MODEL_N1, df_nat, "nationality", "post", "N1 (polluted, text-only)"))

    # Evaluate N1 clean (mask country names before split/predict)

    if os.path.exists(MODEL_N1_CLEAN):
        df_nat_clean = df_nat.copy()
        df_nat_clean["post"] = mask_countries_series(df_nat_clean["post"], countries)
        rows.append(eval_model_text_only(MODEL_N1_CLEAN, df_nat_clean, "nationality", "post", "N1-clean (masked countries)"))

    # Load augmented dataset for N2
    if os.path.exists(DATA_NAT_WITH_POL):
        df_n2 = pd.read_csv(DATA_NAT_WITH_POL)
        required = {"post", "nationality", "pred_political_leaning"}
        if not required.issubset(df_n2.columns):
            raise ValueError(f"{DATA_NAT_WITH_POL} missing columns: {required - set(df_n2.columns)}")

        if os.path.exists(MODEL_N2):
            rows.append(eval_model_text_plus_pol(MODEL_N2, df_n2, "N2 (polluted, text + pred_pol)"))

        if os.path.exists(MODEL_N2_CLEAN):
            df_n2_clean = df_n2.copy()
            df_n2_clean["post"] = mask_countries_series(df_n2_clean["post"], countries)
            rows.append(eval_model_text_plus_pol(MODEL_N2_CLEAN, df_n2_clean, "N2-clean (masked countries + pred_pol)"))

    # Output table
    results = pd.DataFrame(rows)
    if results.empty:
        print("No models found to evaluate. Train and save models first.")
        return

    # Pretty print
    print("\n=== Nationality model summary ===")
    print(results.to_string(index=False))

    # Save CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    print(f"\nSaved results table to: {OUT_CSV}")


if __name__ == "__main__":
    main()