import pandas as pd

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "auhtor_ID" in df.columns and "author_ID" not in df.columns:
        df = df.rename(columns={"auhtor_ID": "author_ID"})
    return df

def load_political(political_csv: str) -> pd.DataFrame:
    df = _normalize_cols(pd.read_csv(political_csv))

    required = {"author_ID", "post", "political_leaning"}
    if not required.issubset(df.columns):
        raise ValueError(f"political_leaning.csv missing columns: {required - set(df.columns)}")

    df["post"] = df["post"].astype(str).fillna("")
    df["political_leaning"] = df["political_leaning"].astype(str).str.lower().str.strip()

    df = df[df["political_leaning"].isin({"left", "center", "right"})].copy()
    df = df[df["post"].str.len() > 0].copy()
    return df

def load_nationality(nationality_csv: str) -> pd.DataFrame:
    df = _normalize_cols(pd.read_csv(nationality_csv))

    required = {"author_ID", "post", "nationality"}
    if not required.issubset(df.columns):
        raise ValueError(f"nationality.csv missing columns: {required - set(df.columns)}")

    df["post"] = df["post"].astype(str).fillna("")
    df["nationality"] = df["nationality"].astype(str).str.strip()

    df = df[df["post"].str.len() > 0].copy()
    return df

