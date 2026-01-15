import os
import joblib
import pandas as pd

from src.data.load_data import load_nationality


def main():

    # Load nationality dataset
    df = load_nationality("src/data/nationality.csv")

    # Load trained political model 
    model = joblib.load("outputs/models/political_P_clean.joblib")

    # Predict political leaning for each post
    posts = df["post"].astype(str).tolist()
    preds = []

    for i in range(0, len(posts), 5000):
        batch = posts[i:i + 5000]
        batch_pred = model.predict(batch)
        preds.extend(batch_pred)

    # Add prediction column
    df["pred_political_leaning"] = preds

    # Save the new dataset
    os.makedirs("outputs/data", exist_ok=True)
    df.to_csv("outputs/data/nationality_with_pred_political.csv", index=False)

    print("Done.")
    print(f"Saved: outputs/data/nationality_with_pred_political.csv")
    print("Column added: pred_political_leaning")
    print(df["pred_political_leaning"].value_counts())


if __name__ == "__main__":
    main()
