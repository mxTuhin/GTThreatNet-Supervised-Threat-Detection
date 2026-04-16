import pandas as pd
import joblib

MODEL_PATH = "data/derived/following_rf.pkl"
INPUT_CSV = "data/derived/pair_windows_for_labeling.csv"
OUTPUT_CSV = "data/derived/pair_windows_scored.csv"

FEATURES = [
    "num_points",
    "direction_similarity",
    "avg_distance",
    "distance_variance",
    "speed_similarity",
    "behind_ratio"
]


def main():
    df = pd.read_csv(INPUT_CSV)
    model = joblib.load(MODEL_PATH)

    X = df[FEATURES]
    df["pred_following"] = model.predict(X)

    if hasattr(model, "predict_proba"):
        df["pred_score"] = model.predict_proba(X)[:, 1]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()