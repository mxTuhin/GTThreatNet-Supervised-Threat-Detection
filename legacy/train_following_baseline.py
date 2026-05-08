import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

INPUT_CSV = "data/derived/labeled_pairs.csv"

FEATURES = [
    "num_points",
    "direction_similarity",
    "avg_distance",
    "distance_variance",
    "speed_similarity",
    "behind_ratio"
]

TARGET = "following_label"


def main():
    df = pd.read_csv(INPUT_CSV)

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    importances = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nFeature Importances:")
    print(importances)

    import joblib
    joblib.dump(model, "data/derived/following_rf.pkl")
    print("\nSaved model to data/derived/following_rf.pkl")


if __name__ == "__main__":
    main()