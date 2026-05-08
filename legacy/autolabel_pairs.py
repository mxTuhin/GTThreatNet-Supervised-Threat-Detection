"""
autolabel_pairs.py

Auto-labels pair_windows_for_labeling.csv using the same heuristic thresholds
as following_logic.py. Outputs labeled_pairs.csv ready for train_following_baseline.py.

This is a valid research baseline approach: use expert-defined rules to bootstrap
labels, then train a model that learns a generalized decision boundary.

You can manually review and correct labels in labeled_pairs.csv afterwards.
"""

import pandas as pd
import os

INPUT_CSV  = "data/derived/pair_windows_for_labeling.csv"
OUTPUT_CSV = "data/derived/labeled_pairs.csv"

# Same thresholds as following_logic.py
MIN_DIRECTION_SIM  = 0.75
MIN_BEHIND_RATIO   = 0.60
MIN_AVG_DISTANCE   = 30.0
MAX_AVG_DISTANCE   = 180.0
MIN_SPEED_SIM      = 0.40   # new: follower should move at similar speed to leader


def label_row(row):
    if row["direction_similarity"] < MIN_DIRECTION_SIM:
        return 0
    if row["behind_ratio"] < MIN_BEHIND_RATIO:
        return 0
    if not (MIN_AVG_DISTANCE <= row["avg_distance"] <= MAX_AVG_DISTANCE):
        return 0
    if row["speed_similarity"] < MIN_SPEED_SIM:
        return 0
    return 1


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Not found: {INPUT_CSV}")
        print("Run mot17_pair_extractor.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    df["following_label"] = df.apply(label_row, axis=1)

    n_total    = len(df)
    n_following = df["following_label"].sum()
    n_not       = n_total - n_following

    print(f"Total pair windows : {n_total}")
    print(f"Following (1)      : {n_following}  ({100*n_following/n_total:.1f}%)")
    print(f"Not following (0)  : {n_not}  ({100*n_not/n_total:.1f}%)")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print("\nYou can open labeled_pairs.csv and manually correct any rows before training.")


if __name__ == "__main__":
    main()
