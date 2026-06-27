"""
Leak-free COUNT_COLS for LIAR2.

Reads train/valid/test from  ../liar2/
Writes corrected splits to   ../liar2loo/  (same filenames, same schema)

Algorithm
---------
TRAIN  : aggregate per-speaker label counts from train labels only,
         then apply leave-one-out — each row's own label is subtracted
         so the model never sees its own contribution as history.

VALID  : look up speaker in train-derived counts.
TEST     Speakers absent from train receive a zero vector (cold-start).
         No val/test label ever enters any count column.
"""

import os
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(ROOT, "data", "liar2")
OUT_DIR = os.path.join(ROOT, "data", "liar2_lfa")
os.makedirs(OUT_DIR, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
SPEAKER_COL = "speaker"

COUNT_COLS = [
    "pants_on_fire_counts",
    "false_counts",
    "mostly_false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "true_counts",
]

# integer label → which count column it increments
LABEL_TO_COL = {
    0: "pants_on_fire_counts",
    1: "false_counts",
    2: "mostly_false_counts",
    3: "half_true_counts",
    4: "mostly_true_counts",
    5: "true_counts",
}


def normalize_speaker(s: pd.Series) -> pd.Series:
    return s.fillna("UNKNOWN").astype(str).str.strip()


def read_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(IN_DIR, f"{name}.csv"))
    df[SPEAKER_COL] = normalize_speaker(df[SPEAKER_COL])
    df["label"] = df["label"].astype(int)
    return df


# ── step 1: aggregate per-speaker counts from TRAIN only ──────────────────────
train = read_split("train")

train_counts = (
    train.groupby([SPEAKER_COL, "label"])
         .size()
         .unstack(fill_value=0)
         .reindex(columns=sorted(LABEL_TO_COL.keys()), fill_value=0)
)
train_counts.columns = [LABEL_TO_COL[c] for c in train_counts.columns]


# ── step 2: apply to each split ───────────────────────────────────────────────
def apply_leak_free(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()

    # zero out all original count columns
    for col in COUNT_COLS:
        df[col] = 0

    # lookup from train-derived counts (cold-start speakers stay at 0)
    for col in COUNT_COLS:
        if col in train_counts.columns:
            df[col] = df[SPEAKER_COL].map(train_counts[col]).fillna(0).astype(int)

    # leave-one-out: train rows must not see their own label in their history
    if is_train:
        for label_int, col_name in LABEL_TO_COL.items():
            mask = df["label"] == label_int
            df.loc[mask, col_name] = (df.loc[mask, col_name] - 1).clip(lower=0)

    return df


for split, is_train in [("train", True), ("valid", False), ("test", False)]:
    df_in  = read_split(split)
    df_out = apply_leak_free(df_in, is_train)
    df_out.to_csv(os.path.join(OUT_DIR, f"{split}.csv"), index=False)
    print(f"{split}: {len(df_out)} rows → {OUT_DIR}/{split}.csv")
