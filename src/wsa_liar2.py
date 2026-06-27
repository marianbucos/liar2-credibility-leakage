"""
Within-Split Aggregation (WSA / L2) for LIAR2.

Reads train/valid/test from  ../liar2/
Writes recomputed splits to  ../liar2_wsa/ (same filenames, same schema)

Algorithm
---------
For each split independently, count how many times each speaker was assigned
each label within that split, then assign those counts back to every row of
that speaker — with leave-one-out correction: each row's own label is
subtracted so it does not appear in its own history vector (consistent with
the original LIAR2 construction, where the current statement is excluded).

Leakage note: val/test rows still see counts derived from other val/test rows'
labels within the same split. This is the residual L2 leakage that LFA
eliminates by using only train-derived counts for val/test.
"""

import os
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(ROOT, "data", "liar2")
OUT_DIR = os.path.join(ROOT, "data", "liar2_wsa")
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

LABEL_TO_COL = {
    0: "pants_on_fire_counts",
    1: "false_counts",
    2: "mostly_false_counts",
    3: "half_true_counts",
    4: "mostly_true_counts",
    5: "true_counts",
}


def apply_split_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[SPEAKER_COL] = df[SPEAKER_COL].fillna("UNKNOWN").astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    per_speaker = (
        df.groupby([SPEAKER_COL, "label"])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=sorted(LABEL_TO_COL.keys()), fill_value=0)
    )
    per_speaker.columns = [LABEL_TO_COL[c] for c in per_speaker.columns]

    for col in COUNT_COLS:
        df[col] = 0

    for col in per_speaker.columns:
        df[col] = df[SPEAKER_COL].map(per_speaker[col]).fillna(0).astype(int)

    # leave-one-out: each row must not see its own label in its history vector
    for label_int, col_name in LABEL_TO_COL.items():
        mask = df["label"] == label_int
        df.loc[mask, col_name] = (df.loc[mask, col_name] - 1).clip(lower=0)

    return df


for split in ("train", "valid", "test"):
    df_in  = pd.read_csv(os.path.join(IN_DIR, f"{split}.csv"))
    df_out = apply_split_count(df_in)
    df_out.to_csv(os.path.join(OUT_DIR, f"{split}.csv"), index=False)
    print(f"{split}: {len(df_out)} rows → {OUT_DIR}/{split}.csv")
