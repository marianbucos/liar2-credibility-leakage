"""
Within-Split Aggregation WITHOUT self-exclusion — counterfactual baseline.

This script exists for one purpose: to empirically measure the impact of the
self-inclusion leakage component (§8.1 of the article plan). The construction
is identical to wsa_liar2.py *except* it skips the self-exclusion step, so
each row's count vector contains its own label's contribution.

Reads train/valid/test from  ../liar2/
Writes recomputed splits to  ../liar2_wsa_no_self_exclusion/

This is NOT a production construction — it is the maximally leaky within-split
baseline, used only to quantify the gap closed by self-exclusion.
"""

import os
import pandas as pd

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(ROOT, "data", "liar2")
OUT_DIR = os.path.join(ROOT, "data", "liar2_wsa_no_self_exclusion")
os.makedirs(OUT_DIR, exist_ok=True)

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


def apply_split_count_no_self_exclusion(df: pd.DataFrame) -> pd.DataFrame:
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

    # NOTE: deliberately omitted — self-exclusion step that wsa_liar2.py applies.
    # Each row's own label REMAINS in its count vector. This is the leaky baseline.
    return df


for split in ("train", "valid", "test"):
    df_in  = pd.read_csv(os.path.join(IN_DIR, f"{split}.csv"))
    df_out = apply_split_count_no_self_exclusion(df_in)
    df_out.to_csv(os.path.join(OUT_DIR, f"{split}.csv"), index=False)
    print(f"{split}: {len(df_out)} rows → {OUT_DIR}/{split}.csv")
