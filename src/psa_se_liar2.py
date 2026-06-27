"""Pre-Split Aggregation WITH self-exclusion (PSA+SE) for LIAR2.

Reads the original (global / pre-split) LIAR2 counts from ../data/liar2/ and only
removes the current row's own label from its history vector (leave-one-out),
keeping the global aggregation scope unchanged. This is the missing cell of the
2x2 {scope: global / train-only} x {self-exclusion: no / yes} design: it isolates
the global-scope leakage from the self-inclusion leakage.

  PSA      = global scope, self-inclusion   (../data/liar2/, as released)
  PSA+SE   = global scope, self-exclusion   (this script)
  LFA      = train-only scope, self-exclusion

Writes to ../data/liar2_psa_se/ (same filenames, same schema).
"""

import os
import pandas as pd

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(ROOT, "data", "liar2")
OUT_DIR = os.path.join(ROOT, "data", "liar2_psa_se")
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_TO_COL = {
    0: "pants_on_fire_counts",
    1: "false_counts",
    2: "mostly_false_counts",
    3: "half_true_counts",
    4: "mostly_true_counts",
    5: "true_counts",
}


def apply_self_exclusion(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].astype(int)
    # global counts are kept as released; only subtract the row's own label
    for label_int, col_name in LABEL_TO_COL.items():
        mask = df["label"] == label_int
        df.loc[mask, col_name] = (df.loc[mask, col_name] - 1).clip(lower=0)
    return df


for split in ("train", "valid", "test"):
    df_in  = pd.read_csv(os.path.join(IN_DIR, f"{split}.csv"))
    df_out = apply_self_exclusion(df_in)
    df_out.to_csv(os.path.join(OUT_DIR, f"{split}.csv"), index=False)
    print(f"{split}: {len(df_out)} rows -> {OUT_DIR}/{split}.csv")
