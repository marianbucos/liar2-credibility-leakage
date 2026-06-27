"""
Within-Split Aggregation (WSA / L2) for all mix/ experiment files.

Reads from  ../mix/
Writes to   ../mix_wsa/  (same filenames, same schema)

Algorithm
---------
Each file is processed independently: for every speaker in the file, count
how many times they were assigned each label within that file, then assign
those counts back to every row belonging to that speaker — with leave-one-out
correction: each row's own label is subtracted so it does not appear in its
own history vector (consistent with the original LIAR2 construction, where the
current statement is excluded).

Applies uniformly to both grouped split files and whole-dataset files.

Leakage note: for val/test files, rows still see counts derived from other
rows' labels within the same file. This is the residual L2 leakage that LFA
eliminates by using only train-derived counts for val/test.
"""

import os
import pandas as pd

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(ROOT, "data", 'mix')
OUT_DIR = os.path.join(ROOT, "data", 'mix_wsa')
os.makedirs(OUT_DIR, exist_ok=True)

SPEAKER_COL = 'speaker'

COUNT_COLS = [
    'true_counts',
    'mostly_true_counts',
    'half_true_counts',
    'mostly_false_counts',
    'false_counts',
    'pants_on_fire_counts',
]

LABEL_TO_COL = {
    0: 'pants_on_fire_counts',
    1: 'false_counts',
    2: 'mostly_false_counts',
    3: 'half_true_counts',
    4: 'mostly_true_counts',
    5: 'true_counts',
}


def apply_split_count(df):
    df = df.copy()
    df[SPEAKER_COL] = df[SPEAKER_COL].fillna('UNKNOWN').astype(str).str.strip()
    df['label'] = df['label'].astype(int)
    for col in COUNT_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    per_speaker = (
        df.groupby([SPEAKER_COL, 'label'])
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
        mask = df['label'] == label_int
        df.loc[mask, col_name] = (df.loc[mask, col_name] - 1).clip(lower=0)

    return df


for filename in sorted(os.listdir(IN_DIR)):
    if not filename.endswith('.csv'):
        continue
    df_in  = pd.read_csv(os.path.join(IN_DIR, filename))
    df_out = apply_split_count(df_in)
    df_out.to_csv(os.path.join(OUT_DIR, filename), index=False)
    print(f'{filename}: {len(df_out)} rows')
