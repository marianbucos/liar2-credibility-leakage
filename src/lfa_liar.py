"""
Leak-free COUNT_COLS for the original LIAR dataset.

Reads train/valid/test from  ../liar/
Writes corrected splits to   ../liar_lfa/  (same filenames, same schema)

Algorithm
---------
TRAIN  : aggregate per-speaker label counts from train labels only,
         then apply leave-one-out — each row's own label is subtracted
         so the model never sees its own contribution as history.

VALID  : look up speaker in train-derived counts.
TEST     Speakers absent from train receive a zero vector (cold-start).
         No val/test label ever enters any count column.

Note: the original LIAR dataset has no 'true_counts' column — the 'true'
label therefore contributes nothing to any count column.
"""

import os
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(ROOT, "data", "liar")
OUT_DIR = os.path.join(ROOT, "data", "liar_lfa")
os.makedirs(OUT_DIR, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
COLS = [
    'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
    'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts',
    'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context',
]

SPEAKER_COL = 'speaker'

COUNT_COLS = [
    'barely_true_counts',
    'false_counts',
    'half_true_counts',
    'mostly_true_counts',
    'pants_on_fire_counts',
]

LABEL_CONVERT = {
    'pants-fire':  0,
    'false':       1,
    'barely-true': 2,
    'half-true':   3,
    'mostly-true': 4,
    'true':        5,
}

# 'true' (5) has no corresponding count column in the original LIAR dataset
LABEL_TO_COL = {
    0: 'pants_on_fire_counts',
    1: 'false_counts',
    2: 'barely_true_counts',
    3: 'half_true_counts',
    4: 'mostly_true_counts',
}


def read_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(IN_DIR, f"{name}.tsv"), sep='\t', names=COLS)
    df[SPEAKER_COL] = df[SPEAKER_COL].fillna('UNKNOWN').astype(str).str.strip()
    df['label_int'] = df['label'].map(LABEL_CONVERT)
    for col in COUNT_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df


def write_split(df: pd.DataFrame, name: str) -> None:
    out = df[COLS].copy()
    out.to_csv(os.path.join(OUT_DIR, f"{name}.tsv"), sep='\t', index=False, header=False)


# ── step 1: aggregate per-speaker counts from TRAIN only ──────────────────────
train = read_split('train')

train_counts = (
    train.groupby([SPEAKER_COL, 'label_int'])
         .size()
         .unstack(fill_value=0)
         .reindex(columns=sorted(LABEL_TO_COL.keys()), fill_value=0)
)
train_counts.columns = [LABEL_TO_COL[c] for c in train_counts.columns]


# ── step 2: apply to each split ───────────────────────────────────────────────
def apply_leak_free(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()

    for col in COUNT_COLS:
        df[col] = 0

    for col in train_counts.columns:
        df[col] = df[SPEAKER_COL].map(train_counts[col]).fillna(0).astype(int)

    if is_train:
        for label_int, col_name in LABEL_TO_COL.items():
            mask = df['label_int'] == label_int
            df.loc[mask, col_name] = (df.loc[mask, col_name] - 1).clip(lower=0)

    return df


for split, is_train in [('train', True), ('valid', False), ('test', False)]:
    df_in  = read_split(split)
    df_out = apply_leak_free(df_in, is_train)
    write_split(df_out, split)
    print(f"{split}: {len(df_out)} rows -> {OUT_DIR}/{split}.tsv")
