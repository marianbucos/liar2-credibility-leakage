"""
Leak-free COUNT_COLS for all mix/ experiment files.

Reads from  ../mix/
Writes to   ../mix_lfa/  (same filenames, same schema)

Algorithm
---------
Grouped experiments (1.2, 1.3, 3.1, 3.2, 4.1):
    TRAIN  : aggregate per-speaker label counts from train only, then apply
             leave-one-out — each row's own label is subtracted so the model
             never sees its own contribution as history.
    TEST   : look up speaker in train-derived counts.
    VAL      Speakers absent from train receive a zero vector (cold-start).
             No test/val label ever enters any count column.

Standalone files (LIAR2023_pt1_origin, LIAR2023_pt2_new, used as unsplit
whole-datasets in experiments 2.1 and 2.2):
    Each file is treated as its own training set and receives within-file
    leave-one-out correction.
"""

import os
import pandas as pd

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(ROOT, "data", 'mix')
OUT_DIR = os.path.join(ROOT, "data", 'mix_lfa')
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

# Experiments with explicit train / test / val files
GROUPS = [
    ('Origin_8_1_1_train.csv',          'Origin_8_1_1_test.csv',          'Origin_8_1_1_val.csv'),
    ('New_8_1_1_train.csv',             'New_8_1_1_test.csv',             'New_8_1_1_val.csv'),
    ('Train_origin_test_new_train.csv', 'Train_origin_test_new_test.csv', 'Train_origin_test_new_val.csv'),
    ('Train_new_test_origin_train.csv', 'Train_new_test_origin_test.csv', 'Train_new_test_origin_val.csv'),
    ('Origin_new_mix_8_1_1_train.csv',  'Origin_new_mix_8_1_1_test.csv',  'Origin_new_mix_8_1_1_val.csv'),
]

# Whole-dataset files (no paired test/val; each receives within-file LFA)
STANDALONE = [
    'LIAR2023_pt1_origin.csv',
    'LIAR2023_pt2_new.csv',
]


def read_csv(filename):
    df = pd.read_csv(os.path.join(IN_DIR, filename))
    df[SPEAKER_COL] = df[SPEAKER_COL].fillna('UNKNOWN').astype(str).str.strip()
    df['label'] = df['label'].astype(int)
    for col in COUNT_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df


def write_csv(df, filename):
    df.to_csv(os.path.join(OUT_DIR, filename), index=False)


def build_train_counts(train_df):
    counts = (
        train_df.groupby([SPEAKER_COL, 'label'])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=sorted(LABEL_TO_COL.keys()), fill_value=0)
    )
    counts.columns = [LABEL_TO_COL[c] for c in counts.columns]
    return counts


def apply_lfa_train(df, train_counts):
    df = df.copy()
    for col in COUNT_COLS:
        df[col] = 0
    for col in train_counts.columns:
        df[col] = df[SPEAKER_COL].map(train_counts[col]).fillna(0).astype(int)
    for label_int, col_name in LABEL_TO_COL.items():
        mask = df['label'] == label_int
        df.loc[mask, col_name] = (df.loc[mask, col_name] - 1).clip(lower=0)
    return df


def apply_lfakup(df, train_counts):
    df = df.copy()
    for col in COUNT_COLS:
        df[col] = 0
    for col in train_counts.columns:
        df[col] = df[SPEAKER_COL].map(train_counts[col]).fillna(0).astype(int)
    return df


# ── grouped experiments ────────────────────────────────────────────────────────
for train_f, test_f, val_f in GROUPS:
    train = read_csv(train_f)
    test  = read_csv(test_f)
    val   = read_csv(val_f)

    counts = build_train_counts(train)

    write_csv(apply_lfa_train(train, counts), train_f)
    write_csv(apply_lfakup(test, counts),     test_f)
    write_csv(apply_lfakup(val, counts),      val_f)

    print(f'{train_f[:-10]}: train={len(train)}, test={len(test)}, val={len(val)}')

# ── standalone whole-dataset files ────────────────────────────────────────────
for filename in STANDALONE:
    df = read_csv(filename)
    counts = build_train_counts(df)
    write_csv(apply_lfa_train(df, counts), filename)
    print(f'{filename}: {len(df)} rows (within-file LFA)')
