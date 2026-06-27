"""
Generates the segmentation-experiment CSV files in ../mix/.

LIAR2 = LIAR + NEW, where LIAR records are identified by matching integer IDs
with the original LIAR dataset (liar/*.tsv). All files follow the 16-column
LIAR2 schema and are read by scripts 1.2 – 4.1.

Files produced
--------------
1.2  Origin_8_1_1_{train,test,val}.csv      LIAR part, independent 8:1:1 split
1.3  New_8_1_1_{train,test,val}.csv         NEW part, independent 8:1:1 split
2.1  LIAR2023_pt1_origin.csv                all LIAR records (unsplit)
2.2  LIAR2023_pt2_new.csv                   all NEW records  (unsplit)
3.1  Train_origin_test_new_{train,test,val} LIAR(1.)+NEW(.557) / NEW(.443)
3.2  Train_new_test_origin_{train,test,val} NEW(1.)+LIAR(.635) / LIAR(.365)
4.1  Origin_new_mix_8_1_1_{train,test,val}  concat(1.2, 1.3) = independent split

Note: 1.2, 1.3 and 4.1 use an INDEPENDENT per-component 8:1:1 partition (each
of LIAR and NEW is split on its own, then merged for 4.1, so 4.1 = 1.2 + 1.3).
This is deliberately distinct from the official LIAR2 split used in 5.1; the two
share row membership only by coincidence, not by construction.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / 'data' / 'mix'
OUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# ── identify LIAR records inside LIAR2 via integer ID ─────────────────────────
LIAR_COLS = [
    'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
    'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts',
    'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context',
]
liar_raw = pd.concat([
    pd.read_csv(ROOT / 'data' / 'liar' / f, sep='\t', names=LIAR_COLS)
    for f in ('train.tsv', 'test.tsv', 'valid.tsv')
], ignore_index=True)
liar_id_set = set(liar_raw['id'].str.replace('.json', '', regex=False).astype(int))

# ── load LIAR2 splits ──────────────────────────────────────────────────────────
tr2 = pd.read_csv(ROOT / 'data' / 'liar2' / 'train.csv')
te2 = pd.read_csv(ROOT / 'data' / 'liar2' / 'test.csv')
va2 = pd.read_csv(ROOT / 'data' / 'liar2' / 'valid.csv')

for df in (tr2, te2, va2):
    df['_is_liar'] = df['id'].isin(liar_id_set)

# ── separate LIAR / NEW within each split ─────────────────────────────────────
def split_liar_new(df):
    liar = df[df['_is_liar']].drop(columns=['_is_liar']).reset_index(drop=True)
    new  = df[~df['_is_liar']].drop(columns=['_is_liar']).reset_index(drop=True)
    return liar, new

liar_tr, new_tr = split_liar_new(tr2)
liar_te, new_te = split_liar_new(te2)
liar_va, new_va = split_liar_new(va2)

liar_all = pd.concat([liar_tr, liar_te, liar_va], ignore_index=True)
new_all  = pd.concat([new_tr,  new_te,  new_va],  ignore_index=True)

n_test = len(te2)   # 2,296
n_val  = len(va2)   # 2,297
n_train = len(tr2)  # 18,369

# ── independent per-component 8:1:1 split (experiments 1.2, 1.3, 4.1) ─────────
# Each component (LIAR, NEW) is split into 8:1:1 with its OWN random partition,
# then merged for 4.1 so that 4.1 = concat(1.2, 1.3). This is distinct from the
# official LIAR2 split (5.1): the two share row membership only by coincidence.
def split_811(df, seed):
    d = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(d); n_te = round(n * 0.1); n_va = round(n * 0.1)
    test  = d.iloc[:n_te].reset_index(drop=True)
    val   = d.iloc[n_te:n_te + n_va].reset_index(drop=True)
    train = d.iloc[n_te + n_va:].reset_index(drop=True)
    return train, test, val

liar_tr, liar_te, liar_va = split_811(liar_all, 42)
new_tr,  new_te,  new_va  = split_811(new_all, 42)

# ── 1.2: LIAR 8:1:1 (independent partition) ──────────────────────────────────
liar_tr.to_csv(OUT_DIR / 'Origin_8_1_1_train.csv', index=False)
liar_te.to_csv(OUT_DIR / 'Origin_8_1_1_test.csv',  index=False)
liar_va.to_csv(OUT_DIR / 'Origin_8_1_1_val.csv',   index=False)
print(f'1.2 LIAR 8:1:1 -> train={len(liar_tr)}, test={len(liar_te)}, val={len(liar_va)}')

# ── 1.3: NEW 8:1:1 ────────────────────────────────────────────────────────────
new_tr.to_csv(OUT_DIR / 'New_8_1_1_train.csv', index=False)
new_te.to_csv(OUT_DIR / 'New_8_1_1_test.csv',  index=False)
new_va.to_csv(OUT_DIR / 'New_8_1_1_val.csv',   index=False)
print(f'1.3 NEW 8:1:1  -> train={len(new_tr)},  test={len(new_te)},  val={len(new_va)}')

# ── 2.1 / 2.2: whole-dataset files (each script assigns train/test/val itself) ─
liar_all.to_csv(OUT_DIR / 'LIAR2023_pt1_origin.csv', index=False)
new_all.to_csv( OUT_DIR / 'LIAR2023_pt2_new.csv',    index=False)
print(f'2.1/2.2 LIAR total={len(liar_all)}, NEW total={len(new_all)}')

# ── 3.1: train=LIAR(1.)+NEW(.557)  test/val=NEW(.443) ────────────────────────
# size chosen so that train size == LIAR2 train size
new_for_train = n_train - len(liar_all)          # 18369 - 12572 = 5797
new_31 = new_all.sample(frac=1, random_state=42).reset_index(drop=True)
new_31_tr   = new_31.iloc[:new_for_train]
new_31_rest = new_31.iloc[new_for_train:]
new_31_te   = new_31_rest.iloc[:n_test]
new_31_va   = new_31_rest.iloc[n_test:]

train_31 = pd.concat([liar_all, new_31_tr], ignore_index=True) \
             .sample(frac=1, random_state=42).reset_index(drop=True)
train_31.to_csv( OUT_DIR / 'Train_origin_test_new_train.csv', index=False)
new_31_te.to_csv(OUT_DIR / 'Train_origin_test_new_test.csv',  index=False)
new_31_va.to_csv(OUT_DIR / 'Train_origin_test_new_val.csv',   index=False)
print(f'3.1 -> train={len(train_31)}, test={len(new_31_te)}, val={len(new_31_va)}')

# ── 3.2: train=NEW(1.)+LIAR(.635)  test/val=LIAR(.365) ──────────────────────
liar_for_train = n_train - len(new_all)          # 18369 - 10390 = 7979
liar_32 = liar_all.sample(frac=1, random_state=42).reset_index(drop=True)
liar_32_tr   = liar_32.iloc[:liar_for_train]
liar_32_rest = liar_32.iloc[liar_for_train:]
liar_32_te   = liar_32_rest.iloc[:n_test]
liar_32_va   = liar_32_rest.iloc[n_test:]

train_32 = pd.concat([new_all, liar_32_tr], ignore_index=True) \
             .sample(frac=1, random_state=42).reset_index(drop=True)
train_32.to_csv( OUT_DIR / 'Train_new_test_origin_train.csv', index=False)
liar_32_te.to_csv(OUT_DIR / 'Train_new_test_origin_test.csv',  index=False)
liar_32_va.to_csv(OUT_DIR / 'Train_new_test_origin_val.csv',   index=False)
print(f'3.2 -> train={len(train_32)}, test={len(liar_32_te)}, val={len(liar_32_va)}')

# ── 4.1: LIAR(.8)+NEW(.8) / LIAR(.2)+NEW(.2) = concat of independent splits ──
mix_tr = pd.concat([liar_tr, new_tr], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
mix_te = pd.concat([liar_te, new_te], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
mix_va = pd.concat([liar_va, new_va], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
mix_tr.to_csv(OUT_DIR / 'Origin_new_mix_8_1_1_train.csv', index=False)
mix_te.to_csv(OUT_DIR / 'Origin_new_mix_8_1_1_test.csv',  index=False)
mix_va.to_csv(OUT_DIR / 'Origin_new_mix_8_1_1_val.csv',   index=False)
print(f'4.1 -> train={len(mix_tr)}, test={len(mix_te)}, val={len(mix_va)} (independent per-component)')
