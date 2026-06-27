# Mix Dataset Splits

This folder contains the pre-generated CSV files used by the segmentation experiments (scripts 1.2 – 4.1).
All files follow the 16-column LIAR2 schema.

LIAR2 is split into two parts: **LIAR** (the original LIAR dataset restructured with the LIAR2 schema, 12,572 records)
and **NEW** (the incremental data added beyond the original LIAR, 10,390 records).
LIAR records are identified by matching integer IDs with the original LIAR dataset.

## Generated Files

| Script | File prefix | Train | Test | Val | Description |
|--------|-------------|------:|-----:|----:|-------------|
| 1.2 | `Origin_8_1_1_*` | 10,013 | 1,281 | 1,278 | LIAR part only, preserving the 8:1:1 split from LIAR2 |
| 1.3 | `New_8_1_1_*` | 8,356 | 1,015 | 1,019 | NEW part only, preserving the 8:1:1 split from LIAR2 |
| 2.1 | `LIAR2023_pt1_origin.csv` | 12,572 | — | — | All LIAR records (unsplit, used as train in 2.1 and test/val in 2.2) |
| 2.2 | `LIAR2023_pt2_new.csv` | 10,390 | — | — | All NEW records (unsplit, used as test/val in 2.1 and train in 2.2) |
| 3.1 | `Train_origin_test_new_*` | 18,369 | 2,296 | 2,297 | Train: LIAR (1.0) + NEW (0.557); Test/Val: NEW (0.443) |
| 3.2 | `Train_new_test_origin_*` | 18,369 | 2,296 | 2,297 | Train: NEW (1.0) + LIAR (0.635); Test/Val: LIAR (0.365) |
| 4.1 | `Origin_new_mix_8_1_1_*` | 18,369 | 2,296 | 2,297 | Full LIAR2 8:1:1 split (LIAR 0.8 + NEW 0.8 / LIAR 0.2 + NEW 0.2) |

## Generation

All files are produced by `src/generate_mix_datasets.py` (random seed 42).
The 3.1 and 3.2 fractional splits are sized so that the training set matches
the LIAR2 training set size (18,369 records).
