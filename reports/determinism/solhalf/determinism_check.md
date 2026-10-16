# Determinism check (D7)

20 DIII-D training shots and 5 MAST test shots rebuilt in a temporary directory and compared with the stored tables. Features and targets built from the rebuilt frame table. Row digest: SHA-256 of the row bytes, column by column; values: exact equality, NaN equal to NaN.

| Table | Split or list | Shots | Rows stored | Rows rebuilt | SHA-256 stored | SHA-256 rebuilt | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frame table | diii_d_train | 20 | 4585 | 4585 | `57ebeb65e928c0fe` | `57ebeb65e928c0fe` | identical |
| frame table | mast_public_test | 5 | 295 | 295 | `936789d071d72f5f` | `936789d071d72f5f` | identical |
| features v2_solhalf | diii_d_train | | | | | | skipped: diii_d_train.parquet not in /Users/kamia/Desktop/PROJECTS/fusion-challenge/data/processed/features/v2_solhalf |
| features v2_solhalf | mast_public_test | 5 | 295 | 295 | `f0afcdb9ad81c13c` | `f0afcdb9ad81c13c` | identical |
| targets v1 | train | 13 | 2951 | 2951 | `1d1c9f214078353c` | `1d1c9f214078353c` | identical |
| targets v1 | validation | 6 | 1481 | 1481 | `4ac68d7ee1ef1fa9` | `4ac68d7ee1ef1fa9` | identical |
| targets v1 | local_test | 1 | 153 | 153 | `4bbcb2a0d7f7cb0f` | `4bbcb2a0d7f7cb0f` | identical |

- Commit now `64d128ebfbd0`; stored builds made at: frame table `393318b0c6b9`, features `64d128ebfbd0`, targets `c89e246de89d`. A difference with a different commit can come from a code change rather than non-determinism.
- Target objects against their checksums in build.json: all match
- Target shape numbers: train scorer reference builder; validation scorer reference builder; local_test scorer reference builder
- Point tables: DIII-D from the cache, MAST from the cache
- Feature build v2_solhalf, coil models from its build.json: DIII-D solenoid_flat, MAST sol_half

Runtime 18 s.
