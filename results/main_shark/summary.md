# Results: main_shark

Mean ± std over seeds. Budget = [np.int64(500), np.int64(2000)] LLM labels. Proxy F1 / precision / recall: against gold labels on a fixed held-out test set that no sampler may label (30% of the table, stratified; LTS shark: the expert-labeled ads). End-to-end F1: the query result over the whole table, using bought labels for labeled rows and the proxy for every other row.

## LTS shark ads, pos_rate = natural, budget = 500 (5.52% positives, 102 of 14990 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 355.6 ± 8.0 | - | 0.668 ± 0.009 | 0.930 ± 0.020 | 0.522 ± 0.008 | - | 24.3 |
| Random (Chung) | 28.2 ± 3.0 | - | 0.729 ± 0.043 | 0.716 ± 0.044 | 0.751 ± 0.095 | - | 0.1 |
| Top-K (Chung) | 24.0 ± 0.0 | - | 0.393 ± 0.000 | 0.249 ± 0.000 | 0.931 ± 0.000 | - | 0.0 |
| Keyword (KBS) | 50.4 ± 5.8 | - | 0.705 ± 0.034 | 0.614 ± 0.042 | 0.829 ± 0.036 | - | 0.1 |
| SEALS-MLP | 402.0 ± 0.0 | - | 0.724 ± 0.000 | 0.967 ± 0.000 | 0.578 ± 0.000 | - | 27.7 |
| LTS-LR | 78.0 ± 35.7 | - | 0.756 ± 0.032 | 0.804 ± 0.079 | 0.727 ± 0.104 | - | 10.4 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 0/5 (p=1.000), more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), more positives in 5/5
- Keyword (KBS): proxy F1 higher in 0/5 (p=1.000), more positives in 5/5
- SEALS-MLP: proxy F1 higher in 0/5 (p=1.000), more positives in 0/5
- LTS-LR: proxy F1 higher in 0/5 (p=1.000), more positives in 5/5

## LTS shark ads, pos_rate = natural, budget = 2000 (5.52% positives, 102 of 14990 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 672.8 ± 4.8 | - | 0.810 ± 0.010 | 0.860 ± 0.014 | 0.767 ± 0.020 | - | 24.1 |
| Random (Chung) | 101.0 ± 9.7 | - | 0.772 ± 0.007 | 0.695 ± 0.024 | 0.871 ± 0.040 | - | 0.1 |
| Top-K (Chung) | 159.0 ± 0.0 | - | 0.434 ± 0.000 | 0.284 ± 0.000 | 0.912 ± 0.000 | - | 0.0 |
| Keyword (KBS) | 193.8 ± 5.6 | - | 0.703 ± 0.019 | 0.595 ± 0.032 | 0.861 ± 0.013 | - | 0.1 |
| SEALS-MLP | 724.0 ± 0.0 | - | 0.806 ± 0.000 | 0.893 ± 0.000 | 0.735 ± 0.000 | - | 34.3 |
| LTS-LR | 241.2 ± 123.7 | - | 0.760 ± 0.047 | 0.705 ± 0.070 | 0.829 ± 0.055 | - | 14.0 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 5/5 (p=0.031), more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), more positives in 5/5
- Keyword (KBS): proxy F1 higher in 5/5 (p=0.031), more positives in 5/5
- SEALS-MLP: proxy F1 higher in 3/5 (p=0.500), more positives in 0/5
- LTS-LR: proxy F1 higher in 4/5 (p=0.188), more positives in 5/5

## LTS shark ads, pos_rate = 0.01, budget = 500 (0.998% positives, 102 of 14328 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 80.6 ± 3.4 | - | 0.732 ± 0.052 | 0.951 ± 0.012 | 0.598 ± 0.072 | - | 19.7 |
| Random (Chung) | 4.8 ± 1.8 | - | 0.568 ± 0.100 | 0.940 ± 0.028 | 0.414 ± 0.100 | - | 0.1 |
| Top-K (Chung) | 2.4 ± 1.1 | - | 0.215 ± 0.152 | 0.305 ± 0.131 | 0.214 ± 0.167 | - | 0.0 |
| Keyword (KBS) | 10.2 ± 1.6 | - | 0.681 ± 0.061 | 0.805 ± 0.074 | 0.600 ± 0.102 | - | 0.1 |
| SEALS-MLP | 77.8 ± 18.3 | - | 0.710 ± 0.051 | 0.879 ± 0.123 | 0.604 ± 0.062 | - | 22.7 |
| LTS-LR | 16.2 ± 17.4 | - | 0.512 ± 0.237 | 0.896 ± 0.083 | 0.380 ± 0.196 | - | 8.1 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 5/5 (p=0.031), more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), more positives in 5/5
- Keyword (KBS): proxy F1 higher in 3/5 (p=0.500), more positives in 5/5
- SEALS-MLP: proxy F1 higher in 2/5 (p=0.812), more positives in 3/5
- LTS-LR: proxy F1 higher in 5/5 (p=0.031), more positives in 5/5

## LTS shark ads, pos_rate = 0.01, budget = 2000 (0.998% positives, 102 of 14328 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 112.2 ± 5.7 | - | 0.808 ± 0.011 | 0.880 ± 0.016 | 0.747 ± 0.025 | - | 21.6 |
| Random (Chung) | 18.0 ± 4.5 | - | 0.749 ± 0.056 | 0.866 ± 0.074 | 0.663 ± 0.058 | - | 0.1 |
| Top-K (Chung) | 27.6 ± 5.9 | - | 0.455 ± 0.064 | 0.312 ± 0.061 | 0.859 ± 0.033 | - | 0.0 |
| Keyword (KBS) | 40.0 ± 9.2 | - | 0.720 ± 0.040 | 0.702 ± 0.085 | 0.749 ± 0.076 | - | 0.1 |
| SEALS-MLP | 124.2 ± 5.2 | - | 0.789 ± 0.025 | 0.886 ± 0.038 | 0.712 ± 0.019 | - | 36.6 |
| LTS-LR | 29.6 ± 20.5 | - | 0.757 ± 0.034 | 0.846 ± 0.054 | 0.690 ± 0.070 | - | 13.0 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 4/5 (p=0.188), more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), more positives in 5/5
- Keyword (KBS): proxy F1 higher in 5/5 (p=0.031), more positives in 5/5
- SEALS-MLP: proxy F1 higher in 4/5 (p=0.188), more positives in 0/5
- LTS-LR: proxy F1 higher in 4/5 (p=0.188), more positives in 5/5

## LTS shark ads, pos_rate = 0.005, budget = 500 (0.501% positives, 102 of 14259 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 42.2 ± 4.8 | - | 0.712 ± 0.046 | 0.931 ± 0.030 | 0.578 ± 0.060 | - | 15.9 |
| Random (Chung) | 1.8 ± 0.8 | - | 0.259 ± 0.250 | 0.733 ± 0.435 | 0.169 ± 0.171 | - | 0.1 |
| Top-K (Chung) | 2.8 ± 2.9 | - | 0.141 ± 0.193 | 0.131 ± 0.138 | 0.200 ± 0.346 | - | 0.0 |
| Keyword (KBS) | 6.0 ± 1.6 | - | 0.635 ± 0.062 | 0.891 ± 0.108 | 0.504 ± 0.096 | - | 0.1 |
| SEALS-MLP | 29.8 ± 27.2 | - | 0.407 ± 0.372 | 0.576 ± 0.526 | 0.316 ± 0.289 | - | 19.3 |
| LTS-LR | 1.6 ± 0.9 | - | 0.096 ± 0.215 | 0.189 ± 0.422 | 0.065 ± 0.145 | - | 5.8 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 5/5 (p=0.031), more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), more positives in 5/5
- Keyword (KBS): proxy F1 higher in 4/5 (p=0.188), more positives in 5/5
- SEALS-MLP: proxy F1 higher in 5/5 (p=0.031), more positives in 3/5
- LTS-LR: proxy F1 higher in 5/5 (p=0.031), more positives in 5/5

## LTS shark ads, pos_rate = 0.005, budget = 2000 (0.501% positives, 102 of 14259 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 57.0 ± 4.5 | - | 0.724 ± 0.038 | 0.872 ± 0.052 | 0.620 ± 0.046 | - | 18.7 |
| Random (Chung) | 8.0 ± 3.4 | - | 0.544 ± 0.234 | 0.819 ± 0.142 | 0.427 ± 0.202 | - | 0.1 |
| Top-K (Chung) | 15.4 ± 3.0 | - | 0.591 ± 0.071 | 0.520 ± 0.079 | 0.688 ± 0.058 | - | 0.0 |
| Keyword (KBS) | 21.8 ± 2.4 | - | 0.729 ± 0.052 | 0.838 ± 0.039 | 0.649 ± 0.081 | - | 0.1 |
| SEALS-MLP | 59.6 ± 5.2 | - | 0.740 ± 0.043 | 0.903 ± 0.039 | 0.629 ± 0.062 | - | 31.8 |
| LTS-LR | 17.6 ± 9.4 | - | 0.657 ± 0.111 | 0.914 ± 0.026 | 0.524 ± 0.123 | - | 11.2 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 5/5 (p=0.031), more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), more positives in 5/5
- Keyword (KBS): proxy F1 higher in 3/5 (p=0.500), more positives in 5/5
- SEALS-MLP: proxy F1 higher in 3/5 (p=0.500), more positives in 1/5
- LTS-LR: proxy F1 higher in 2/5 (p=0.812), more positives in 5/5
