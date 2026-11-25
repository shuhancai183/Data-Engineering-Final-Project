# Results: main_jigsaw

Mean ± std over seeds. Budget = [np.int64(500)] LLM labels. Proxy F1 / precision / recall: against gold labels on a fixed held-out test set that no sampler may label (30% of the table, stratified; LTS shark: the expert-labeled ads). End-to-end F1: the query result over the whole table, using bought labels for labeled rows and the proxy for every other row.

## Jigsaw: threat, pos_rate = natural (0.3% positives, 478 of 159571 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 90.6 ± 26.9 | - | 0.370 ± 0.051 | 0.323 ± 0.084 | 0.466 ± 0.064 | 0.413 ± 0.066 | 50.9 |
| Random (Chung) | 1.8 ± 0.8 | - | 0.088 ± 0.105 | 0.351 ± 0.281 | 0.060 ± 0.082 | 0.091 ± 0.108 | 0.2 |
| Top-K (Chung) | 4.8 ± 0.4 | - | 0.180 ± 0.027 | 0.159 ± 0.021 | 0.225 ± 0.092 | 0.194 ± 0.022 | 0.0 |
| Keyword (KBS) | 28.8 ± 5.7 | - | 0.107 ± 0.011 | 0.057 ± 0.006 | 0.724 ± 0.048 | 0.106 ± 0.010 | 0.2 |
| SEALS-MLP | 136.0 ± 9.4 | - | 0.408 ± 0.020 | 0.465 ± 0.061 | 0.371 ± 0.043 | 0.516 ± 0.025 | 166.3 |
| LTS-LR | 3.6 ± 1.1 | - | 0.154 ± 0.135 | 0.356 ± 0.245 | 0.110 ± 0.102 | 0.144 ± 0.121 | 15.5 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- Keyword (KBS): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- SEALS-MLP: proxy F1 higher in 1/5 (p=0.969), end-to-end F1 higher in 0/5, more positives in 0/5
- LTS-LR: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5

## Jigsaw: threat / identity hate / severe, pos_rate = natural (1.87% positives, 2987 of 159571 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 200.6 ± 52.5 | 3.0 / 3 | 0.432 ± 0.026 | 0.534 ± 0.063 | 0.373 ± 0.067 | 0.446 ± 0.030 | 81.8 |
| Random (Chung) | 8.2 ± 5.0 | 2.8 / 3 | 0.352 ± 0.078 | 0.423 ± 0.154 | 0.422 ± 0.231 | 0.352 ± 0.077 | 0.2 |
| Top-K (Chung) | 5.8 ± 1.3 | 2.8 / 3 | 0.199 ± 0.053 | 0.264 ± 0.044 | 0.177 ± 0.086 | 0.200 ± 0.051 | 0.0 |
| Keyword (KBS) | 94.2 ± 7.5 | 3.0 / 3 | 0.390 ± 0.032 | 0.272 ± 0.035 | 0.704 ± 0.034 | 0.396 ± 0.034 | 0.3 |
| SEALS-MLP | 247.8 ± 55.8 | 3.0 / 3 | 0.359 ± 0.036 | 0.581 ± 0.068 | 0.267 ± 0.057 | 0.370 ± 0.034 | 172.6 |
| LTS-LR | 49.4 ± 30.2 | 2.8 / 3 | 0.451 ± 0.032 | 0.399 ± 0.064 | 0.541 ± 0.077 | 0.455 ± 0.028 | 16.1 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 5/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- Keyword (KBS): proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 4/5, more positives in 5/5
- SEALS-MLP: proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 5/5, more positives in 1/5
- LTS-LR: proxy F1 higher in 1/5 (p=0.969), end-to-end F1 higher in 1/5, more positives in 5/5

## Jigsaw: threat / identity hate / severe, pos_rate = 0.005 (0.5% positives, 787 of 157371 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 56.2 ± 14.3 | 3.0 / 3 | 0.267 ± 0.008 | 0.239 ± 0.021 | 0.308 ± 0.036 | 0.290 ± 0.010 | 46.9 |
| Random (Chung) | 2.4 ± 1.7 | 1.8 / 3 | 0.061 ± 0.074 | 0.135 ± 0.148 | 0.064 ± 0.100 | 0.062 ± 0.075 | 0.2 |
| Top-K (Chung) | 3.2 ± 1.6 | 2.6 / 3 | 0.044 ± 0.036 | 0.093 ± 0.077 | 0.036 ± 0.033 | 0.057 ± 0.054 | 0.0 |
| Keyword (KBS) | 32.8 ± 3.6 | 3.0 / 3 | 0.182 ± 0.031 | 0.111 ± 0.025 | 0.542 ± 0.060 | 0.191 ± 0.021 | 0.3 |
| SEALS-MLP | 86.0 ± 36.0 | 3.0 / 3 | 0.240 ± 0.045 | 0.306 ± 0.113 | 0.212 ± 0.047 | 0.286 ± 0.046 | 164.6 |
| LTS-LR | 12.2 ± 10.1 | 2.6 / 3 | 0.169 ± 0.094 | 0.214 ± 0.058 | 0.188 ± 0.132 | 0.167 ± 0.090 | 17.3 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- Keyword (KBS): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- SEALS-MLP: proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 3/5, more positives in 1/5
- LTS-LR: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
