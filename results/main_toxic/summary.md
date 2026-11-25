# Results: main_toxic

Mean ± std over seeds. Budget = [np.int64(500)] LLM labels. Proxy F1 / precision / recall: against gold labels on a fixed held-out test set that no sampler may label (30% of the table, stratified; LTS shark: the expert-labeled ads). End-to-end F1: the query result over the whole table, using bought labels for labeled rows and the proxy for every other row.

## Toxic Conversations, pos_rate = natural (7.94% positives, 7904 of 99583 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 243.2 ± 5.5 | - | 0.252 ± 0.031 | 0.562 ± 0.083 | 0.166 ± 0.032 | 0.262 ± 0.028 | 61.8 |
| Random (Chung) | 36.6 ± 9.9 | - | 0.294 ± 0.027 | 0.231 ± 0.025 | 0.423 ± 0.099 | 0.299 ± 0.028 | 0.2 |
| Top-K (Chung) | 76.8 ± 1.8 | - | 0.274 ± 0.006 | 0.184 ± 0.008 | 0.544 ± 0.026 | 0.277 ± 0.007 | 0.0 |
| Keyword (KBS) | 229.4 ± 9.8 | - | 0.231 ± 0.007 | 0.141 ± 0.006 | 0.628 ± 0.032 | 0.235 ± 0.011 | 0.2 |
| SEALS-MLP | 366.0 ± 9.7 | - | 0.175 ± 0.024 | 0.711 ± 0.075 | 0.100 ± 0.016 | 0.188 ± 0.017 | 110.0 |
| LTS-LR | 41.6 ± 12.6 | - | 0.320 ± 0.014 | 0.252 ± 0.013 | 0.443 ± 0.047 | 0.323 ± 0.013 | 13.8 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 1/5 (p=0.969), end-to-end F1 higher in 1/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 2/5 (p=0.812), end-to-end F1 higher in 2/5, more positives in 5/5
- Keyword (KBS): proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 4/5, more positives in 5/5
- SEALS-MLP: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 0/5
- LTS-LR: proxy F1 higher in 0/5 (p=1.000), end-to-end F1 higher in 0/5, more positives in 5/5

## Toxic Conversations, pos_rate = 0.01 (1% positives, 926 of 92605 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 26.8 ± 14.3 | - | 0.052 ± 0.045 | 0.117 ± 0.079 | 0.035 ± 0.032 | 0.092 ± 0.046 | 33.0 |
| Random (Chung) | 5.6 ± 1.9 | - | 0.025 ± 0.020 | 0.035 ± 0.011 | 0.024 ± 0.021 | 0.033 ± 0.023 | 0.2 |
| Top-K (Chung) | 10.6 ± 2.7 | - | 0.028 ± 0.017 | 0.020 ± 0.014 | 0.096 ± 0.071 | 0.038 ± 0.021 | 0.0 |
| Keyword (KBS) | 44.6 ± 7.9 | - | 0.039 ± 0.009 | 0.021 ± 0.005 | 0.391 ± 0.050 | 0.040 ± 0.008 | 0.2 |
| SEALS-MLP | 29.2 ± 29.4 | - | 0.041 ± 0.051 | 0.165 ± 0.195 | 0.024 ± 0.029 | 0.072 ± 0.070 | 114.7 |
| LTS-LR | 4.0 ± 0.7 | - | 0.020 ± 0.022 | 0.042 ± 0.025 | 0.017 ± 0.023 | 0.024 ± 0.018 | 12.9 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 4/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 5/5, more positives in 5/5
- Keyword (KBS): proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 5/5, more positives in 1/5
- SEALS-MLP: proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 4/5, more positives in 3/5
- LTS-LR: proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 5/5, more positives in 5/5

## Toxic Conversations, pos_rate = 0.005 (0.5% positives, 461 of 92140 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 12.8 ± 6.6 | - | 0.016 ± 0.017 | 0.097 ± 0.138 | 0.009 ± 0.009 | 0.058 ± 0.028 | 22.4 |
| Random (Chung) | 2.4 ± 2.1 | - | 0.013 ± 0.013 | 0.029 ± 0.026 | 0.009 ± 0.009 | 0.018 ± 0.016 | 0.2 |
| Top-K (Chung) | 5.2 ± 2.3 | - | 0.016 ± 0.030 | 0.014 ± 0.027 | 0.020 ± 0.035 | 0.028 ± 0.019 | 0.0 |
| Keyword (KBS) | 23.2 ± 5.1 | - | 0.025 ± 0.007 | 0.014 ± 0.004 | 0.186 ± 0.041 | 0.029 ± 0.012 | 0.2 |
| SEALS-MLP | 13.2 ± 10.4 | - | 0.011 ± 0.011 | 0.168 ± 0.212 | 0.006 ± 0.006 | 0.057 ± 0.044 | 112.6 |
| LTS-LR | 2.2 ± 0.8 | - | 0.002 ± 0.005 | 0.007 ± 0.015 | 0.001 ± 0.003 | 0.013 ± 0.009 | 7.7 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 1/5 (p=0.969), end-to-end F1 higher in 5/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 2/5 (p=0.812), end-to-end F1 higher in 5/5, more positives in 5/5
- Keyword (KBS): proxy F1 higher in 1/5 (p=0.969), end-to-end F1 higher in 5/5, more positives in 0/5
- SEALS-MLP: proxy F1 higher in 2/5 (p=0.812), end-to-end F1 higher in 2/5, more positives in 2/5
- LTS-LR: proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 5/5, more positives in 5/5

## Toxic Conversations, pos_rate = 0.001 (0.1% positives, 92 of 91771 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 1.0 ± 1.0 | - | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.021 ± 0.021 | 11.5 |
| Random (Chung) | 0.6 ± 0.9 | - | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.011 ± 0.017 | 0.1 |
| Top-K (Chung) | 1.6 ± 0.5 | - | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.031 ± 0.009 | 0.0 |
| Keyword (KBS) | 3.4 ± 1.1 | - | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.031 ± 0.009 | 0.2 |
| SEALS-MLP | 2.4 ± 3.1 | - | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.049 ± 0.061 | 94.6 |
| LTS-LR | 0.6 ± 0.5 | - | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.013 ± 0.011 | 3.0 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 0/5 (p=1.000), end-to-end F1 higher in 3/5, more positives in 2/5
- Top-K (Chung): proxy F1 higher in 0/5 (p=1.000), end-to-end F1 higher in 2/5, more positives in 1/5
- Keyword (KBS): proxy F1 higher in 0/5 (p=1.000), end-to-end F1 higher in 2/5, more positives in 0/5
- SEALS-MLP: proxy F1 higher in 0/5 (p=1.000), end-to-end F1 higher in 2/5, more positives in 2/5
- LTS-LR: proxy F1 higher in 0/5 (p=1.000), end-to-end F1 higher in 3/5, more positives in 2/5
