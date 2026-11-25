# Results: budget_20ng

Mean ± std over seeds. Budget = [np.int64(200), np.int64(500), np.int64(1000)] LLM labels. Proxy F1 / precision / recall: against gold labels on a fixed held-out test set that no sampler may label (30% of the table, stratified; LTS shark: the expert-labeled ads). End-to-end F1: the query result over the whole table, using bought labels for labeled rows and the proxy for every other row.

## 20 Newsgroups (comp.*), pos_rate = 0.01, budget = 200 (1% positives, 136 of 13575 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 38.6 ± 7.1 | 4.6 / 5 | 0.448 ± 0.070 | 0.652 ± 0.136 | 0.360 ± 0.098 | 0.539 ± 0.039 | 13.2 |
| Random (Chung) | 2.0 ± 1.6 | 1.8 / 5 | 0.057 ± 0.078 | 0.100 ± 0.138 | 0.040 ± 0.055 | 0.071 ± 0.083 | 0.1 |
| Top-K (Chung) | 19.8 ± 3.0 | 5.0 / 5 | 0.057 ± 0.011 | 0.030 ± 0.006 | 0.540 ± 0.055 | 0.059 ± 0.007 | 0.0 |
| SEALS-MLP | 47.4 ± 6.5 | 5.0 / 5 | 0.415 ± 0.088 | 0.678 ± 0.151 | 0.315 ± 0.099 | 0.591 ± 0.049 | 20.4 |
| LTS-LR | 2.6 ± 0.9 | 2.2 / 5 | 0.065 ± 0.083 | 0.301 ± 0.435 | 0.040 ± 0.052 | 0.081 ± 0.048 | 6.0 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- SEALS-MLP: proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 0/5, more positives in 0/5
- LTS-LR: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5

## 20 Newsgroups (comp.*), pos_rate = 0.01, budget = 1000 (1% positives, 136 of 13575 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 78.6 ± 8.0 | 5.0 / 5 | 0.424 ± 0.018 | 0.334 ± 0.014 | 0.585 ± 0.074 | 0.700 ± 0.036 | 15.4 |
| Random (Chung) | 12.0 ± 4.5 | 4.4 / 5 | 0.274 ± 0.073 | 0.277 ± 0.166 | 0.445 ± 0.220 | 0.316 ± 0.067 | 0.0 |
| Top-K (Chung) | 56.6 ± 3.0 | 5.0 / 5 | 0.213 ± 0.035 | 0.131 ± 0.026 | 0.590 ± 0.095 | 0.285 ± 0.050 | 0.0 |
| SEALS-MLP | 84.0 ± 1.7 | 5.0 / 5 | 0.448 ± 0.055 | 0.392 ± 0.038 | 0.530 ± 0.105 | 0.768 ± 0.018 | 26.0 |
| LTS-LR | 12.8 ± 10.4 | 4.4 / 5 | 0.206 ± 0.119 | 0.174 ± 0.066 | 0.295 ± 0.226 | 0.247 ± 0.117 | 9.3 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- SEALS-MLP: proxy F1 higher in 1/5 (p=0.969), end-to-end F1 higher in 0/5, more positives in 1/5
- LTS-LR: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5

## 20 Newsgroups (comp.*), pos_rate = 0.01, budget = 500 (1% positives, 136 of 13575 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 60.2 ± 11.3 | 5.0 / 5 | 0.472 ± 0.048 | 0.492 ± 0.086 | 0.470 ± 0.097 | 0.653 ± 0.052 | 13.6 |
| Random (Chung) | 5.8 ± 1.3 | 3.6 / 5 | 0.198 ± 0.068 | 0.325 ± 0.289 | 0.200 ± 0.088 | 0.232 ± 0.067 | 0.1 |
| Top-K (Chung) | 38.0 ± 3.3 | 5.0 / 5 | 0.145 ± 0.040 | 0.084 ± 0.027 | 0.580 ± 0.112 | 0.173 ± 0.056 | 0.0 |
| SEALS-MLP | 75.6 ± 4.3 | 5.0 / 5 | 0.437 ± 0.033 | 0.452 ± 0.008 | 0.425 ± 0.056 | 0.743 ± 0.019 | 23.0 |
| LTS-LR | 4.2 ± 1.1 | 2.8 / 5 | 0.075 ± 0.088 | 0.106 ± 0.084 | 0.065 ± 0.091 | 0.118 ± 0.049 | 7.2 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- SEALS-MLP: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 0/5, more positives in 0/5
- LTS-LR: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
