# Results: noise_20ng

Mean ± std over seeds. Budget = [np.int64(500)] LLM labels. Proxy F1 / precision / recall: against gold labels on a fixed held-out test set that no sampler may label (30% of the table, stratified; LTS shark: the expert-labeled ads). End-to-end F1: the query result over the whole table, using bought labels for labeled rows and the proxy for every other row.

## 20 Newsgroups (comp.*), pos_rate = 0.01, label flip rate = 0.01 (1% positives, 136 of 13575 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 63.0 ± 11.4 | 5.0 / 5 | 0.423 ± 0.050 | 0.425 ± 0.061 | 0.440 ± 0.113 | 0.602 ± 0.054 | 14.0 |
| Random (Chung) | 10.0 ± 2.9 | 3.6 / 5 | 0.147 ± 0.042 | 0.129 ± 0.050 | 0.235 ± 0.098 | 0.163 ± 0.038 | 0.1 |
| Top-K (Chung) | 43.2 ± 2.6 | 5.0 / 5 | 0.093 ± 0.028 | 0.051 ± 0.018 | 0.580 ± 0.099 | 0.114 ± 0.044 | 0.0 |
| SEALS-MLP | 78.6 ± 4.4 | 5.0 / 5 | 0.443 ± 0.043 | 0.449 ± 0.038 | 0.440 ± 0.063 | 0.720 ± 0.021 | 23.4 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- SEALS-MLP: proxy F1 higher in 2/5 (p=0.812), end-to-end F1 higher in 0/5, more positives in 0/5

## 20 Newsgroups (comp.*), pos_rate = 0.01, label flip rate = 0.02 (1% positives, 136 of 13575 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 67.0 ± 7.4 | 5.0 / 5 | 0.426 ± 0.056 | 0.414 ± 0.039 | 0.445 ± 0.097 | 0.597 ± 0.046 | 13.9 |
| Random (Chung) | 14.8 ± 3.1 | 3.6 / 5 | 0.089 ± 0.024 | 0.054 ± 0.015 | 0.255 ± 0.072 | 0.106 ± 0.017 | 0.1 |
| Top-K (Chung) | 44.8 ± 4.1 | 5.0 / 5 | 0.088 ± 0.026 | 0.048 ± 0.015 | 0.640 ± 0.086 | 0.100 ± 0.034 | 0.0 |
| SEALS-MLP | 78.6 ± 7.4 | 5.0 / 5 | 0.454 ± 0.035 | 0.446 ± 0.033 | 0.465 ± 0.060 | 0.705 ± 0.015 | 23.0 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- SEALS-MLP: proxy F1 higher in 1/5 (p=0.969), end-to-end F1 higher in 0/5, more positives in 1/5

## 20 Newsgroups (comp.*), pos_rate = 0.01, label flip rate = 0.05 (1% positives, 136 of 13575 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 73.2 ± 8.6 | 5.0 / 5 | 0.329 ± 0.035 | 0.295 ± 0.067 | 0.415 ± 0.118 | 0.463 ± 0.039 | 14.2 |
| Random (Chung) | 28.0 ± 3.5 | 3.6 / 5 | 0.051 ± 0.021 | 0.028 ± 0.012 | 0.325 ± 0.174 | 0.062 ± 0.020 | 0.1 |
| Top-K (Chung) | 59.8 ± 2.8 | 5.0 / 5 | 0.055 ± 0.018 | 0.029 ± 0.010 | 0.665 ± 0.068 | 0.061 ± 0.019 | 0.0 |
| SEALS-MLP | 84.8 ± 6.1 | 5.0 / 5 | 0.403 ± 0.025 | 0.398 ± 0.029 | 0.410 ± 0.038 | 0.619 ± 0.039 | 23.4 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- SEALS-MLP: proxy F1 higher in 0/5 (p=1.000), end-to-end F1 higher in 0/5, more positives in 0/5

## 20 Newsgroups (comp.*), pos_rate = 0.01, label flip rate = 0 (1% positives, 136 of 13575 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 60.2 ± 11.3 | 5.0 / 5 | 0.472 ± 0.048 | 0.492 ± 0.086 | 0.470 ± 0.097 | 0.653 ± 0.052 | 13.1 |
| Random (Chung) | 5.8 ± 1.3 | 3.6 / 5 | 0.198 ± 0.068 | 0.325 ± 0.289 | 0.200 ± 0.088 | 0.232 ± 0.067 | 0.1 |
| Top-K (Chung) | 38.0 ± 3.3 | 5.0 / 5 | 0.145 ± 0.040 | 0.084 ± 0.027 | 0.580 ± 0.112 | 0.173 ± 0.056 | 0.0 |
| SEALS-MLP | 75.6 ± 4.3 | 5.0 / 5 | 0.437 ± 0.033 | 0.452 ± 0.008 | 0.425 ± 0.056 | 0.743 ± 0.019 | 22.9 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- SEALS-MLP: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 0/5, more positives in 0/5
