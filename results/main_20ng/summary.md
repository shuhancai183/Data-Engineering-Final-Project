# Results: main_20ng

Mean ± std over seeds. Budget = [np.int64(500)] LLM labels. Proxy F1 / precision / recall: against gold labels on a fixed held-out test set that no sampler may label (30% of the table, stratified; LTS shark: the expert-labeled ads). End-to-end F1: the query result over the whole table, using bought labels for labeled rows and the proxy for every other row.

## 20 Newsgroups (comp.*), pos_rate = 0.05 (5% positives, 707 of 14146 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 296.4 ± 15.1 | 5.0 / 5 | 0.633 ± 0.041 | 0.664 ± 0.056 | 0.607 ± 0.043 | 0.688 ± 0.026 | 32.3 |
| Random (Chung) | 25.8 ± 6.9 | 4.8 / 5 | 0.577 ± 0.013 | 0.472 ± 0.025 | 0.745 ± 0.048 | 0.588 ± 0.026 | 0.1 |
| Top-K (Chung) | 169.6 ± 8.4 | 5.0 / 5 | 0.261 ± 0.014 | 0.158 ± 0.010 | 0.742 ± 0.006 | 0.272 ± 0.013 | 0.1 |
| Keyword (KBS) | 113.8 ± 11.5 | 5.0 / 5 | 0.571 ± 0.048 | 0.459 ± 0.057 | 0.761 ± 0.026 | 0.597 ± 0.047 | 0.2 |
| SEALS-MLP | 317.2 ± 6.4 | 5.0 / 5 | 0.621 ± 0.011 | 0.852 ± 0.030 | 0.489 ± 0.017 | 0.734 ± 0.005 | 39.2 |
| LTS-LR | 59.0 ± 26.6 | 5.0 / 5 | 0.619 ± 0.037 | 0.551 ± 0.070 | 0.714 ± 0.030 | 0.634 ± 0.048 | 17.5 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 5/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- Keyword (KBS): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- SEALS-MLP: proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 0/5, more positives in 0/5
- LTS-LR: proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 4/5, more positives in 5/5

## 20 Newsgroups (comp.*), pos_rate = 0.01 (1% positives, 136 of 13575 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 60.2 ± 11.3 | 5.0 / 5 | 0.472 ± 0.048 | 0.492 ± 0.086 | 0.470 ± 0.097 | 0.653 ± 0.052 | 19.7 |
| Random (Chung) | 5.8 ± 1.3 | 3.6 / 5 | 0.198 ± 0.068 | 0.325 ± 0.289 | 0.200 ± 0.088 | 0.232 ± 0.067 | 0.1 |
| Top-K (Chung) | 38.0 ± 3.3 | 5.0 / 5 | 0.145 ± 0.040 | 0.084 ± 0.027 | 0.580 ± 0.112 | 0.173 ± 0.056 | 0.0 |
| Keyword (KBS) | 25.8 ± 6.8 | 5.0 / 5 | 0.312 ± 0.067 | 0.234 ± 0.050 | 0.500 ± 0.170 | 0.379 ± 0.063 | 0.1 |
| SEALS-MLP | 75.6 ± 4.3 | 5.0 / 5 | 0.437 ± 0.033 | 0.452 ± 0.008 | 0.425 ± 0.056 | 0.743 ± 0.019 | 30.0 |
| LTS-LR | 4.2 ± 1.1 | 2.8 / 5 | 0.075 ± 0.088 | 0.106 ± 0.084 | 0.065 ± 0.091 | 0.118 ± 0.049 | 10.8 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- Keyword (KBS): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- SEALS-MLP: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 0/5, more positives in 0/5
- LTS-LR: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5

## 20 Newsgroups (comp.*), pos_rate = 0.005 (0.503% positives, 68 of 13507 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 28.4 ± 3.2 | 4.8 / 5 | 0.369 ± 0.116 | 0.349 ± 0.120 | 0.400 ± 0.122 | 0.600 ± 0.080 | 12.7 |
| Random (Chung) | 4.6 ± 2.1 | 2.8 / 5 | 0.141 ± 0.201 | 0.140 ± 0.222 | 0.160 ± 0.192 | 0.185 ± 0.146 | 0.1 |
| Top-K (Chung) | 21.0 ± 2.7 | 5.0 / 5 | 0.107 ± 0.042 | 0.062 ± 0.026 | 0.450 ± 0.071 | 0.145 ± 0.065 | 0.0 |
| Keyword (KBS) | 13.6 ± 1.3 | 5.0 / 5 | 0.256 ± 0.050 | 0.215 ± 0.060 | 0.330 ± 0.057 | 0.359 ± 0.037 | 0.1 |
| SEALS-MLP | 36.6 ± 2.7 | 5.0 / 5 | 0.364 ± 0.120 | 0.371 ± 0.111 | 0.360 ± 0.129 | 0.707 ± 0.044 | 22.5 |
| LTS-LR | 3.4 ± 1.9 | 2.6 / 5 | 0.087 ± 0.115 | 0.165 ± 0.210 | 0.090 ± 0.147 | 0.121 ± 0.067 | 7.8 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- Keyword (KBS): proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- SEALS-MLP: proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 0/5, more positives in 0/5
- LTS-LR: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
