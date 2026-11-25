# Results: budget_toxic

Mean ± std over seeds. Budget = [np.int64(200), np.int64(500), np.int64(1000)] LLM labels. Proxy F1 / precision / recall: against gold labels on a fixed held-out test set that no sampler may label (30% of the table, stratified; LTS shark: the expert-labeled ads). End-to-end F1: the query result over the whole table, using bought labels for labeled rows and the proxy for every other row.

## Toxic Conversations, pos_rate = 0.005, budget = 200 (0.5% positives, 461 of 92140 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 3.8 ± 4.1 | - | 0.011 ± 0.018 | 0.286 ± 0.440 | 0.006 ± 0.009 | 0.021 ± 0.026 | 15.8 |
| Random (Chung) | 1.4 ± 0.9 | - | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.006 ± 0.005 | 0.2 |
| Top-K (Chung) | 2.4 ± 1.7 | - | 0.007 ± 0.012 | 0.007 ± 0.010 | 0.013 ± 0.025 | 0.013 ± 0.011 | 0.0 |
| SEALS-MLP | 6.4 ± 5.0 | - | 0.008 ± 0.017 | 0.032 ± 0.071 | 0.004 ± 0.010 | 0.030 ± 0.025 | 95.9 |
| LTS-LR | 1.6 ± 0.9 | - | 0.003 ± 0.006 | 0.015 ± 0.034 | 0.001 ± 0.003 | 0.010 ± 0.006 | 9.2 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 2/5 (p=0.812), end-to-end F1 higher in 4/5, more positives in 4/5
- Top-K (Chung): proxy F1 higher in 1/5 (p=0.969), end-to-end F1 higher in 3/5, more positives in 3/5
- SEALS-MLP: proxy F1 higher in 2/5 (p=0.812), end-to-end F1 higher in 2/5, more positives in 2/5
- LTS-LR: proxy F1 higher in 2/5 (p=0.812), end-to-end F1 higher in 3/5, more positives in 3/5

## Toxic Conversations, pos_rate = 0.005, budget = 1000 (0.5% positives, 461 of 92140 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 17.4 ± 8.6 | - | 0.047 ± 0.034 | 0.120 ± 0.087 | 0.035 ± 0.029 | 0.081 ± 0.041 | 22.8 |
| Random (Chung) | 4.4 ± 1.5 | - | 0.012 ± 0.015 | 0.018 ± 0.019 | 0.010 ± 0.015 | 0.025 ± 0.015 | 0.1 |
| Top-K (Chung) | 11.0 ± 3.2 | - | 0.025 ± 0.011 | 0.017 ± 0.009 | 0.119 ± 0.110 | 0.029 ± 0.012 | 0.0 |
| SEALS-MLP | 31.2 ± 17.9 | - | 0.051 ± 0.049 | 0.157 ± 0.143 | 0.030 ± 0.030 | 0.134 ± 0.078 | 106.4 |
| LTS-LR | 6.8 ± 2.3 | - | 0.022 ± 0.019 | 0.020 ± 0.015 | 0.029 ± 0.024 | 0.034 ± 0.010 | 13.3 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 5/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 5/5, more positives in 4/5
- SEALS-MLP: proxy F1 higher in 2/5 (p=0.812), end-to-end F1 higher in 1/5, more positives in 1/5
- LTS-LR: proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 5/5, more positives in 5/5

## Toxic Conversations, pos_rate = 0.005, budget = 500 (0.5% positives, 461 of 92140 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 12.8 ± 6.6 | - | 0.016 ± 0.017 | 0.097 ± 0.138 | 0.009 ± 0.009 | 0.058 ± 0.028 | 21.8 |
| Random (Chung) | 2.4 ± 2.1 | - | 0.013 ± 0.013 | 0.029 ± 0.026 | 0.009 ± 0.009 | 0.018 ± 0.016 | 0.2 |
| Top-K (Chung) | 5.2 ± 2.3 | - | 0.016 ± 0.030 | 0.014 ± 0.027 | 0.020 ± 0.035 | 0.028 ± 0.019 | 0.0 |
| SEALS-MLP | 13.2 ± 10.4 | - | 0.011 ± 0.011 | 0.168 ± 0.212 | 0.006 ± 0.006 | 0.057 ± 0.044 | 102.0 |
| LTS-LR | 2.2 ± 0.8 | - | 0.002 ± 0.005 | 0.007 ± 0.015 | 0.001 ± 0.003 | 0.013 ± 0.009 | 7.8 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 1/5 (p=0.969), end-to-end F1 higher in 5/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 2/5 (p=0.812), end-to-end F1 higher in 5/5, more positives in 5/5
- SEALS-MLP: proxy F1 higher in 2/5 (p=0.812), end-to-end F1 higher in 2/5, more positives in 2/5
- LTS-LR: proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 5/5, more positives in 5/5
