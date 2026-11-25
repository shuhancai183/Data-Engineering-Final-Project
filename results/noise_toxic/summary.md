# Results: noise_toxic

Mean ± std over seeds. Budget = [np.int64(500)] LLM labels. Proxy F1 / precision / recall: against gold labels on a fixed held-out test set that no sampler may label (30% of the table, stratified; LTS shark: the expert-labeled ads). End-to-end F1: the query result over the whole table, using bought labels for labeled rows and the proxy for every other row.

## Toxic Conversations, pos_rate = 0.01, label flip rate = 0.01 (1% positives, 926 of 92605 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 29.6 ± 10.3 | - | 0.067 ± 0.035 | 0.120 ± 0.051 | 0.050 ± 0.028 | 0.092 ± 0.037 | 27.8 |
| Random (Chung) | 9.6 ± 1.5 | - | 0.018 ± 0.007 | 0.015 ± 0.008 | 0.024 ± 0.009 | 0.030 ± 0.010 | 0.2 |
| Top-K (Chung) | 16.2 ± 4.0 | - | 0.030 ± 0.018 | 0.018 ± 0.011 | 0.104 ± 0.054 | 0.037 ± 0.019 | 0.0 |
| SEALS-MLP | 27.4 ± 28.6 | - | 0.044 ± 0.053 | 0.206 ± 0.211 | 0.027 ± 0.035 | 0.065 ± 0.070 | 105.9 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 5/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 4/5, more positives in 5/5
- SEALS-MLP: proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 3/5, more positives in 3/5

## Toxic Conversations, pos_rate = 0.01, label flip rate = 0.02 (1% positives, 926 of 92605 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 26.2 ± 9.7 | - | 0.039 ± 0.036 | 0.075 ± 0.065 | 0.027 ± 0.025 | 0.054 ± 0.046 | 27.3 |
| Random (Chung) | 15.8 ± 3.3 | - | 0.016 ± 0.004 | 0.010 ± 0.004 | 0.047 ± 0.016 | 0.026 ± 0.008 | 0.2 |
| Top-K (Chung) | 22.6 ± 4.9 | - | 0.031 ± 0.017 | 0.020 ± 0.013 | 0.145 ± 0.063 | 0.034 ± 0.019 | 0.0 |
| SEALS-MLP | 29.4 ± 31.9 | - | 0.022 ± 0.042 | 0.092 ± 0.110 | 0.014 ± 0.027 | 0.046 ± 0.077 | 108.5 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 3/5, more positives in 4/5
- Top-K (Chung): proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 3/5, more positives in 3/5
- SEALS-MLP: proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 2/5, more positives in 4/5

## Toxic Conversations, pos_rate = 0.01, label flip rate = 0.05 (1% positives, 926 of 92605 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 32.0 ± 5.7 | - | 0.019 ± 0.011 | 0.022 ± 0.011 | 0.017 ± 0.011 | 0.032 ± 0.030 | 28.8 |
| Random (Chung) | 31.8 ± 6.3 | - | 0.015 ± 0.007 | 0.008 ± 0.004 | 0.099 ± 0.037 | 0.018 ± 0.003 | 0.2 |
| Top-K (Chung) | 36.2 ± 5.9 | - | 0.026 ± 0.014 | 0.015 ± 0.010 | 0.217 ± 0.138 | 0.029 ± 0.017 | 0.0 |
| SEALS-MLP | 48.6 ± 13.6 | - | 0.042 ± 0.039 | 0.161 ± 0.110 | 0.024 ± 0.023 | 0.059 ± 0.037 | 105.1 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 3/5, more positives in 3/5
- Top-K (Chung): proxy F1 higher in 0/5 (p=1.000), end-to-end F1 higher in 2/5, more positives in 1/5
- SEALS-MLP: proxy F1 higher in 1/5 (p=0.969), end-to-end F1 higher in 1/5, more positives in 0/5

## Toxic Conversations, pos_rate = 0.01, label flip rate = 0 (1% positives, 926 of 92605 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| QSB (ours) | 26.8 ± 14.3 | - | 0.052 ± 0.045 | 0.117 ± 0.079 | 0.035 ± 0.032 | 0.092 ± 0.046 | 25.5 |
| Random (Chung) | 5.6 ± 1.9 | - | 0.025 ± 0.020 | 0.035 ± 0.011 | 0.024 ± 0.021 | 0.033 ± 0.023 | 0.2 |
| Top-K (Chung) | 10.6 ± 2.7 | - | 0.028 ± 0.017 | 0.020 ± 0.014 | 0.096 ± 0.071 | 0.038 ± 0.021 | 0.0 |
| SEALS-MLP | 29.2 ± 29.4 | - | 0.041 ± 0.051 | 0.165 ± 0.195 | 0.024 ± 0.029 | 0.072 ± 0.070 | 101.5 |

Paired by seed (one-sided sign test), QSB (ours) vs:
- Random (Chung): proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 4/5, more positives in 5/5
- Top-K (Chung): proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 5/5, more positives in 5/5
- SEALS-MLP: proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 4/5, more positives in 3/5
