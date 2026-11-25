# Results: ablation

Mean ± std over seeds. Budget = [np.int64(500)] LLM labels. Proxy F1 / precision / recall: against gold labels on a fixed held-out test set that no sampler may label (30% of the table, stratified; LTS shark: the expert-labeled ads). End-to-end F1: the query result over the whole table, using bought labels for labeled rows and the proxy for every other row.

## 20 Newsgroups (comp.*), pos_rate = 0.01 (1% positives, 136 of 13575 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| path0_lts_lr | 4.2 ± 1.1 | 2.8 / 5 | 0.075 ± 0.088 | 0.106 ± 0.084 | 0.065 ± 0.091 | 0.118 ± 0.049 | 7.3 |
| path1_pos_reward | 26.0 ± 8.1 | 4.8 / 5 | 0.375 ± 0.106 | 0.365 ± 0.119 | 0.415 ± 0.178 | 0.451 ± 0.077 | 9.1 |
| path2_kmeans | 3.0 ± 2.8 | 2.4 / 5 | 0.074 ± 0.120 | 0.205 ± 0.301 | 0.055 ± 0.097 | 0.111 ± 0.157 | 5.7 |
| path3_per_slot_score | 58.4 ± 7.8 | 5.0 / 5 | 0.419 ± 0.053 | 0.413 ± 0.023 | 0.435 ± 0.101 | 0.620 ± 0.039 | 11.8 |
| path4_query_prior_seed | 63.4 ± 9.5 | 5.0 / 5 | 0.450 ± 0.035 | 0.480 ± 0.073 | 0.430 ± 0.048 | 0.677 ± 0.036 | 11.6 |
| path5_qsb | 60.2 ± 11.3 | 5.0 / 5 | 0.472 ± 0.048 | 0.492 ± 0.086 | 0.470 ± 0.097 | 0.653 ± 0.052 | 13.6 |
| qsb_no_prior | 50.6 ± 7.2 | 5.0 / 5 | 0.426 ± 0.047 | 0.425 ± 0.073 | 0.450 ± 0.120 | 0.566 ± 0.035 | 13.3 |
| qsb_no_seed | 56.4 ± 9.6 | 5.0 / 5 | 0.445 ± 0.034 | 0.469 ± 0.090 | 0.450 ± 0.110 | 0.632 ± 0.049 | 13.8 |
| qsb_no_knn | 64.0 ± 9.1 | 5.0 / 5 | 0.451 ± 0.030 | 0.458 ± 0.048 | 0.450 ± 0.056 | 0.674 ± 0.033 | 11.3 |
| qsb_no_random | 63.4 ± 9.5 | 5.0 / 5 | 0.455 ± 0.066 | 0.492 ± 0.081 | 0.445 ± 0.115 | 0.677 ± 0.034 | 13.9 |
| qsb_random30 | 55.6 ± 11.9 | 5.0 / 5 | 0.497 ± 0.046 | 0.521 ± 0.112 | 0.505 ± 0.114 | 0.635 ± 0.042 | 13.7 |
| qsb_random50 | 51.6 ± 8.4 | 5.0 / 5 | 0.449 ± 0.062 | 0.439 ± 0.040 | 0.475 ± 0.121 | 0.600 ± 0.046 | 13.3 |
| qsb_lda | 70.6 ± 6.8 | 5.0 / 5 | 0.425 ± 0.035 | 0.396 ± 0.071 | 0.480 ± 0.096 | 0.642 ± 0.061 | 14.6 |
| qsb_f1_reward | 30.0 ± 6.7 | 4.6 / 5 | 0.391 ± 0.078 | 0.367 ± 0.087 | 0.430 ± 0.086 | 0.485 ± 0.070 | 12.0 |

Paired by seed (one-sided sign test), path5_qsb vs:
- path0_lts_lr: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- path1_pos_reward: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- path2_kmeans: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- path3_per_slot_score: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 4/5, more positives in 4/5
- path4_query_prior_seed: proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 0/5, more positives in 0/5
- qsb_no_prior: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5
- qsb_no_seed: proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 4/5, more positives in 3/5
- qsb_no_knn: proxy F1 higher in 4/5 (p=0.188), end-to-end F1 higher in 0/5, more positives in 0/5
- qsb_no_random: proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 1/5, more positives in 0/5
- qsb_random30: proxy F1 higher in 1/5 (p=0.969), end-to-end F1 higher in 4/5, more positives in 4/5
- qsb_random50: proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 5/5, more positives in 5/5
- qsb_lda: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 3/5, more positives in 0/5
- qsb_f1_reward: proxy F1 higher in 5/5 (p=0.031), end-to-end F1 higher in 5/5, more positives in 5/5

## Toxic Conversations, pos_rate = 0.005 (0.5% positives, 461 of 92140 rows)

| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |
|---|---|---|---|---|---|---|---|
| path0_lts_lr | 2.2 ± 0.8 | - | 0.002 ± 0.005 | 0.007 ± 0.015 | 0.001 ± 0.003 | 0.013 ± 0.009 | 7.9 |
| path1_pos_reward | 1.8 ± 1.8 | - | 0.003 ± 0.006 | 0.009 ± 0.020 | 0.001 ± 0.003 | 0.008 ± 0.008 | 6.9 |
| path2_kmeans | 3.6 ± 2.3 | - | 0.014 ± 0.017 | 0.021 ± 0.028 | 0.010 ± 0.012 | 0.021 ± 0.014 | 9.1 |
| path3_per_slot_score | 12.2 ± 9.1 | - | 0.005 ± 0.007 | 0.036 ± 0.051 | 0.003 ± 0.004 | 0.052 ± 0.037 | 19.0 |
| path4_query_prior_seed | 12.0 ± 12.1 | - | 0.021 ± 0.025 | 0.369 ± 0.446 | 0.012 ± 0.015 | 0.054 ± 0.053 | 18.8 |
| path5_qsb | 12.8 ± 6.6 | - | 0.016 ± 0.017 | 0.097 ± 0.138 | 0.009 ± 0.009 | 0.058 ± 0.028 | 22.6 |
| qsb_no_prior | 9.4 ± 7.2 | - | 0.023 ± 0.031 | 0.098 ± 0.120 | 0.013 ± 0.018 | 0.047 ± 0.039 | 21.1 |
| qsb_no_seed | 10.2 ± 8.6 | - | 0.014 ± 0.024 | 0.034 ± 0.051 | 0.009 ± 0.016 | 0.047 ± 0.043 | 21.4 |
| qsb_no_knn | 14.4 ± 9.9 | - | 0.022 ± 0.020 | 0.180 ± 0.179 | 0.012 ± 0.011 | 0.065 ± 0.044 | 18.2 |
| qsb_no_random | 11.0 ± 7.9 | - | 0.018 ± 0.021 | 0.147 ± 0.149 | 0.010 ± 0.012 | 0.050 ± 0.037 | 21.7 |
| qsb_random30 | 7.6 ± 7.6 | - | 0.015 ± 0.026 | 0.219 ± 0.438 | 0.010 ± 0.019 | 0.037 ± 0.039 | 19.4 |
| qsb_random50 | 9.0 ± 3.5 | - | 0.021 ± 0.023 | 0.059 ± 0.055 | 0.013 ± 0.015 | 0.044 ± 0.020 | 18.8 |
| qsb_lda | 10.8 ± 4.4 | - | 0.006 ± 0.008 | 0.072 ± 0.110 | 0.003 ± 0.004 | 0.046 ± 0.018 | 25.2 |
| qsb_f1_reward | 6.8 ± 4.3 | - | 0.016 ± 0.022 | 0.128 ± 0.264 | 0.012 ± 0.017 | 0.033 ± 0.023 | 17.7 |

Paired by seed (one-sided sign test), path5_qsb vs:
- path0_lts_lr: proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 5/5, more positives in 5/5
- path1_pos_reward: proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 5/5, more positives in 5/5
- path2_kmeans: proxy F1 higher in 3/5 (p=0.500), end-to-end F1 higher in 4/5, more positives in 4/5
- path3_per_slot_score: proxy F1 higher in 2/5 (p=0.812), end-to-end F1 higher in 2/5, more positives in 2/5
- path4_query_prior_seed: proxy F1 higher in 2/5 (p=0.812), end-to-end F1 higher in 4/5, more positives in 3/5
- qsb_no_prior: proxy F1 higher in 2/5 (p=0.812), end-to-end F1 higher in 4/5, more positives in 4/5
- qsb_no_seed: proxy F1 higher in 2/5 (p=0.812), end-to-end F1 higher in 4/5, more positives in 3/5
- qsb_no_knn: proxy F1 higher in 1/5 (p=0.969), end-to-end F1 higher in 2/5, more positives in 1/5
- qsb_no_random: proxy F1 higher in 2/5 (p=0.812), end-to-end F1 higher in 3/5, more positives in 2/5
- qsb_random30: proxy F1 higher in 2/5 (p=0.812), end-to-end F1 higher in 4/5, more positives in 4/5
- qsb_random50: proxy F1 higher in 1/5 (p=0.969), end-to-end F1 higher in 4/5, more positives in 4/5
- qsb_lda: proxy F1 higher in 2/5 (p=0.812), end-to-end F1 higher in 4/5, more positives in 3/5
- qsb_f1_reward: proxy F1 higher in 1/5 (p=0.969), end-to-end F1 higher in 5/5, more positives in 5/5
