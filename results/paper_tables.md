# Main results (mean over seeds; best per row in bold)

## Proxy F1 on the held-out test set

| dataset | positives | QSB (ours) | Random (Chung) | Top-K (Chung) | Keyword (KBS) | SEALS-MLP | LTS-LR |
|---|---|---|---|---|---|---|---|
| 20 Newsgroups (comp.*) (budget 500) | 5% | **0.633** | 0.577 | 0.261 | 0.571 | 0.621 | 0.619 |
| 20 Newsgroups (comp.*) (budget 500) | 1% | **0.472** | 0.198 | 0.145 | 0.312 | 0.437 | 0.075 |
| 20 Newsgroups (comp.*) (budget 500) | 0.5% | **0.369** | 0.141 | 0.107 | 0.256 | 0.364 | 0.087 |
| Toxic Conversations (budget 500) | 7.9% | 0.252 | 0.294 | 0.274 | 0.231 | 0.175 | **0.320** |
| Toxic Conversations (budget 500) | 1% | **0.052** | 0.025 | 0.028 | 0.039 | 0.041 | 0.020 |
| Toxic Conversations (budget 500) | 0.5% | 0.016 | 0.013 | 0.016 | **0.025** | 0.011 | 0.002 |
| Toxic Conversations (budget 500) | 0.1% | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## End-to-end F1 of the query result

| dataset | positives | QSB (ours) | Random (Chung) | Top-K (Chung) | Keyword (KBS) | SEALS-MLP | LTS-LR |
|---|---|---|---|---|---|---|---|
| 20 Newsgroups (comp.*) (budget 500) | 5% | 0.688 | 0.588 | 0.272 | 0.597 | **0.734** | 0.634 |
| 20 Newsgroups (comp.*) (budget 500) | 1% | 0.653 | 0.232 | 0.173 | 0.379 | **0.743** | 0.118 |
| 20 Newsgroups (comp.*) (budget 500) | 0.5% | 0.600 | 0.185 | 0.145 | 0.359 | **0.707** | 0.121 |
| Toxic Conversations (budget 500) | 7.9% | 0.262 | 0.299 | 0.277 | 0.235 | 0.188 | **0.323** |
| Toxic Conversations (budget 500) | 1% | **0.092** | 0.033 | 0.038 | 0.040 | 0.072 | 0.024 |
| Toxic Conversations (budget 500) | 0.5% | **0.058** | 0.018 | 0.028 | 0.029 | 0.057 | 0.013 |
| Toxic Conversations (budget 500) | 0.1% | 0.021 | 0.011 | 0.031 | 0.031 | **0.049** | 0.013 |

## Positives found with the budget

| dataset | positives | QSB (ours) | Random (Chung) | Top-K (Chung) | Keyword (KBS) | SEALS-MLP | LTS-LR |
|---|---|---|---|---|---|---|---|
| 20 Newsgroups (comp.*) (budget 500) | 5% | 296.4 | 25.8 | 169.6 | 113.8 | **317.2** | 59.0 |
| 20 Newsgroups (comp.*) (budget 500) | 1% | 60.2 | 5.8 | 38.0 | 25.8 | **75.6** | 4.2 |
| 20 Newsgroups (comp.*) (budget 500) | 0.5% | 28.4 | 4.6 | 21.0 | 13.6 | **36.6** | 3.4 |
| Toxic Conversations (budget 500) | 7.9% | 243.2 | 36.6 | 76.8 | 229.4 | **366.0** | 41.6 |
| Toxic Conversations (budget 500) | 1% | 26.8 | 5.6 | 10.6 | **44.6** | 29.2 | 4.0 |
| Toxic Conversations (budget 500) | 0.5% | 12.8 | 2.4 | 5.2 | **23.2** | 13.2 | 2.2 |
| Toxic Conversations (budget 500) | 0.1% | 1.0 | 0.6 | 1.6 | **3.4** | 2.4 | 0.6 |
