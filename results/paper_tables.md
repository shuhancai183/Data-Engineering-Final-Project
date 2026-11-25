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
| LTS shark ads (budget 500) | 5.5% | 0.668 | 0.729 | 0.393 | 0.705 | 0.724 | **0.756** |
| LTS shark ads (budget 2000) | 5.5% | **0.810** | 0.772 | 0.434 | 0.703 | 0.806 | 0.760 |
| LTS shark ads (budget 500) | 1% | **0.732** | 0.568 | 0.215 | 0.681 | 0.710 | 0.512 |
| LTS shark ads (budget 2000) | 1% | **0.808** | 0.749 | 0.455 | 0.720 | 0.789 | 0.757 |
| LTS shark ads (budget 500) | 0.5% | **0.712** | 0.259 | 0.141 | 0.635 | 0.407 | 0.096 |
| LTS shark ads (budget 2000) | 0.5% | 0.724 | 0.544 | 0.591 | 0.729 | **0.740** | 0.657 |
| Jigsaw: threat (budget 500) | 0.3% | 0.370 | 0.088 | 0.180 | 0.107 | **0.408** | 0.154 |
| Jigsaw: threat / identity hate / severe (budget 500) | 1.9% | 0.432 | 0.352 | 0.199 | 0.390 | 0.359 | **0.451** |
| Jigsaw: threat / identity hate / severe (budget 500) | 0.5% | **0.267** | 0.061 | 0.044 | 0.182 | 0.240 | 0.169 |

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
| LTS shark ads (budget 500) | 5.5% | - | - | - | - | - | - |
| LTS shark ads (budget 2000) | 5.5% | - | - | - | - | - | - |
| LTS shark ads (budget 500) | 1% | - | - | - | - | - | - |
| LTS shark ads (budget 2000) | 1% | - | - | - | - | - | - |
| LTS shark ads (budget 500) | 0.5% | - | - | - | - | - | - |
| LTS shark ads (budget 2000) | 0.5% | - | - | - | - | - | - |
| Jigsaw: threat (budget 500) | 0.3% | 0.413 | 0.091 | 0.194 | 0.106 | **0.516** | 0.144 |
| Jigsaw: threat / identity hate / severe (budget 500) | 1.9% | 0.446 | 0.352 | 0.200 | 0.396 | 0.370 | **0.455** |
| Jigsaw: threat / identity hate / severe (budget 500) | 0.5% | **0.290** | 0.062 | 0.057 | 0.191 | 0.286 | 0.167 |

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
| LTS shark ads (budget 500) | 5.5% | 355.6 | 28.2 | 24.0 | 50.4 | **402.0** | 78.0 |
| LTS shark ads (budget 2000) | 5.5% | 672.8 | 101.0 | 159.0 | 193.8 | **724.0** | 241.2 |
| LTS shark ads (budget 500) | 1% | **80.6** | 4.8 | 2.4 | 10.2 | 77.8 | 16.2 |
| LTS shark ads (budget 2000) | 1% | 112.2 | 18.0 | 27.6 | 40.0 | **124.2** | 29.6 |
| LTS shark ads (budget 500) | 0.5% | **42.2** | 1.8 | 2.8 | 6.0 | 29.8 | 1.6 |
| LTS shark ads (budget 2000) | 0.5% | 57.0 | 8.0 | 15.4 | 21.8 | **59.6** | 17.6 |
| Jigsaw: threat (budget 500) | 0.3% | 90.6 | 1.8 | 4.8 | 28.8 | **136.0** | 3.6 |
| Jigsaw: threat / identity hate / severe (budget 500) | 1.9% | 200.6 | 8.2 | 5.8 | 94.2 | **247.8** | 49.4 |
| Jigsaw: threat / identity hate / severe (budget 500) | 0.5% | 56.2 | 2.4 | 3.2 | 32.8 | **86.0** | 12.2 |
