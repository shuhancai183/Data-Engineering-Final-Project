# Data-Engineering-Final-Project

**Proxy models for needle-in-a-haystack semantic filters (Project 6, Direction B: finding positives cheaply).**

A semantic filter such as `WHERE AI.IF('the ad sells a real shark product', description)` calls an LLM
on every row. The standard shortcut (Chung et al., SIGMOD 2026) labels a few hundred *random* rows with
the LLM, trains a logistic-regression **proxy** on embeddings, and runs the proxy on the rest. When only
0.1–1% of rows match, a random sample contains almost no positives and the proxy fails. LTS
(Barbosa et al., SIGMOD 2025) finds rare positives with a cluster bandit, but retrains BERT every round
and runs offline for hours.

**Question.** With a fixed budget of LLM labels, can a sampler find more, and more diverse, positives
than random sampling, train a better proxy, and still run at query-engine speed?

**Hypotheses.** For tables with ≤1% positives, at equal budget, QSB
- **H1** finds more positives per LLM call than Random and Top-K,
- **H2** covers more kinds of positives and yields a proxy with higher F1 on human gold labels,
- **H3** needs only milliseconds per round (select + retrain).

## Method: QSB (Query-Seeded Bandit)

1. **Index** (no labels): embed every row (MiniLM, 384-d) and the query text; k-means into 50 clusters.
2. **Query prior**: each cluster starts with Beta(α, β) where α grows with the cosine similarity of the
   cluster centroid to the query (strength 5). Solves the cold start of LTS.
3. **Seeds** (10% of the budget): rows most similar to the query, at most 3 per cluster.
4. **Bandit rounds** (batches of 25): 15% of each batch is uniform random; every other slot draws a
   Thompson sample per cluster, takes the best cluster, and inside it the unlabeled row with the highest
   proxy probability plus a bonus for being a k-NN of a found positive (SEALS idea).
   Reward = the row is positive. The logistic regression is retrained after each batch (milliseconds).
5. **Proxy**: logistic regression with balanced class weights on all bought labels (as in Chung et al.).

## Baselines (same embeddings, budget and final proxy)

| name | what it is |
|---|---|
| `random` | uniform sample, the default in Chung et al. |
| `topk` | rows most similar to the query embedding (Top-K in Chung et al., Table 10) |
| `keyword` | rows containing a keyword (KBS in LTS); hand-written lists in `configs/keywords/` |
| `seals_mlp` | SEALS (Coleman et al., AAAI 2022): candidates = 100-NN of labeled rows, pick most-likely positive |
| `lts_lr` | LTS with BERT replaced by logistic regression: LDA (10 topics), Thompson sampling with decay 0.99, reward = validation-F1 improvement, up to half predicted positives per batch |

`configs/ablation.yaml` walks from `lts_lr` to `qsb` one change at a time and also removes each QSB
component separately.

## Datasets (all with human gold labels; no manual annotation)

| task | role | rows | positives | how to get it |
|---|---|---|---|---|
| `20ng_comp` | development / control | 18,161 | comp.* posts, 26.0% (5 subtypes); rate lowered to 5 / 1 / 0.5% | automatic (scikit-learn) |
| `toxic_conv` | Chung et al. benchmark | 99,583 | toxic, 7.9%; lowered to 1 / 0.5 / 0.1% | automatic (HuggingFace `mteb/toxic_conversations_50k`) |
| `jigsaw_threat`, `jigsaw_severe` | naturally rare | ~160K | threat 0.30%; threat / identity hate / severe toxic | Kaggle *Jigsaw Toxic Comment Classification Challenge*: put `train.csv` (or the zip) in `data/raw/jigsaw/` |
| `lts_shark` | nuanced predicate | ~14.5K ads + ~500 expert labels | real shark / ray / chimaera products | `shark_trophy.csv`, `validation_sharks.csv` from github.com/VIDA-NYU/LTS `data_use_cases/` into `data/raw/lts/` |

Lowering the rate keeps every negative and a subtype-stratified random subset of positives
(a different subset per seed). On `lts_shark` only the expert rows have labels, so samplers buy labels
from a single cached labeling pass of a real LLM (`nhs.llm_label`) and F1 is computed on the expert rows,
which are excluded from sampling.

## Setup

Python 3.11. Install PyTorch for your machine first (CPU build is fine), then:

```bash
pip install -r requirements.txt
```

## Reproduce

```bash
python -m nhs.data.prepare --task 20ng_comp toxic_conv     # add: jigsaw lts_shark once raw files exist
python -m nhs.embed --task 20ng_comp toxic_conv
python -m pytest tests
python -m nhs.run --config configs/main_20ng.yaml
python -m nhs.run --config configs/main_toxic.yaml
python -m nhs.run --config configs/ablation.yaml
python -m nhs.plot --exp main_20ng main_toxic ablation
```

LLM labels for `lts_shark` (needs `OPENAI_API_KEY` or `DEEPSEEK_API_KEY`; costs a few dollars):

```bash
python -m nhs.llm_label --task lts_shark --provider openai --model gpt-4o-mini --limit 50
python -m nhs.llm_label --task lts_shark --provider openai --model gpt-4o-mini
python -m nhs.llm_label --task lts_shark --model gpt-4o-mini --compile
```

Runs are resumable: `nhs.run` skips runs already in `results/<exp>/runs.csv`.

## Outputs

`results/<exp>/runs.csv` (one row per run), `traces.csv` (one row per round), `summary.md`
(mean ± std tables and paired comparisons), `figures/*.png`.

Metrics per run: positives found (and per LLM call), subtypes covered, proxy precision / recall / F1
on rows the sampler did not label, end-to-end F1 (bought labels + proxy), time per round.

## Layout

```
nhs/data/prepare.py   raw data -> data/<task>/{texts.jsonl, y.npy, subtype.npy, meta.json}
nhs/embed.py          sentence embeddings (cached)
nhs/tasks.py          positive-rate control, run context, clusters, k-NN
nhs/samplers.py       all sampling strategies
nhs/labelers.py       label oracle (gold or cached LLM labels, optional label noise)
nhs/proxy.py          logistic-regression proxy
nhs/run.py            experiment grid runner
nhs/plot.py           figures and summary tables
nhs/llm_label.py      one-off LLM labeling with a cache (OpenAI-compatible API)
configs/              experiment grids, keyword lists, prompts
docs/SPEC.md          original build specification
```

## Design decisions and deviations

- **Gold labels as a perfect LLM** on datasets with complete human labels; `flip_rate` in a config
  simulates a weaker labeler. Real LLM labels are used where gold labels are incomplete (`lts_shark`).
- **Clusters are fitted on each (sub)table**, never on the full table with all positives, so the index
  carries no information about removed positives. LDA reuses one vocabulary per dataset.
- **SEALS** starts from labeled positives in the paper; a query has none, so it gets the same seed
  budget as QSB (query Top-K).
- **LTS-LR** follows LTS Algorithm 1 (reward = F1 gain on a validation set, baseline 0.5, old model kept
  when F1 does not improve) with two changes: logistic regression instead of BERT, and the validation set
  is 10% of the budget labeled by the same labeler (LTS used a separate expert set).
- **Keyword lists** are hand-written placeholders; the `lts_shark` prompt follows the labeling rules of
  the LTS code (`labeling.py`) with our own examples.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (Chung et al. use Gecko), so absolute numbers
  differ from theirs; trends are comparable.
