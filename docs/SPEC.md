# Build Prompt：Proxy Models for Needle-in-a-Haystack Semantic Filters（Direction B）

> 把本文件整体作为提示词交给 coding agent。你（agent）的任务是**从零搭建本项目的基础代码仓库**，让团队之后可以在上面加数据集、加方法、跑实验。请完整读完再动手。

---

## 0. 你的角色与今天的范围

你是这个课程研究项目的工程负责人。今天的目标是**搭好可复用、可测试、可扩展的实验框架**，并在 20 Newsgroups 上跑通第一组实验。

**今天要做：**
- 仓库骨架、环境、统一数据格式、embedding 缓存
- 标注器抽象（只实现 Oracle 标注器 = 用 gold label 模拟 LLM）
- 全部 6 个采样器（不需要 LLM 就能实现）
- 实验 runner、指标、结果记录、画图、单元测试
- 20 Newsgroups 和 Toxic Conversations 的数据准备；Jigsaw、LTS shark 的数据准备脚本（读取用户手动放置的文件）

**今天明确不做（Non-goals）：**
- ❌ **不调用任何 LLM API**，不需要也不要写入任何 API key。`LLMLabeler` 只写接口和缓存设计，方法体抛 `NotImplementedError`。
- ❌ 不做 Chung et al. 的复现实验（数据准备可以做，但不跑复现）。
- ❌ 不自动下载 Kaggle 数据（需要用户登录并接受比赛规则）；不自动从 Google Drive 下载。
- ❌ 不做任何人工标注。
- ❌ 不实现原版 LTS 的 BERT 训练。

---

## 1. 项目背景（必须理解，代码设计以此为准）

### 1.1 问题
数据库支持 `WHERE AI.IF('the ad sells a real shark product', description)` 这种语义过滤：对每一行调用 LLM 判断真假。百万行时成本太高。常见捷径（**proxy 模型**）：
1. 对每行算 embedding；
2. 让 LLM 标注一小部分样本；
3. 在这些标签上训练一个便宜的分类器（逻辑回归），即 proxy；
4. 用 proxy 判断剩下所有行。

当正例比例只有 **0.1%–5%**（大海捞针：走私广告、威胁言论、相关论文）时，随机抽 500 行可能只有个位数正例，而且都是同一种，proxy 就会失败。

### 1.2 两篇核心论文（以下事实已对照原文核实，不要自行发挥）
**Chung et al., SIGMOD 2026（arXiv 2603.15970）——BigQuery/AlloyDB 中的 proxy 模型**
- 语义过滤默认用**随机抽样**选训练样本（§5.3）。
- Proxy = embedding 上的 **LogisticRegression，sklearn 默认参数 + `class_weight='balanced'`**（§4）。
- 回退规则：proxy 在训练样本上（以 LLM 标签为准）的 accuracy 与 LLM 相差不超过 τ（例如 0.1）就用 proxy，否则回退到 LLM（§4.3–4.4）。
- Table 10 还比较了 **Top-K（按 embedding 相似度预筛样本）** 和分层主动学习，后者比随机慢 51,639 倍。
- 在 SciFact / FiQA / SCIDOCS 这种每个查询只有 1.1–4.9 个相关文档的场景，他们称训练“近乎不可能”，直接回退 LLM（§5.3）。

**Barbosa et al., SIGMOD 2025（arXiv 2504.21211）——LTS（Learn to Sample），野生动物广告**
- 聚类用 **LDA 主题模型**（代码中 K=5 shark / 10 leather / 10 animal），只聚一次。
- **Thompson Sampling**：每个簇 θ ~ Beta(α+wins, β+losses)，代码默认 α=β=0.5，衰减 δ=0.99（每轮 wins/losses 都乘 δ）。
- **奖励不是“抽到正例”**，而是：本轮重训后的模型在**验证集**上的 F1 是否超过 baseline（初始 0.5）；超过则 reward=1 并把 baseline 更新为该 F1，否则 reward=0 且保留旧模型。
- 簇内选行：第一轮随机；之后用当前模型预测簇内行，**最多一半选预测为正的，其余选预测为负的，两组内部随机抽**。
- 每轮 200 条，10 轮 = 2000 条标注；每轮重训 BERT；标注器在论文中是 GPT-4，代码里是 gpt-4o-mini。
- Baseline：随机（RS）、关键词（KBS，人工词表，如 leather 约 226 个产品词 + 250 个动物名）、主动学习 ACTL（coreset）。
- 文本 F1（Table 2，RS / KBS / ACTL / LTS）：Animal 0.745/0.685/0.663/0.816；Leather 0.465/0.776/0.822/0.855；Shark 0.774/0.780/0.924/0.872。
- 每个任务约 500 条专家标签用于评估。

**SEALS, Coleman et al., AAAI 2022（arXiv 2007.00077）**
- 不是独立选样规则，而是**包装器**：候选池 = 已标注样本各自的 k 个最近邻（k=100）之并集，再在池内套用基础策略。基础策略之一 **MLP = most-likely positive**（选 proxy 预测为正概率最高的）。
- 模型：冻结 embedding 上的逻辑回归。初始 5 个正例 + 95 个随机负例，每轮 100 条。

### 1.3 研究空白与我们的方向（Direction B：Finding positives cheaply）
LTS 能找到稀有正例但慢（每轮训 BERT、离线跑数小时）；Chung 快但在稀有正例下失效。**我们要做的是：在固定 LLM 标注预算下，设计一个采样器，快速找到更多、更多样的正例，训出更好的 proxy。**

**三个假设**
- H1：正例 ≤1% 时，QSB 每次标注找到的正例数明显多于 Random 和 Top-K。
- H2：QSB 覆盖的正例子类型更多，最终 proxy 在 gold label 上的 F1 更高。
- H3：QSB 每轮“选样 + 训练”耗时 <1 秒（毫秒级）。

**三个核心指标**（老师指定）：每次 LLM 调用找到的正例数、找到的正例有多少种不同类型、最终 F1（以人写的 gold label 为准）。

---

## 2. 我们的方法：QSB（Query-Seeded Bandit）

| 步骤 | 做法 | 与已有工作的区别 |
|---|---|---|
| 预处理 | 全表 embedding（MiniLM），MiniBatchKMeans 聚成 K 簇（默认 K=50）；查询语句算 embedding `q` | LTS 用 LDA |
| ① 种子阶段（默认占预算 10%） | 以 `q`（以及可选的合成正例 embedding，今天没有）为种子，按与种子的最大余弦相似度排序，**每个簇最多取 `seed_per_cluster_cap` 条**，贪心选够 | 比 Chung Top-K 多了按簇限量 |
| ② 簇级 bandit（主体） | 每簇 Beta 先验由簇中心与 `q` 的相似度决定；**奖励 = 抽到的行是正例**；簇内按“逻辑回归正例概率 + kNN 加分（是否是已找到正例的近邻）”打分选最高 | LTS 奖励是验证集 F1 提升；kNN 加分来自 SEALS |
| ③ 随机保留（默认占预算 15%） | 每轮按比例混入纯随机样本 | 修正主动采样带来的负例偏差 |
| ④ 最终 proxy | 用所有已标注行训练 `LogisticRegression(class_weight='balanced')` | 与 Chung 一致 |

### 2.1 QSB 精确规格（按此实现，参数全部进 config）
- **先验**：`s_c = cos(centroid_c, q)`；`p_c = (s_c - min s) / (max s - min s + 1e-12)`；`α_c = a0 + λ·p_c`，`β_c = b0 + λ·(1 - p_c)`。默认 `a0=1, b0=1, λ=prior_strength=5`。`use_query_prior=False` 时 `α_c=a0, β_c=b0`。
- **种子阶段**：`n_seed = round(seed_frac * budget)`；按 `max_j cos(x, seed_j)` 降序遍历，簇内已选数 < `seed_per_cluster_cap`（默认 3）才选。标注后更新对应簇的 α/β（正例 α+=1，负例 β+=1）。
- **每轮**（batch 大小 `batch_size`，默认 25）：
  - `n_rand = round(random_frac * k)` 个槽位给纯随机（未标注行中均匀抽）。
  - 其余槽位逐个选：对所有未耗尽的簇各抽 θ_c ~ Beta(α_c, β_c)，取 argmax 簇；在该簇未标注、本轮未选的行中，按 `score = p_LR(x) + γ·1[x ∈ kNN(已找到的正例)]` 选最高（`γ=knn_bonus` 默认 0.5；`use_knn_bonus=False` 时 γ=0）。若当前没有可用的 LR（标注中只有一个类别），`p_LR` 用 `cos(x, q)` 代替。
  - 标注 → 更新簇的 α/β（reward=是否为正例）→ 重训 LR（每轮一次）。
  - 可选衰减 `decay`（默认 1.0 即不衰减）。
- **消融开关**（必须全部可配置）：`use_query_prior`、`use_synthetic_seeds`（今天无文件时自动为 False）、`use_knn_bonus`、`random_frac`、`cluster_method`（`kmeans` / `lda`）、`reward`（`positive` / `f1_gain`）。
- **工程要求**：QSB 与 LTS-LR 共用同一个 `ClusterBandit` 组件，这样“从 LTS-LR 逐项改到 QSB”的消融只需要改 config。

---

## 3. 对比方法（全部基于同一 embedding、同一预算、同一最终 proxy）

| 名称 | 规格 |
|---|---|
| `random` | 未标注行中均匀无放回抽样（Chung 默认） |
| `topk` | 按 `cos(x, q)` 降序取下一批未标注行（Chung Table 10 的 Top-K，作为采样器在整个预算中使用） |
| `keyword` | 关键词表从 `configs/keywords/<task>.txt` 读取（**今天是人工占位词表**，meta 里记录 `keywords_source: manual_placeholder`；最终版改为 LLM 生成）。文本含任一关键词（大小写不敏感、按词边界）的行组成候选池，池内均匀抽；池耗尽后退化为随机 |
| `seals_mlp` | 种子与 QSB 相同（query Top-K，`seed_frac`）。之后每轮：候选池 = 所有已标注行的 kNN（k=100）之并集减去已标注行；用 LR 正例概率选最高的 k 个；LR 不可用时用 `cos(x,q)`；池空时退化为随机 |
| `lts_lr` | 按 LTS 原算法：`cluster_method=lda`（sklearn `LatentDirichletAllocation` + `CountVectorizer`，K=10，可配）；Thompson α=β=0.5、decay=0.99；**每轮只选一个簇**取整批；簇内第一轮随机，之后用 LR 预测，≤k/2 取预测为正（随机），其余取预测为负（随机）；`reward=f1_gain`：重训后的 LR 在**验证集**上的 F1 > baseline（初始 0.5）则 reward=1 并更新 baseline，否则 reward=0 且保留旧模型。**验证集**：开局随机抽 `val_frac`（默认 0.1）× 预算的行，标注计入预算。 |
| `qsb` | 第 2 节 |

---

## 4. 数据集

### 4.1 统一格式（所有数据集都必须转成这个样子）
```
data/<task_name>/
  texts.parquet     # 列：row_id (int), text (str)
  y.npy             # int8, 1=正例, 0=负例, -1=无 gold（仅 LTS 类数据集会出现）
  subtype.npy       # int16, 正例的子类型编号; 负例和未知为 -1; 没有子类型的数据集正例全为 0
  meta.json         # 见下
  emb_<model>.npy       # float32, L2 归一化, (N, d)
  query_emb_<model>.npy # float32, L2 归一化, (d,)
  knn100_<model>.npy    # int32, (N, 100), 每行余弦最近邻（不含自身）
  clusters_kmeans<K>_<model>.npy / clusters_lda<K>.npy  # int32, (N,)
```
`meta.json` 字段：`task_name, source, query (过滤条件的自然语言描述), positive_definition, subtype_names (list), n_rows, n_pos, base_rate, has_full_gold (bool), gold_eval_rows (仅 LTS: 有专家标签的 row_id 列表文件名), keywords_source, notes`。

### 4.2 任务列表

| task_name | 来源 | 正例定义 / 子类型 | query | 今天状态 |
|---|---|---|---|---|
| `20ng_comp` | `sklearn.datasets.fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))`，去掉去空白后少于 20 个字符的文本 | `comp.*` 共 5 个子组 = 5 个子类型 | "The post discusses computers: computer graphics, operating systems, PC or Mac hardware, or the X Window System." | **必须完成并跑实验** |
| `toxic_conv` | HuggingFace MTEB Toxic Conversations（预期 id `mteb/toxic_conversations_50k`；**请先查看 dataset card 确认 id、字段和 split**，合并全部 split，去重） | label=1 为正例；无子类型 | "The comment is toxic: rude, disrespectful, insulting, or hateful." | 完成数据准备和 embedding；**在 dataset card 中确认数据来源（据说取自 Civil Comments），把结论写进 meta.notes** |
| `jigsaw_threat` | Kaggle *Jigsaw Toxic Comment Classification Challenge* 的 `train.csv`，用户手动放到 `data/raw/jigsaw/train.csv`（列：id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate） | threat=1（天然约 0.30%）；无子类型 | "The comment contains a threat of violence or harm against someone." | 写好脚本；文件不存在时打印下载说明并正常退出 |
| `jigsaw_severe` | 同上 | threat ∨ identity_hate ∨ severe_toxic；子类型按优先级 threat > identity_hate > severe_toxic 取第一个命中的 | "The comment contains a threat, identity-based hate, or severely toxic language." | 同上 |
| `lts_shark` | LTS 仓库 `VIDA-NYU/LTS` 的 `data_use_cases/shark_trophy.csv`（全量广告）和 `validation_sharks.csv`（约 500 条专家标签），用户手动放到 `data/raw/lts/` | 专家标签中的正例；无子类型；`has_full_gold=false` | "The ad sells a real shark product (e.g., shark jaws, teeth, fins, skin). Exclude fossils, replicas, and non-shark items." | 写好脚本：**先读文件检查列名后再写适配逻辑**；检查验证集和全量数据的重叠并按文本去重；**专家标注的行从采样池中剔除，只用于评估**。文件不存在时打印说明并退出 |

### 4.3 正例比例控制
`make_task(task, pos_rate, seed) -> 子集行索引`：保留全部负例，把正例下采样到使 `n_pos/(n_pos+n_neg) ≈ pos_rate`；下采样按子类型分层（每个子类型按原比例，且尽量每个子类型至少保留 1 条）；`pos_rate=None` 表示原始比例。若目标比例高于原始比例则报错。同一 `(task, pos_rate, seed)` 结果必须确定。

### 4.4 Embedding
- 模型：`sentence-transformers/all-MiniLM-L6-v2`（d=384），`max_seq_length=256`，`normalize_embeddings=True`；模型名可配置，文件名中带模型短名。
- 设备自动选择：CUDA 可用用 GPU（fp16），否则 CPU。机器是 RTX 3050 Laptop 4GB，batch size 默认 64，OOM 时自动减半重试。
- kNN：用归一化 embedding 做分块矩阵乘（torch 在 GPU 上或 numpy 在 CPU 上），每块 2048 行，取 top-101 后去掉自身，存 int32。16 万行也要能跑。
- 都要缓存，已存在就跳过（提供 `--force`）。

---

## 5. 标注器（Labeler）

```python
class Labeler(ABC):
    def label(self, task_rows: np.ndarray) -> np.ndarray: ...   # 返回 0/1
    calls: int   # 累计标注调用次数（= 标注的行数）
```
- `OracleLabeler(y_gold, flip_rate=0.0, seed)`：返回 gold label；`flip_rate>0` 时按比例随机翻转，用于模拟弱 LLM。gold 为 -1 的行调用时直接报错（所以 `lts_shark` 不能用 Oracle）。
- `LLMLabeler`：**只写接口、docstring 和缓存设计**，方法体 `raise NotImplementedError`。缓存设计：`cache/llm_labels/<task>/<model>__<prompt_sha1[:8]>.jsonl`，每行 `{row_id, label, raw_response, ts}`；同一行只调用一次，不同采样器共享缓存。
- **预算记账**：runner 保证标注总数 == budget，任何采样器都不能超；生成合成种子、关键词表这类额外 LLM 调用记在 `extra_llm_calls` 字段（今天恒为 0）。

---

## 6. 核心接口

```python
@dataclass
class TaskContext:
    name: str
    emb: np.ndarray          # (N, d) 子集后的 embedding
    query_emb: np.ndarray    # (d,)
    texts: list[str] | None  # keyword 需要
    knn: np.ndarray          # (N, 100) 子集内重新映射后的近邻（见下）
    clusters: dict[str, np.ndarray]
    subtype: np.ndarray      # 只给评估用，采样器禁止读取
    pool_mask: np.ndarray    # 可被采样的行（lts_shark 中剔除专家评估行）
    config: dict

@dataclass
class LabelState:
    labeled_idx: list[int]
    labels: list[int]
    budget: int
    used: int

class Sampler(ABC):
    name: str
    def setup(self, ctx: TaskContext, rng: np.random.Generator) -> None: ...
    def select(self, state: LabelState, k: int) -> np.ndarray: ...   # 返回 k 个不重复、未标注、在 pool_mask 内的索引
    def update(self, idx: np.ndarray, labels: np.ndarray) -> None: ...
```
- **kNN 子集重映射**：全量 kNN 在子集上要过滤掉不在子集里的邻居；实现一个函数把全量 kNN 映射成子集内索引（不足 100 个时用 -1 填充），按 (task, pos_rate, seed) 缓存在内存中。
- **信息隔离**：采样器只能通过 Labeler 得到标签，不能读取 `y` 或 `subtype`。写一个测试来检查这一点（例如给 ctx 传入打乱后的 y，采样结果应该不变）。
- 共用组件：`proxy.py`（训练和预测 LR，处理单一类别的情况）、`bandit.py`（`ClusterBandit`）、`seeding.py`（按簇限量的 query Top-K 种子）。

### 6.1 运行循环（runner）
```
ctx = build_context(task, pos_rate, seed)
sampler.setup(ctx, rng); labeler = OracleLabeler(...)
while state.used < budget:
    k = min(batch_size, budget - state.used)
    t0; idx = sampler.select(state, k); t_select
    labels = labeler.label(idx); state 更新
    sampler.update(idx, labels); t_update
    记录 trace 行：round, used, cum_pos_found, cum_subtypes_covered, t_select_ms, t_update_ms
最终：在 labeled 上训练 LR(class_weight='balanced', max_iter=1000)，计算指标
```

---

## 7. 指标（`metrics.py`）

每次运行一行（`runs.csv`）：
- 配置：`exp, task, pos_rate, n_rows, n_pos, budget, batch_size, method, method_params_json, seed, embed_model, git_commit`
- `pos_found`：已标注中的正例数（以 labeler 返回的标签为准；Oracle 时等于 gold）
- `pos_per_call = pos_found / budget`
- `subtypes_covered`、`n_subtypes`、`subtype_coverage = covered / n_subtypes`（按 gold subtype 统计已找到的正例）
- `proxy_f1 / proxy_precision / proxy_recall`：proxy 在**未标注的行**上对 gold 的指标（`lts_shark`：只在专家评估行上算）
- `e2e_f1`：端到端查询结果，即已标注行用标注值、其余用 proxy 预测，对全体 gold 计算 F1（`lts_shark` 不计算）
- `proxy_trained`（bool，标注中是否两个类别都有）；单一类别时 proxy 全部预测为 0
- `t_select_ms_mean, t_update_ms_mean, t_total_s, t_final_train_ms`
- `extra_llm_calls`

每轮一行（`traces.csv`）：`run_id, round, used, cum_pos_found, cum_subtypes_covered, t_select_ms, t_update_ms`。

注意：正例比例很低时 F1 方差很大，所以必须多个 seed，并同时报告 precision 和 recall。

---

## 8. 实验配置与命令行

YAML 配置，例如 `configs/mvp_20ng.yaml`：
```yaml
exp: mvp_20ng
embed_model: sentence-transformers/all-MiniLM-L6-v2
tasks: [20ng_comp]
pos_rates: [0.05, 0.01, 0.005]
budgets: [500]
batch_size: 25
seeds: [0, 1, 2]
methods:
  random: {}
  topk: {}
  keyword: {}
  seals_mlp: {seed_frac: 0.1, knn_k: 100}
  lts_lr: {cluster_method: lda, n_clusters: 10, alpha: 0.5, beta: 0.5, decay: 0.99, val_frac: 0.1, reward: f1_gain}
  qsb: {cluster_method: kmeans, n_clusters: 50, seed_frac: 0.1, seed_per_cluster_cap: 3,
        prior_strength: 5, a0: 1, b0: 1, use_query_prior: true, use_knn_bonus: true,
        knn_bonus: 0.5, random_frac: 0.15, reward: positive, decay: 1.0}
```
另外写 `configs/ablation_20ng.yaml`：`qsb` 的几个变体（`qsb_noprior`、`qsb_noknn`、`qsb_norand`、`qsb_lda`、`qsb_f1gain`），方法名可以写成 `name: {base: qsb, ...覆盖参数}`。

命令行：
```bash
python -m nhs.data.prepare --task 20ng_comp
python -m nhs.embed --task 20ng_comp            # 同时生成 emb、query_emb、knn、clusters
python -m nhs.run --config configs/mvp_20ng.yaml   # 自动展开网格；已完成的运行跳过（断点续跑）
python -m nhs.plot --exp mvp_20ng
```
输出：`results/<exp>/runs.csv`、`traces.csv`、`config_snapshot.yaml`、`figures/*.png`。

**画图**（`nhs.plot`）：
1. `f1_vs_posrate.png`：横轴 pos_rate（对数），纵轴 proxy_f1 的均值 ± 标准差，每个方法一条线
2. `posfound_vs_labels.png`：按 traces 画 cum_pos_found 随 used 变化（每个 pos_rate 一个子图）
3. `subtype_coverage.png`：各方法的子类型覆盖率柱状图
4. `timing.png`：每轮选样耗时（对数轴）
同时输出 `summary.md`：按 task × pos_rate × method 汇总均值 ± 标准差的表格。

---

## 9. 仓库结构

```
proxy-needle/
  README.md               # 项目简介、安装、数据放置说明、命令、目录说明
  pyproject.toml          # 包名 nhs，可编辑安装 pip install -e .
  requirements.txt
  .gitignore              # data/ cache/ .venv/ __pycache__/ *.npy；results/ 下只提交 csv、md、png
  configs/
    mvp_20ng.yaml
    ablation_20ng.yaml
    keywords/20ng_comp.txt  toxic_conv.txt  jigsaw_threat.txt  jigsaw_severe.txt  lts_shark.txt
  src/nhs/
    __init__.py
    paths.py              # 统一的数据和缓存路径
    data/
      prepare.py          # 命令行入口，分派到各数据集
      newsgroups.py  toxic_conv.py  jigsaw.py  lts.py
      tasks.py            # make_task（正例比例控制）、build_context
    embed.py              # embedding、kNN、聚类缓存
    labelers.py           # Labeler / OracleLabeler / LLMLabeler(stub)
    proxy.py
    bandit.py
    seeding.py
    samplers/
      __init__.py         # 注册表: name -> class
      base.py  random.py  topk.py  keyword.py  seals.py  lts_lr.py  qsb.py
    metrics.py
    run.py
    plot.py
  tests/
    conftest.py           # 合成玩具数据集：高斯簇，正例 1%，3 个子类型，N=3000
    test_tasks.py         # 比例控制正确、确定性、子类型分层
    test_samplers.py      # 每个采样器：不超预算、无重复、在 pool_mask 内、同 seed 结果相同、不偷看 y
    test_metrics.py
    test_runner.py        # 在玩具数据上端到端跑通所有方法
  results/
  data/                   # 被 gitignore
```

---

## 10. 环境（Windows）

- 机器：Windows 11，Python 3.11.9，GPU 是 NVIDIA RTX 3050 Laptop 4GB；已装 numpy 2.2.3、pandas 2.2.3、scikit-learn 1.5.1、matplotlib 3.9.1。
- 在仓库内建 `.venv`，`requirements.txt` 包含：`numpy pandas scikit-learn matplotlib pyyaml pyarrow tqdm sentence-transformers datasets pytest`。torch 单独安装，在 README 写清楚：先用 `nvidia-smi` 看 CUDA 版本，再从 PyTorch 官方 wheel 源安装对应的 CUDA 版本；失败就装 CPU 版（20NG 在 CPU 上算 embedding 也只要几分钟）。
- 路径一律用 `pathlib`；命令要同时兼容 PowerShell 和 Git Bash。
- 全部随机性都通过 `np.random.default_rng(seed)` 传递，禁止使用全局随机状态。

---

## 11. 执行顺序与验收标准（按里程碑提交）

**M0 骨架**：目录、pyproject、requirements、README、.gitignore、`git init`。→ 提交
**M1 数据 + embedding**：`20ng_comp` 准备、embedding、kNN、K-means/LDA 聚类缓存；`make_task`。打印各任务的 n_rows、n_pos、base_rate、子类型分布。→ 提交
**M2 框架**：Labeler、Sampler 基类、`random`、proxy、runner、metrics、玩具数据测试。`pytest` 全部通过。→ 提交
**M3 QSB**：`bandit.py`、`seeding.py`、`qsb`；在 20NG 上与 random 对比。→ 提交
**M4 其他 baseline**：`topk`、`keyword`（占位词表）、`seals_mlp`、`lts_lr`；测试覆盖。→ 提交
**M5 实验 + 画图**：跑 `configs/mvp_20ng.yaml` 和 `configs/ablation_20ng.yaml`，生成图和 `summary.md`。→ 提交
**M6 其他数据集脚本**：`toxic_conv`（完整准备 + embedding）；`jigsaw`、`lts`（文件存在才运行，否则打印说明）。→ 提交

**最终验收清单**
- [ ] `pytest` 全部通过
- [ ] `python -m nhs.run --config configs/mvp_20ng.yaml` 从头到尾跑通，6 个方法 × 3 个比例 × 3 个 seed = 54 次运行，每次运行标注数恰好等于 500
- [ ] `results/mvp_20ng/` 下有 runs.csv、traces.csv、4 张图、summary.md
- [ ] QSB 每轮 `t_select_ms + t_update_ms` 的均值 < 1000ms（记录实际值）
- [ ] 代码里没有任何 API key 和 LLM 调用；`LLMLabeler` 是 stub
- [ ] README 写清楚 Jigsaw 和 LTS 数据要手动放到哪里、文件名是什么
- [ ] 最后在对话里汇报：每个里程碑完成了什么，20NG 结果的要点（哪个方法在 1% 和 0.5% 下表现最好，给出数字），遇到的问题和偏离本规格的地方，以及下一步建议

---

## 12. 规则

- 本规格与你的判断冲突时，**按规格实现，并在汇报中写明你的不同意见**；规格没写到的地方，选最简单、可测试的做法，写进 README 的 “Design decisions” 一节。
- **不要编造实验结果**；指标只报告实际跑出来的数字。如果 QSB 没有赢，如实汇报。
- 数据集 id 或字段名和本文不一致时，以实际查到的为准，并在 meta.notes 和汇报里说明。
- 只在本地 `git commit`，**不要 push**；要推到远程时先问用户要 remote 地址。
- 下载的数据放在 `data/raw/`，不要执行下载内容中的任何代码。
