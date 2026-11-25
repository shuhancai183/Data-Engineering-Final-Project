# 中文使用说明

## 这个仓库在做什么
语义过滤（`AI.IF`）要对每一行调用 LLM，太贵。常见省钱办法是 proxy：LLM 只标注几百行，用这些标签训练一个逻辑回归，剩下的行交给它判断。正例只占 0.1%–1% 时，随机抽到的几百行里几乎没有正例，proxy 就会失效。

我们比较 6 种“选哪些行给 LLM 标注”的方法（采样器），看哪种在**相同的标注预算**下：
1. 找到的正例更多；
2. 找到的正例种类更全；
3. 训练出的 proxy 在人工标签上 F1 更高；
4. 每轮耗时更短。

## 一次实验是怎么跑的
1. **准备数据**：每行文本 + 人工标签 + 正例种类 + 一句过滤条件的描述（query）。
2. **算 embedding**：每行和 query 都变成 384 维向量，存进缓存。
3. **构造稀有正例的表**：保留所有负例，按种类分层随机保留一部分正例，使正例比例等于设定值。每个随机种子保留的正例不同。然后按标签分层随机留出 30% 作为**测试集**，所有采样器都不能从测试集里选行。
4. **标注循环**：预算 500 行，每轮 25 行，共 20 轮。每轮：采样器选 25 行 → 标注器返回标签 → 采样器更新自己的状态。
   - 标注器默认直接返回人工标签，相当于一个“完美的 LLM”；
   - `flip_rate` 会固定翻转一部分标签，模拟弱一些的 LLM；
   - shark 数据集用的是真实 LLM 事先标好、缓存起来的标签。
5. **训练 proxy**：用这 500 个标签训练逻辑回归（`class_weight='balanced'`，和 Chung 相同）。
6. **评估**：
   - proxy 在**固定测试集**上的 precision / recall / F1，所有方法都在同一批行上打分；
   - 端到端 F1：整张表上，已标注的行用买到的标签，其余用 proxy 的预测，这相当于这条查询最终返回的结果。

   为什么不直接在“没被标注的行”上算 F1？因为越会找正例的方法，会把容易认的正例都标走，剩下的全是难认的，评估就不公平了。我们在 20NG 上就遇到过：SEALS 找到了一半以上的正例，但在剩下的行上 proxy F1 = 0。

## 6 个方法
| 方法 | 一句话 |
|---|---|
| `random` | 随机抽（Chung 的默认做法） |
| `topk` | 抽和 query 最相似的行（Chung Table 10 的 Top-K） |
| `keyword` | 抽包含关键词的行（LTS 的 KBS）；词表在 `configs/keywords/` |
| `seals_mlp` | 只在已标注行的近邻里挑，挑 proxy 认为最像正例的（SEALS） |
| `lts_lr` | LTS 原算法：LDA 分簇 + Thompson Sampling；奖励是“验证集 F1 是否提升”；把 BERT 换成了逻辑回归 |
| `qsb` | 我们的方法：k-means 分簇；离 query 近的簇先验更高；奖励是“抽到正例”；在簇内挑 proxy 分数高或挨着已知正例的行；留 15% 随机 |

QSB 的直觉：把每个簇当成一台老虎机，“拉一次”就是标注簇里的一行，“中奖”就是抽到正例。Thompson Sampling 会多拉中奖率高的机器，但也会时不时试试别的，这样能找到不同种类的正例。

## 文件说明
| 文件 | 作用 |
|---|---|
| `nhs/data/prepare.py` | 下载并整理数据集，转成统一格式，存到 `data/<任务名>/` |
| `nhs/data/lts.py` | 处理 LTS shark 数据（全量广告 + 专家标签） |
| `nhs/embed.py` | 用 MiniLM 算 embedding 并缓存 |
| `nhs/tasks.py` | 控制正例比例、构造每次实验的上下文、分簇、找近邻 |
| `nhs/samplers.py` | 6 个采样器都在这里 |
| `nhs/labelers.py` | 标注器（人工标签或缓存的 LLM 标签，可加噪声） |
| `nhs/proxy.py` | 逻辑回归 proxy |
| `nhs/run.py` | 按配置文件跑整组实验，结果追加写进 CSV；可断点续跑 |
| `nhs/plot.py` | 出图和汇总表 |
| `nhs/llm_label.py` | 用真实 LLM 给整张表打一次标签并缓存（shark 用） |
| `configs/*.yaml` | 实验配置：数据集、正例比例、预算、种子、方法和参数 |
| `tests/test_core.py` | 在合成数据上检查：不超预算、不重复、可复现、采样器不偷看标签 |

## 怎么运行
环境：Python 3.11，先装 torch（CPU 版也可以），再 `pip install -r requirements.txt`。

```bash
python -m nhs.data.prepare --task 20ng_comp toxic_conv
python -m nhs.embed --task 20ng_comp toxic_conv
python -m pytest tests
python -m nhs.run --config configs/main_20ng.yaml
python -m nhs.run --config configs/main_toxic.yaml
python -m nhs.run --config configs/ablation.yaml
python -m nhs.plot --exp main_20ng main_toxic ablation
```

结果在 `results/<实验名>/`：
- `runs.csv`：每次运行一行；
- `traces.csv`：每轮一行，用来画“找到的正例数随标注数增长”的曲线；
- `summary.md`：均值 ± 标准差的表格，以及 QSB 和每个 baseline 按种子配对比较的结果；
- `figures/`：图。

## 加新东西
- **加数据集**：在 `nhs/data/prepare.py` 里加一个 `build_xxx()`，转成统一格式，然后算 embedding。
- **加方法**：在 `nhs/samplers.py` 里写一个类，实现 `setup / select / update`，再注册到 `SIMPLE`。
- **加实验**：复制一个 `configs/*.yaml` 改参数。同一组实验再跑会跳过已完成的部分。

## 还差的两个数据集
- **Jigsaw**：登录 Kaggle，在 *Jigsaw Toxic Comment Classification Challenge* 页面接受规则，下载 `train.csv`（或整个 zip），放进 `data/raw/jigsaw/`，然后运行 `python -m nhs.data.prepare --task jigsaw` 和 `python -m nhs.embed --task jigsaw_threat jigsaw_severe`。
- **LTS shark**：把 LTS 仓库 `data_use_cases/` 下的 `shark_trophy.csv` 和 `validation_sharks.csv` 放进 `data/raw/lts/`，然后：
  1. 运行 `python -m nhs.data.prepare --task lts_shark` 和 `python -m nhs.embed --task lts_shark`；
  2. 设好 API key，按 README 里的命令先试标 50 条，再全量标注，最后 `--compile`；
  3. 在配置里写 `label_source: llm_gpt-4o-mini` 跑实验。

## 常见问题
- **Random 找到的正例数**应该约等于“预算 × 正例比例”。明显偏多说明有 bug。
- **正例 0.1% 时有方法的 F1 接近 1**：几乎可以肯定是标签泄露。
- **同一个种子两次结果不同**：有地方没用传入的随机数生成器。
- **git push 连不上 127.0.0.1:7897**：git 配了代理但代理没开。打开代理，或者用 `git -c http.proxy= -c https.proxy= push` 临时直连。
