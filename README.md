# Search Agent

围绕搜索 Agent 的证据利用、重复搜索和停止决策，完成从检索框架改造到小模型强化学习训练的工程实践。

**文档检索 → 段落级检索 → 证据判别 → Search-R1 → AutoSearch**

## 核心实现

- **检索与 Agent**：共享 ReAct 循环，支持文档/段落级 BM25、原文访问及完整轨迹记录。
- **证据判别**：接入本地模型、DeepSeek、JEV，提供阅读优先级建议与证据充分性判断。
- **Search-R1**：使用 veRL + GRPO + LoRA 训练 4B 模型学习多轮搜索与回答。
- **AutoSearch**：加入分轮回答探测、质量增益和搜索效率奖励，复用训练与评测入口。

```text
agent/          ReAct 循环与判别器
search_r1/      GRPO 训练、工具、数据转换与最终答案奖励
autosearch/     分轮回答探测与过程奖励
retrieval.py    文档/段落索引与检索服务
evaluate.py     统一答案与搜索次数评测
```

## 结果

**检索框架：BrowseComp-Plus，27B 主模型，同批 30 题。**

| 配置 | 答对 |
|---|---:|
| 文档检索 | 12/30 |
| 段落检索 | 23/30 |
| 段落检索 + 本地 9B 判别 | 23/30 |
| 段落检索 + DeepSeek 判别 | 20/30 |
| 段落检索 + JEV 判别 | 20/30 |

段落检索改善了证据可见性；额外判别仍未稳定解决重复搜索和答案不收敛的问题。

**模型训练：4B、LoRA rank 8、16K 上下文；训练集为 NQ/HotpotQA 各 2500 题，固定评测集为 NQ、HotpotQA、Bamboogle、TriviaQA 各 25 题。**

| 模型 | 准确率 | 平均搜索次数 |
|---|---:|---:|
| 基模 | 33% | 1.99 |
| Search-R1，450 步 | **46%** | **1.28** |
| AutoSearch，350 步 | 45% | 1.65 |
| AutoSearch，500 步 | 44% | 1.53 |

模型训练采用最终短答案 EM；检索框架使用模型辅助判分，两组结果分别展示。

## 使用

### 1. 检索与 Agent

Linux 环境需要 Java 17。模型使用 OpenAI 兼容接口；把 `.env.example` 复制为 `.env` 后填写自己的配置。

```bash
pip install -e ".[retrieval]"
python retrieval.py index --corpus data/corpus.jsonl --output data/passages
python retrieval.py serve --index data/passages
```

语料格式为每行 `{"id":"doc1","contents":"文档正文"}`；索引时加 `--mode document` 可切换为文档级检索。已有 Lucene 索引可直接传给 `serve --index`，外置语料库用 `--corpus-db` 指定。

另开终端运行 Agent：

```bash
python -m agent.core --question "Which country contains Paris?" --output runs/demo
python -m agent.core --questions data/eval100.jsonl --decision jev --output runs/jev
```

`--decision` 支持 `none / local / deepseek / jev`；本地与 DeepSeek 后端通过 `DECISION_*` 配置。

### 2. 模型训练

训练环境使用 Linux、Python 3.12、PyTorch 2.13 + CUDA 13、vLLM 0.29，实测单卡 RTX 6000D。先准备匹配的 PyTorch/vLLM 环境，再安装训练依赖和固定版本 veRL：

```bash
pip install -e ".[train]"
pip install git+https://github.com/volcengine/verl.git@dc709afb7a74fd2645640a4e5f2fdf40166b4fbb
```

将基模放到 `MODEL_PATH`，准备 `data/train5000.jsonl` 和 `data/eval100.jsonl`，每行格式：

```json
{"id":"nq:1","question":"问题","answers":["答案"],"data_source":"nq"}
```

```bash
python -m search_r1.data
python -m search_r1.train --mode eval --base --output runs/base
python -m search_r1.train --method search-r1 --output runs/search-r1
python -m search_r1.train --method autosearch --output runs/autosearch
```

默认训练至 500 步，在 20 步及每 50 步评测并保存结果。`--resume <checkpoint目录>` 恢复权重、优化器和采样进度；`--dry-run` 只生成配置。AutoSearch 探测仅在训练阶段启用。

### 3. 独立评测

```bash
python evaluate.py --questions data/eval100.jsonl \
  --trajectories runs/jev/trajectories --output runs/jev/summary.json
```

训练、评测和轨迹输出均保存在各自 `runs/` 下，数据、权重和真实密钥不纳入仓库。

## 参考

[Search Agent](https://github.com/liuqi6777/search_agent) · [Search-R1](https://github.com/PeterGriffinJin/Search-R1) · [AutoSearch](https://github.com/bofusun/AutoSearch) · [veRL](https://github.com/volcengine/verl)

相关方法的核心代码经整理适配，许可见 `LICENSE`。
