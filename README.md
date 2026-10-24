# gpt-mini (Chinese Edition)

A minimal GPT-2 style language model written from scratch.
**面向中文语料的基于GPT-2架构大语言模型**

本项目通过 PyTorch 代码实现了一个大体基于 GPT-2 架构的小型语言模型（SLM），旨在帮助学习者深入理解 Transformer 的内部机制（因果自注意力、Pre-LN 残差结构、词嵌入共享等）以及完整的训练闭环：从中文文本读取、BPE 分词、切块、Next-token 预测到模型权重的保存与推理。

> **⚠️ 声明**：这是一个**教学基线项目 (Learning Baseline)**，并非生产可用的大模型。它可以帮助你跑通原理，但受限于数据量（默认仅 1000 行中文语料），模型极易过拟合，无法生成具有逻辑的长文本。

## ✨ 核心特性 (Features)

- **Pre-LN Transformer Blocks**: 采用现代大模型广泛使用的预归一化残差架构。
- **Multi-Head Causal Self-Attention**: 将多头注意力拆解为 `n_head` 个独立的单头注意力组合，代码可读性极高。
- **Weight Tying**: 输入 Token Embedding 与输出 LM Head 共享权重，大幅减少参数量。
- **中文数据 Pipeline**: 提供从中文 JSONL 语料到 `tiktoken` 分词、定长切块的完整数据处理流。
- **Reproducible**: 固定了全局随机种子，保证训练过程可复现。

## 📁 项目结构 (Project Structure)

```text
gpt-mini/
├── config.py        # GPTConfig (模型架构的超参数)
├── model.py         # 核心网络结构 (Attention / FFN / Block / GPT)
├── dataset.py       # 数据处理类 (JSONL -> Token 流 -> 定长切块)
├── train.py         # 完整的训练循环、验证评估与 Checkpoint 保存
├── generate.py      # 加载权重，输入中文 Prompt 进行文本生成
├── utils.py         # 通用工具 (set_seed / get_device 等)
├── requirements.txt # 依赖项
├── data/            # 存放训练数据 (不提交至 Git)
└── checkpoints/     # 存放训练输出的模型权重 (不提交至 Git)
🛠️ 环境准备 (Setup)
bash
# 安装依赖
pip install -r requirements.txt

# 预热分词器 (首次运行 tiktoken 会下载约 1MB 的 GPT-2 BPE 词表)
python -c "import tiktoken; tiktoken.get_encoding('gpt2')"
📚 中文数据集 (Data)
本项目期待的数据格式为 JSONL（每行一个 JSON 对象，包含 text 字段）：

json
{"id": "1", "text": "今天天气很好，我们一起去公园散步。"}
这里使用的数据源：中文维基百科
推荐使用 HuggingFace 上的 rfikd/wikipedia-zh-cn 数据集。

由于完整数据集高达 2.1GB，而我们仅需要验证代码跑通，建议在本地截取前 1000 行，再上传到云服务器，节省传输时间：

bash
# 截取前 1000 行 (仅几 MB)
head -n 1000 wikipedia-zh-cn-20240901.json > data/wiki_1000.jsonl
🚀 训练指南 (Train)
第 1 步：冒烟测试 (Smoke Test)
在消耗大量算力前，可以先用极小的模型配置和数据跑通流程：

bash
python train.py \
  --data_path data/wiki_1000.jsonl \
  --max_lines 100 \
  --block_size 64 \
  --batch_size 4 \
  --n_layer 2 --n_head 2 --n_embd 128 \
  --epochs 1 \
  --out_dir checkpoints_smoke
如果终端成功打印 loss 并在 checkpoints_smoke/ 下生成了 .pt 文件，说明环境没问题。
预期输出示例：

text
Device: cpu
Chunks: total=21522, train=19370, val=2152
Parameters: 6.84M
[epoch 0] step    0/4843 loss=10.8378 lr=6.00e-06
[epoch 0] step  100/4843 loss=6.8965 lr=3.00e-04
[epoch 0] step  200/4843 loss=5.1409 lr=3.00e-04
...
== epoch 0 | train=3.52 | val=3.48 | 340.5s
Saved checkpoints_smoke/model_epoch_0.pt

第 2 步：正式训练 (Real Training)
使用完整的 1000 行数据进行训练（可根据显存调整 batch_size）：

bash
python train.py \
  --data_path data/wiki_1000.jsonl \
  --max_lines 1000 \
  --block_size 512 \
  --batch_size 12 \
  --n_layer 6 --n_head 6 --n_embd 384 \
  --epochs 2 \
  --lr 3e-4 \
  --out_dir checkpoints
训练完成后，checkpoints/ 目录下会生成每个 Epoch 的权重以及验证集 Loss 最低的 best.pt。

💬 生成测试 (Generate)
使用训练好的模型进行中文文本续写：

bash
python generate.py \
  --ckpt checkpoints/best.pt \
  --prompt "人工智能是" \
  --max_new_tokens 100
(提示：由于训练数据极少，模型生成的内容可能会重复或缺乏连贯逻辑，这属于正常现象。)

⚠️ 注意事项 (Caveats)
过拟合 (Overfitting)：1000 行中文数据产生的 Token 数远不足以喂饱 120M 参数量的模型。训练 Loss 会显著下降，但验证 Loss 可能会反弹。这主要是为了跑通教学流程。

无 KV Cache：generate.py 每次推理都会重新计算完整上下文。适合短文本生成，长文本推理较慢。

分词效率：项目使用 GPT-2 BPE 分词器，它对中文的切分粒度较细（一个汉字会被切为多个 Byte-level Token）。在真实中文大模型中，通常会使用针对中文优化的 SentencePiece 或扩充词表。

📄 License
MIT License

