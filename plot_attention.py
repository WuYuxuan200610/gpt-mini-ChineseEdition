import torch
import matplotlib.pyplot as plt
import seaborn as sns
import tiktoken
import numpy as np
import os
import matplotlib.font_manager as fm
from config import GPTConfig
from model import GPT

# 强制加载微软雅黑字体
font_path = 'C:/Windows/Fonts/msyh.ttc'
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs("assets", exist_ok=True)

device = "cpu"
# 确保模型路径正确
ckpt = torch.load("checkpoints/best_model_100k.pt", map_location=device)
config = GPTConfig(**ckpt["config"])
model = GPT(config).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

text = "北京是中国的首都"
enc = tiktoken.get_encoding("gpt2")

# 核心逻辑：把句子拆成字，并记录每个字对应的 Token ID 索引
chars = list(text)
tokens = []
char_to_token_indices = []

for c in chars:
    c_tokens = enc.encode(c)
    start_idx = len(tokens)
    tokens.extend(c_tokens)
    char_to_token_indices.append(list(range(start_idx, len(tokens))))

idx = torch.tensor([tokens], dtype=torch.long, device=device)

# 截获底层字节级的注意力矩阵
attention_weights = []
def hook_fn(module, input, output):
    x = input[0]
    B, T, C = x.size()
    q, k = module.query(x), module.key(x)
    weight = q @ k.transpose(-2, -1) / (module.head_size ** 0.5)
    weight = weight.masked_fill(module.attention_mask[:T, :T] == 0, float('-inf'))
    weight = torch.softmax(weight, dim=-1)
    attention_weights.append(weight.detach().cpu()[0])

model.blocks[0].att.heads[0].register_forward_hook(hook_fn)
with torch.no_grad():
    model(idx)

raw_att = attention_weights[0].numpy() # 形状: (Total_Tokens, Total_Tokens)

# 将字节级的注意力聚合成 字级别的注意力
num_chars = len(chars)
char_att = np.zeros((num_chars, num_chars))
for i, q_indices in enumerate(char_to_token_indices):
    for j, k_indices in enumerate(char_to_token_indices):
        char_att[i, j] = raw_att[np.ix_(q_indices, k_indices)].mean()

# 绘制高逼格热力图
plt.style.use('default')
plt.figure(figsize=(9, 7))
ax = sns.heatmap(
    char_att,
    xticklabels=chars,
    yticklabels=chars,
    cmap="YlGnBu",
    annot=True,
    fmt=".2f",
    cbar_kws={'label': 'Attention Weight'}
)

# 强制给坐标轴应用中文字体
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(font_prop)
    label.set_fontsize(14)

# 核心修复：给标题也应用中文字体
plt.title("Word-level Attention Map (Layer 0, Head 0)\nInput: '北京是中国的首都'",
          fontsize=15, fontweight='bold', pad=15, fontproperties=font_prop)

plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig("assets/attention_map.png", dpi=300, bbox_inches='tight')
print("✅ assets/attention_map.png")