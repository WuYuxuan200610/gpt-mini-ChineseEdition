import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from config import GPTConfig
from model import GPT

os.makedirs("assets", exist_ok=True)
plt.style.use('dark_background')

device = "cpu"
ckpt = torch.load("checkpoints/best_model_100k.pt", map_location=device)
config = GPTConfig(**ckpt["config"])
model = GPT(config).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# ==========================================
# 图 5：模型各层权重参数分布直方图
# ==========================================
plt.figure(figsize=(10, 5))
# 提取所有 Linear 层的权重
all_weights = []
for name, param in model.named_parameters():
    if "weight" in name and len(param.shape) >= 2:
        all_weights.extend(param.detach().cpu().numpy().flatten())

# 只取一部分以防内存爆炸
all_weights = np.array(all_weights[:100000])
plt.hist(all_weights, bins=100, color='#17becf', alpha=0.7, edgecolor='black')
plt.title('Weight Distribution (Sample)', fontsize=14, fontweight='bold')
plt.xlabel('Weight Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(color='gray', linestyle='--', alpha=0.5)
plt.savefig('assets/weight_hist.png', dpi=300, bbox_inches='tight')
plt.close()


# ==========================================
# 图 6：词向量 (Token Embedding) PCA 降维可视化
# ==========================================
# 提取前 200 个 token 的 embedding 进行降维
emb_matrix = model.token_embedding_table.weight[:200].detach().cpu().numpy()
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(emb_matrix)

plt.figure(figsize=(8, 6))
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.6, color='#e377c2', edgecolors='white')
plt.title('Token Embeddings PCA Projection (First 200 tokens)', fontsize=14, fontweight='bold')
plt.xlabel('PCA Dimension 1', fontsize=12)
plt.ylabel('PCA Dimension 2', fontsize=12)
plt.grid(color='gray', linestyle='--', alpha=0.5)
plt.savefig('assets/embedding_pca.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ assets/weight_hist.png, assets/embedding_pca.png")