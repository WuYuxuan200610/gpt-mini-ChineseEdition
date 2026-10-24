import re
import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("assets", exist_ok=True)
log_path = 'results/train_log_100k.txt'

steps, losses, lrs = [], [], []
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        match = re.search(r'step\s+(\d+)/\d+\s+loss=([\d\.]+)\s+lr=([\d\.e-]+)', line)
        if match:
            steps.append(int(match.group(1)))
            losses.append(float(match.group(2)))
            lrs.append(float(match.group(3)))

plt.style.use('dark_background')
window = 50

# 1. Loss
plt.figure(figsize=(10, 5))
plt.plot(steps, losses, color='#ff7f0e', alpha=0.3, label='Raw Loss')
if len(losses) > window:
    smooth_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
    plt.plot(steps[window-1:], smooth_loss, color='#ff7f0e', linewidth=2, label='Smoothed Loss')
plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
plt.grid(color='gray', linestyle='--', alpha=0.5)
plt.legend()
plt.savefig('assets/loss_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. LR
plt.figure(figsize=(10, 4))
plt.plot(steps, lrs, color='#2ca02c', linewidth=2)
plt.title('Learning Rate (Warmup + Cosine Decay)', fontsize=14, fontweight='bold')
plt.grid(color='gray', linestyle='--', alpha=0.5)
plt.savefig('assets/lr_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. PPL (修复：使用对数 Y 轴)
ppls = [np.exp(l) for l in losses]
plt.figure(figsize=(10, 5))
plt.plot(steps, ppls, color='#9467bd', alpha=0.3, label='Raw PPL')
if len(ppls) > window:
    smooth_ppl = np.convolve(ppls, np.ones(window)/window, mode='valid')
    plt.plot(steps[window-1:], smooth_ppl, color='#9467bd', linewidth=2, label='Smoothed PPL')
plt.yscale('log') # 核心修复：改为对数坐标
plt.title('Perplexity Curve (Log Scale, Lower is Better)', fontsize=14, fontweight='bold')
plt.grid(color='gray', linestyle='--', alpha=0.5)
plt.legend()
plt.savefig('assets/ppl_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ assets/ppl_curve.png")