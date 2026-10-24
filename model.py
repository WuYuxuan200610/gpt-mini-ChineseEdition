import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.query = nn.Linear(config.n_embd, config.head_size, bias=False)

        self.head_size = config.head_size
        self.dropout = nn.Dropout(config.dropout)

        # 保持 2D 形状，与原版一致，避免广播时多出维度
        self.register_buffer(
            'attention_mask',
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x)
        v = self.value(x)
        q = self.query(x)

        # 缩放点积
        weight = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)

        # 因果掩码 (未来位置填 -inf)
        weight = weight.masked_fill(
            self.attention_mask[:T, :T] == 0,
            float('-inf')
        )

        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        out = weight @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 独立创建多个单头
        self.heads = nn.ModuleList(
            [SingleHeadAttention(config) for _ in range(config.n_head)]
        )
        # 拼接后的统一输出投影
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # 在最后一个维度拼接
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # Pre-LN 残差结构
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )

        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 初始化权重后，进行权重共享 (Weight Tying)
        self.apply(self._init_weights)
        self.token_embedding_table.weight = self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        # 词嵌入 + 位置嵌入
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb

        # 经过多层 Transformer Block
        x = self.blocks(x)
        x = self.ln_final(x)

        # 输出头
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int = None) -> torch.Tensor:
        """带温度和 Top-K 的生成策略。

        Args:
            temperature: 温度值，越小越保守，越大越随机。建议 0.7 - 1.0
            top_k: 只保留概率最高的前 k 个词。建议 50
        """
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.block_size
                else idx[:, -self.block_size:]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab)

            # 1. 应用温度控制 (Temperature)
            if temperature != 1.0:
                logits = logits / temperature

            # 2. 应用 Top-K 采样
            if top_k is not None:
                # 获取前 k 个最大的 logits 值
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # 把小于第 k 个值的 logits 设为负无穷，这样在 softmax 后概率就是 0
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx