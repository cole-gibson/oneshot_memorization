import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, causal=True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None, return_attention=False):
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        if self.causal:
            causal_mask = torch.ones(
                seq_len, seq_len, dtype=torch.bool, device=x.device
            ).tril()
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~mask, float("-inf"))

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        out = attention @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.embed_dim)
        out = self.proj(out)

        if return_attention:
            return out, attention
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_hidden_layers=1, dropout=0.0):
        super().__init__()
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be nonnegative")

        if num_hidden_layers == 0:
            self.net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Dropout(dropout),
            )
            return

        layers = []
        for i in range(num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(embed_dim if i == 0 else hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )

        layers.extend(
            [
                nn.Linear(hidden_dim, embed_dim),
                nn.Dropout(dropout),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4,
        mlp_num_layers=2,
        dropout=0.0,
        causal=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=mlp_ratio * embed_dim,
            num_hidden_layers=mlp_num_layers,
            dropout=dropout,
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attention(self.norm1(x), attention_mask=attention_mask)
        x = x + self.feed_forward(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        mlp_ratio=4,
        mlp_num_layers=2,
        dropout=0.0,
        causal=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.init_std = embed_dim**-0.5
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    mlp_num_layers=mlp_num_layers,
                    dropout=dropout,
                    causal=causal,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.max_seq_len = max_seq_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)

    def forward(self, input_ids, attention_mask=None, targets=None):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError("input sequence is longer than max_seq_len")

        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return {"logits": logits, "loss": loss}


if __name__ == "__main__":
    model = Transformer(
        vocab_size=128,
        max_seq_len=64,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        mlp_num_layers=2,
        dropout=0.1,
    )

    example_input = torch.randint(0, 128, (2, 16))
    output = model(example_input)
    print(output["logits"].shape)
