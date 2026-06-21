import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        embed_dim=256,
        num_heads=1,
        num_layers=1,
        mlp_ratio=4,
        mlp_num_layers=2,
        dropout=0.0,
        causal=True,
    ):
        super().__init__()
        if num_layers != 1:
            raise ValueError("minimal_model.Transformer only supports num_layers=1")
        if mlp_num_layers < 1:
            raise ValueError("mlp_num_layers must be at least 1")
        if not causal:
            raise ValueError("minimal_model.Transformer only supports causal=True")

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.init_std = embed_dim**-0.5
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        hidden_dim = mlp_ratio * embed_dim
        if mlp_num_layers == 1:
            layers = [nn.Linear(embed_dim, vocab_size)]
        else:
            layers = [
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            for _ in range(mlp_num_layers - 2):
                layers.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )
            layers.append(nn.Linear(hidden_dim, vocab_size))

        self.mlp = nn.Sequential(*layers)
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

        x = self.token_embedding(input_ids)

        if attention_mask is None:
            counts = torch.arange(
                1,
                seq_len + 1,
                device=input_ids.device,
                dtype=x.dtype,
            )
            counts = counts.view(1, seq_len, 1)
            x = x.cumsum(dim=1) / counts
        else:
            mask = attention_mask[:, :, None].to(dtype=x.dtype)
            x = x * mask
            counts = mask.cumsum(dim=1).clamp_min(1.0)
            x = x.cumsum(dim=1) / counts

        logits = self.mlp(x)

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
        mlp_num_layers=3,
        dropout=0.1,
    )

    example_input = torch.randint(0, 128, (2, 16))
    output = model(example_input)
    print(output["logits"].shape)
