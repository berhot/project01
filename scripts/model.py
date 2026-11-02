
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 256
    block_size: int = 64
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Learned Query, Key and Value projections
        self.qkv = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
        )

        self.proj = nn.Linear(
            config.n_embd,
            config.n_embd,
        )

    def forward(self, x):
        B, T, C = x.shape

        # Produce Q, K, V
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Split embeddings across attention heads
        q = q.view(
            B, T, self.n_head, self.head_size
        ).transpose(1, 2)

        k = k.view(
            B, T, self.n_head, self.head_size
        ).transpose(1, 2)

        v = v.view(
            B, T, self.n_head, self.head_size
        ).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=(
                self.dropout if self.training else 0.0
            ),
            is_causal=True,
        )

        # Recombine attention heads
        y = y.transpose(1, 2).contiguous()
        y = y.view(B, T, C)

        return self.proj(y)


class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x):

        # Attention with residual connection
        x = x + self.attn(self.ln1(x))

        # MLP with residual connection
        x = x + self.mlp(self.ln2(x))

        return x


class JSMModel(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd,
        )

        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd,
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(config)
                for _ in range(config.n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        # Initialize parameters from scratch
        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError(
                "Input exceeds model context length"
            )

        # Token and position embeddings
        token_emb = self.token_embedding(idx)

        positions = torch.arange(
            T,
            device=idx.device,
        )

        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb

        # Transformer layers
        x = self.blocks(x)

        # Final normalization
        x = self.ln_f(x)

        # Predict vocabulary logits
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, V = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T),
            )

        return logits, loss


def main():

    config = ModelConfig()

    model = JSMModel(config)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    # Create a sample input tensor for architecture testing
    x = torch.randint(
        0,
        config.vocab_size,
        (2, config.block_size),
    )

    logits, _ = model(x)

    assert logits.shape == (
        2,
        config.block_size,
        config.vocab_size,
    )

    print("JSM Mini — Transformer")
    print("----------------------")
    print(f"Parameters:   {total_params:,}")
    print(f"Layers:       {config.n_layer}")
    print(f"Heads:        {config.n_head}")
    print(f"Embedding:    {config.n_embd}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print("Status: PASS")


if __name__ == "__main__":
    main()
