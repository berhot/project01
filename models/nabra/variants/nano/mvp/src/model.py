
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_mels: int = 80
    vocab_size: int = 50
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1


class NabraASR(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.conv = nn.Sequential(
            nn.Conv1d(
                config.n_mels,
                config.hidden_size,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.GELU(),
            nn.Conv1d(
                config.hidden_size,
                config.hidden_size,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.GELU(),
        )

        self.encoder = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                config.dropout
                if config.num_layers > 1
                else 0.0
            ),
        )

        self.output = nn.Linear(
            config.hidden_size * 2,
            config.vocab_size,
        )

    @staticmethod
    def output_lengths(lengths):
        # Both convolution layers use stride 2.
        lengths = (lengths + 1) // 2
        lengths = (lengths + 1) // 2
        return lengths

    def forward(self, features, feature_lengths):
        # features: [B, T, 80]
        x = features.transpose(1, 2)

        x = self.conv(x)

        lengths = self.output_lengths(
            feature_lengths
        )

        x = x.transpose(1, 2)

        # Avoid learning from padded sequence tails.
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        encoded, _ = self.encoder(packed)

        encoded, _ = (
            nn.utils.rnn.pad_packed_sequence(
                encoded,
                batch_first=True,
                total_length=x.size(1),
            )
        )

        logits = self.output(encoded)

        return logits, lengths


def ctc_loss(
    logits,
    output_lengths,
    targets,
    target_lengths,
):
    log_probs = F.log_softmax(
        logits.float(),
        dim=-1,
    ).transpose(0, 1)

    # CPU CTCLoss avoids unsupported MPS
    # kernels on some PyTorch/macOS versions.
    if log_probs.device.type == "mps":
        log_probs = log_probs.cpu()

    loss_fn = nn.CTCLoss(
        blank=0,
        reduction="mean",
        zero_infinity=False,
    )

    return loss_fn(
        log_probs,
        targets.cpu(),
        output_lengths.cpu(),
        target_lengths.cpu(),
    )


def main():
    config = ModelConfig()

    model = NabraASR(config)

    features = torch.randn(2, 200, 80)
    lengths = torch.tensor([200, 160])

    logits, output_lengths = model(
        features, lengths
    )

    print("NABRA — ASR Model")
    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))
    print("Logits:", tuple(logits.shape))
    print("Output lengths:", output_lengths.tolist())
    print("Status: PASS")


if __name__ == "__main__":
    main()
