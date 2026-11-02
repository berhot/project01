
import torch

from scripts.ingestion import load_text
from scripts.tokenizer import CharTokenizer


class TextDataset:

    def __init__(
        self,
        file_path: str,
        block_size: int = 64,
        train_ratio: float = 0.9,
    ):
        if block_size < 1:
            raise ValueError("block_size must be positive")

        if not 0 < train_ratio < 1:
            raise ValueError("Invalid train_ratio")

        self.block_size = block_size

        text = load_text(file_path)

        # Learn vocabulary from training corpus.
        self.tokenizer = CharTokenizer.from_text(text)

        token_ids = self.tokenizer.encode(text)

        data = torch.tensor(
            token_ids,
            dtype=torch.long,
        )

        split_index = int(len(data) * train_ratio)

        self.train_data = data[:split_index]
        self.val_data = data[split_index:]

        minimum = block_size + 1

        if (
            len(self.train_data) < minimum
            or len(self.val_data) < minimum
        ):
            raise ValueError(
                "Dataset is too small for this context length"
            )

    def get_batch(
        self,
        split: str,
        batch_size: int = 8,
        device: str = "cpu",
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        if split == "train":
            data = self.train_data
        elif split == "val":
            data = self.val_data
        else:
            raise ValueError("Unknown dataset split")

        max_start = len(data) - self.block_size

        starts = torch.randint(
            0,
            max_start,
            (batch_size,),
        )

        x = torch.stack([
            data[i:i + self.block_size]
            for i in starts.tolist()
        ])

        y = torch.stack([
            data[i + 1:i + self.block_size + 1]
            for i in starts.tolist()
        ])

        return x.to(device), y.to(device)


def main():
    dataset = TextDataset(
        "data/saudi.txt",
        block_size=64,
    )

    x, y = dataset.get_batch(
        split="train",
        batch_size=8,
    )

    assert torch.equal(x[:, 1:], y[:, :-1])

    print("JSM Dataset")
    print("----------------")
    print("Vocabulary:", dataset.tokenizer.vocab_size)
    print("Train tokens:", len(dataset.train_data))
    print("Val tokens:", len(dataset.val_data))
    print("X shape:", tuple(x.shape))
    print("Y shape:", tuple(y.shape))
    print("Status: PASS")


if __name__ == "__main__":
    main()
