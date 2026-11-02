
import argparse

import torch

from .ingestion import load_text
from .paths import DEFAULT_CORPUS
from .tokenizer import CharTokenizer


class TextDataset:

    def __init__(
        self,
        file_path=DEFAULT_CORPUS,
        block_size=64,
        train_ratio=0.9,
    ):
        if block_size < 1:
            raise ValueError(
                "block_size must be positive"
            )

        if not 0 < train_ratio < 1:
            raise ValueError(
                "train_ratio must be between 0 and 1"
            )

        self.block_size = block_size
        self.source = str(file_path)
        self.train_ratio = train_ratio

        text = load_text(file_path)

        self.tokenizer = CharTokenizer.from_text(
            text
        )

        token_ids = self.tokenizer.encode(text)

        data = torch.tensor(
            token_ids,
            dtype=torch.long,
        )

        split_index = int(
            len(data) * train_ratio
        )

        self.train_data = data[:split_index]
        self.val_data = data[split_index:]

        minimum = block_size + 1

        if (
            len(self.train_data) < minimum
            or len(self.val_data) < minimum
        ):
            raise ValueError(
                "Corpus is too small for both splits "
                "at the selected context length"
            )

    def get_batch(
        self,
        split,
        batch_size=8,
        device="cpu",
    ):
        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive"
            )

        if split == "train":
            data = self.train_data

        elif split == "val":
            data = self.val_data

        else:
            raise ValueError(
                "Split must be train or val"
            )

        max_start = len(data) - self.block_size

        starts = torch.randint(
            low=0,
            high=max_start,
            size=(batch_size,),
        ).tolist()

        x = torch.stack([
            data[i:i + self.block_size]
            for i in starts
        ])

        y = torch.stack([
            data[i + 1:i + self.block_size + 1]
            for i in starts
        ])

        return x.to(device), y.to(device)

    def get_validation_windows(
        self,
        batch_size=8,
    ):
        """
        Fixed, non-overlapping input windows
        for repeatable checkpoint evaluation.
        """

        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive"
            )

        data = self.val_data
        length = self.block_size

        starts = list(
            range(
                0,
                len(data) - length,
                length,
            )
        )

        for offset in range(
            0,
            len(starts),
            batch_size,
        ):
            selected = starts[
                offset:offset + batch_size
            ]

            x = torch.stack([
                data[i:i + length]
                for i in selected
            ])

            y = torch.stack([
                data[i + 1:i + length + 1]
                for i in selected
            ])

            yield x, y


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default=str(DEFAULT_CORPUS),
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    dataset = TextDataset(
        file_path=args.source,
        block_size=args.block_size,
    )

    x, y = dataset.get_batch(
        "train",
        batch_size=8,
    )

    assert torch.equal(
        x[:, 1:],
        y[:, :-1],
    )

    print("JSM Nano MVP — Dataset")
    print("----------------------")
    print("Vocabulary:", dataset.tokenizer.vocab_size)
    print("Train tokens:", len(dataset.train_data))
    print("Val tokens:", len(dataset.val_data))
    print("X shape:", tuple(x.shape))
    print("Y shape:", tuple(y.shape))
    print("Status: PASS")


if __name__ == "__main__":
    main()
