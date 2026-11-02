
import argparse

from .ingestion import load_text
from .paths import DEFAULT_CORPUS


class CharTokenizer:
    TYPE = "char-v1"

    def __init__(self, chars):
        self.chars = sorted(set(chars))

        if not self.chars:
            raise ValueError("Empty vocabulary")

        self.stoi = {
            char: index
            for index, char in enumerate(self.chars)
        }

        self.itos = {
            index: char
            for char, index in self.stoi.items()
        }

        self.vocab_size = len(self.chars)

    @classmethod
    def from_text(cls, text):
        return cls(list(text))

    def encode(self, text):
        token_ids = []

        for char in text:
            if char not in self.stoi:
                raise ValueError(
                    f"Unknown character: {char!r}"
                )

            token_ids.append(self.stoi[char])

        return token_ids

    def decode(self, token_ids):
        return "".join(
            self.itos[int(token)]
            for token in token_ids
        )

    def to_dict(self):
        return {
            "type": self.TYPE,
            "chars": self.chars,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("type") != cls.TYPE:
            raise ValueError(
                "Unsupported tokenizer type"
            )

        chars = data["chars"]

        if len(chars) != len(set(chars)):
            raise ValueError(
                "Duplicate vocabulary characters"
            )

        tokenizer = cls(chars)

        if tokenizer.vocab_size != data["vocab_size"]:
            raise ValueError(
                "Vocabulary size mismatch"
            )

        return tokenizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default=str(DEFAULT_CORPUS),
    )

    args = parser.parse_args()

    text = load_text(args.source)

    tokenizer = CharTokenizer.from_text(text)

    tokens = tokenizer.encode(text)
    restored = tokenizer.decode(tokens)

    assert restored == text

    print("JSM Nano MVP — Tokenizer")
    print("------------------------")
    print("Type:", tokenizer.TYPE)
    print("Vocabulary:", tokenizer.vocab_size)
    print("Characters:", len(text))
    print("Tokens:", len(tokens))
    print("Reversible:", restored == text)
    print("Status: PASS")


if __name__ == "__main__":
    main()
