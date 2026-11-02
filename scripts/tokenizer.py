
from scripts.ingestion import load_text


class CharTokenizer:
    """
    Character-level tokenizer.

    Learns its vocabulary from real training text.
    One Unicode character = one token.
    """

    def __init__(self, chars: list[str]):
        self.chars = sorted(set(chars))

        self.stoi = {
            char: i
            for i, char in enumerate(self.chars)
        }

        self.itos = {
            i: char
            for char, i in self.stoi.items()
        }

        self.vocab_size = len(self.chars)

    @classmethod
    def from_text(cls, text: str):
        return cls(list(text))

    def encode(self, text: str) -> list[int]:
        tokens = []

        for char in text:
            if char not in self.stoi:
                raise ValueError(
                    f"Character {char!r} is not in vocabulary"
                )

            tokens.append(self.stoi[char])

        return tokens

    def decode(self, token_ids: list[int]) -> str:
        return "".join(
            self.itos[int(token)]
            for token in token_ids
        )


def main():
    text = load_text("data/saudi.txt")

    tokenizer = CharTokenizer.from_text(text)

    tokens = tokenizer.encode(text)
    restored = tokenizer.decode(tokens)

    assert restored == text

    print("JSM Character Tokenizer")
    print("-----------------------")
    print("Vocabulary:", tokenizer.vocab_size)
    print("Characters:", len(text))
    print("Tokens:", len(tokens))
    print("First 20:", tokens[:20])
    print("Reversible:", restored == text)
    print("Status: PASS")


if __name__ == "__main__":
    main()
