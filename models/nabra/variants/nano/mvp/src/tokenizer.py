
import json


class CharTokenizer:
    TYPE = "nabra-char-ctc-v1"
    BLANK_ID = 0

    def __init__(self, chars):
        self.chars = sorted(set(chars))

        if not self.chars:
            raise ValueError("Empty vocabulary")

        self.stoi = {
            char: index + 1
            for index, char in enumerate(self.chars)
        }

        self.itos = {
            index: char
            for char, index in self.stoi.items()
        }

        self.vocab_size = len(self.chars) + 1

    @classmethod
    def from_transcripts(cls, transcripts):
        text = "".join(transcripts)
        return cls(list(text))

    def encode(self, text):
        ids = []

        for char in text:
            if char not in self.stoi:
                raise ValueError(
                    f"Unknown transcript character: {char!r}"
                )

            ids.append(self.stoi[char])

        return ids

    def decode(self, ids):
        return "".join(
            self.itos[int(i)]
            for i in ids
            if int(i) != self.BLANK_ID
        )

    def ctc_decode(self, ids):
        result = []
        previous = None

        for token in ids:
            token = int(token)

            if (
                token != self.BLANK_ID
                and token != previous
            ):
                result.append(token)

            previous = token

        return self.decode(result)

    def to_dict(self):
        return {
            "type": self.TYPE,
            "chars": self.chars,
            "vocab_size": self.vocab_size,
            "blank_id": self.BLANK_ID,
        }

    @classmethod
    def from_dict(cls, data):
        if data["type"] != cls.TYPE:
            raise ValueError("Unsupported tokenizer")

        if data.get("blank_id") != 0:
            raise ValueError("Invalid blank ID")

        if len(data["chars"]) != len(set(data["chars"])):
            raise ValueError("Duplicate characters")

        tokenizer = cls(data["chars"])

        if tokenizer.vocab_size != data["vocab_size"]:
            raise ValueError("Vocabulary mismatch")

        return tokenizer
