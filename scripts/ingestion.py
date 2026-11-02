
from pathlib import Path
import hashlib


def load_text(file_path: str) -> str:
    """
    Load real UTF-8 text from disk.
    No synthetic data or fallback examples.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Training file not found: {path}"
        )

    text = path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError(
            "Training text is empty."
        )

    return text


def inspect_text(text: str) -> dict:
    """
    Inspect the text before tokenization.
    """

    encoded = text.encode("utf-8")

    return {
        "characters": len(text),
        "bytes": len(encoded),
        "lines": len(text.splitlines()),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def main():
    text = load_text("data/saudi.txt")
    info = inspect_text(text)

    print("JSM Mini — Data Ingestion")
    print("-------------------------")
    print(f"Characters: {info['characters']}")
    print(f"Bytes:      {info['bytes']}")
    print(f"Lines:      {info['lines']}")
    print(f"SHA256:     {info['sha256']}")
    print("Status: PASS")


if __name__ == "__main__":
    main()
