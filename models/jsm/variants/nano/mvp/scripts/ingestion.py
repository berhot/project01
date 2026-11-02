
import argparse
import hashlib
import json

from pathlib import Path

from .paths import DEFAULT_CORPUS


def discover_text_files(source):
    source = Path(source).expanduser().resolve()

    if source.is_file():
        if source.suffix.lower() != ".txt":
            raise ValueError("Expected a .txt file")
        return [source]

    if not source.is_dir():
        raise FileNotFoundError(
            f"Corpus not found: {source}"
        )

    files = sorted(
        (
            path
            for path in source.rglob("*")
            if path.is_file()
            and path.suffix.lower() == ".txt"
        ),
        key=lambda path: (
            path.relative_to(source).as_posix()
        ),
    )

    if not files:
        raise ValueError(
            f"No .txt files found in {source}"
        )

    return files


def load_documents(source):
    source = Path(source).expanduser().resolve()

    files = discover_text_files(source)

    documents = []
    manifest = []

    for path in files:
        raw = path.read_bytes()

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Invalid UTF-8: {path}"
            ) from error

        if not text.strip():
            raise ValueError(
                f"Empty document: {path}"
            )

        relative_name = (
            path.relative_to(source).as_posix()
            if source.is_dir()
            else path.name
        )

        documents.append(text)

        manifest.append({
            "path": relative_name,
            "bytes": len(raw),
            "characters": len(text),
            "sha256": hashlib.sha256(
                raw
            ).hexdigest(),
        })

    return documents, manifest


def load_text(source=DEFAULT_CORPUS):
    documents, _ = load_documents(source)

    # Explicit separator between documents.
    return "\n\n".join(documents)


def corpus_sha256(source=DEFAULT_CORPUS):
    text = load_text(source)

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def inspect_text(text):
    encoded = text.encode("utf-8")

    return {
        "characters": len(text),
        "bytes": len(encoded),
        "lines": len(text.splitlines()),
        "sha256": hashlib.sha256(
            encoded
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default=str(DEFAULT_CORPUS),
    )

    args = parser.parse_args()

    documents, manifest = load_documents(
        args.source
    )

    text = "\n\n".join(documents)
    info = inspect_text(text)

    print("JSM Nano MVP — Ingestion")
    print("------------------------")
    print("Source:", Path(args.source).resolve())
    print("Documents:", len(documents))
    print("Characters:", info["characters"])
    print("Bytes:", info["bytes"])
    print("Lines:", info["lines"])
    print("SHA256:", info["sha256"])
    print("Status: PASS")

    print("\nSource manifest:")
    print(json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
