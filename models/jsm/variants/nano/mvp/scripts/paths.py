
from pathlib import Path
from datetime import datetime, timezone
import re
import uuid


MVP_ROOT = Path(__file__).resolve().parents[1]
BERHOT_ROOT = Path(__file__).resolve().parents[6]

INTAKE_ROOT = BERHOT_ROOT / "intake"
DEFAULT_CORPUS = INTAKE_ROOT / "wikipedia-ar"

ARTIFACTS_ROOT = (
    BERHOT_ROOT
    / "storage"
    / "artifacts"
    / "jsm"
    / "nano"
    / "mvp"
)

RELEASES_ROOT = (
    BERHOT_ROOT
    / "storage"
    / "releases"
    / "jsm"
    / "nano"
)


def new_run_id():
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    return (
        f"nano-{timestamp}-"
        f"{uuid.uuid4().hex[:8]}"
    )


def validate_run_id(run_id):
    if not isinstance(run_id, str):
        raise ValueError("Invalid run ID")

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*",
        run_id,
    ):
        raise ValueError("Invalid run ID")

    return run_id


def get_run_dir(run_id):
    return ARTIFACTS_ROOT / validate_run_id(run_id)


def prepare_run(run_id):
    run_dir = get_run_dir(run_id)

    for folder in (
        "checkpoints",
        "evaluations",
        "logs",
    ):
        (run_dir / folder).mkdir(
            parents=True,
            exist_ok=True,
        )

    return run_dir


def get_release_dir(version):
    if not isinstance(version, str):
        raise ValueError("Invalid release version")

    if not re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+"
        r"(?:-[A-Za-z0-9.-]+)?",
        version,
    ):
        raise ValueError(
            "Expected version such as 0.1.0"
        )

    return RELEASES_ROOT / version


def resolve_corpus(source=None):
    if source is None:
        return DEFAULT_CORPUS

    return Path(source).expanduser().resolve()


def main():
    print("BERHOT ROOT:", BERHOT_ROOT)
    print("MVP ROOT:", MVP_ROOT)
    print("INTAKE:", INTAKE_ROOT)
    print("DEFAULT CORPUS:", DEFAULT_CORPUS)
    print("ARTIFACTS:", ARTIFACTS_ROOT)
    print("RELEASES:", RELEASES_ROOT)


if __name__ == "__main__":
    main()
