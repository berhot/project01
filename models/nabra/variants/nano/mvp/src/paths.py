
from pathlib import Path
from datetime import datetime, timezone
import uuid
import re


MVP_ROOT = Path(__file__).resolve().parents[1]

BERHOT_ROOT = MVP_ROOT.parents[4]

INTAKE_ROOT = BERHOT_ROOT / "intake"

DATA_ROOT = INTAKE_ROOT / "speech"

TRAIN_ROOT = DATA_ROOT / "train"
VAL_ROOT = DATA_ROOT / "val"

ARTIFACTS_ROOT = (
    BERHOT_ROOT
    / "storage"
    / "artifacts"
    / "nabra"
    / "nano"
    / "mvp"
)

RELEASES_ROOT = (
    BERHOT_ROOT
    / "storage"
    / "releases"
    / "nabra"
    / "nano"
)


def new_run_id():
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    suffix = uuid.uuid4().hex[:8]

    return f"nabra-{timestamp}-{suffix}"


def validate_run_id(run_id):
    if not isinstance(run_id, str):
        raise ValueError("Run ID must be a string")

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*",
        run_id,
    ):
        raise ValueError(
            f"Invalid run ID: {run_id}"
        )

    return run_id


def get_run_dir(run_id):
    return (
        ARTIFACTS_ROOT
        / validate_run_id(run_id)
    )


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
        raise ValueError(
            "Release version must be a string"
        )

    if not re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+"
        r"(?:-[A-Za-z0-9.-]+)?",
        version,
    ):
        raise ValueError(
            "Invalid release version. "
            "Example: 0.1.0"
        )

    return RELEASES_ROOT / version


def resolve_data_path(source=None):
    if source is None:
        return DATA_ROOT

    return Path(source).expanduser().resolve()


def main():
    print("NABRA Nano MVP — Paths")
    print("----------------------")
    print("BERHOT:", BERHOT_ROOT)
    print("MVP:", MVP_ROOT)
    print("INTAKE:", INTAKE_ROOT)
    print("SPEECH:", DATA_ROOT)
    print("TRAIN:", TRAIN_ROOT)
    print("VALIDATION:", VAL_ROOT)
    print("ARTIFACTS:", ARTIFACTS_ROOT)
    print("RELEASES:", RELEASES_ROOT)
    print("Status: PASS")


if __name__ == "__main__":
    main()
