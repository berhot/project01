
import json
from datetime import datetime, timezone

from .dataset import SpeechDataset
from .inference import transcribe
from .paths import get_run_dir


def evaluate_checkpoint(checkpoint_path, device="cpu", validation_source=None):
    """Verify labels/audio and bind an evaluation to the exact checkpoint bytes."""
    import torch
    from .inference import load_checkpoint
    from .ingestion import read_records, sha256_file
    from .runtime import fingerprint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = checkpoint["source_manifest"]["val"]
    records = read_records(validation_source) if validation_source else expected
    if fingerprint(records) != fingerprint(expected):
        raise ValueError("Validation dataset changed since training")
    for record in records:
        if sha256_file(record["audio"]) != record["audio_sha256"]:
            raise ValueError("Validation audio changed since training")
    model, tokenizer, features = load_checkpoint(checkpoint_path, device)
    result = evaluate_asr(model, tokenizer, features, records)
    output = get_run_dir(checkpoint["run_id"]) / "evaluations" / f"epoch_{checkpoint['epoch']:04d}.json"
    digest = sha256_file(checkpoint_path)
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get("checkpoint_sha256") != digest or previous.get("run_id") != checkpoint["run_id"] or previous.get("epoch") != checkpoint["epoch"]:
            raise ValueError("Existing evaluation has mismatched provenance")
        if any(previous.get(key) != value for key, value in result.items()):
            raise ValueError("Existing evaluation differs from fresh results; preserve it and investigate")
    else:
        write_evaluation(checkpoint["run_id"], checkpoint["epoch"], digest, result)
    return result, output


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))

    for i, a in enumerate(reference, 1):
        current = [i] + [0] * len(hypothesis)

        for j, b in enumerate(hypothesis, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (a != b),
            )

        previous = current

    return previous[-1]


def evaluate_asr(
    model,
    tokenizer,
    feature_config,
    records,
):
    total_char_errors = 0
    total_chars = 0
    total_word_errors = 0
    total_words = 0
    predictions = []

    for record in records:
        reference = record["transcript"]

        hypothesis = transcribe(
            model,
            tokenizer,
            feature_config,
            record["audio"],
        )

        total_char_errors += edit_distance(
            list(reference),
            list(hypothesis),
        )
        total_chars += len(reference)

        ref_words = reference.split()
        hyp_words = hypothesis.split()

        total_word_errors += edit_distance(
            ref_words,
            hyp_words,
        )
        total_words += len(ref_words)

        predictions.append({
            "audio": record["audio"],
            "reference": reference,
            "hypothesis": hypothesis,
        })

    if total_chars == 0 or total_words == 0:
        raise ValueError("Insufficient evaluation labels")

    return {
        "cer": total_char_errors / total_chars,
        "wer": total_word_errors / total_words,
        "evaluated_recordings": len(records),
        "predictions": predictions,
    }


def write_evaluation(
    run_id,
    epoch,
    checkpoint_sha256,
    result,
):
    run_dir = get_run_dir(run_id)

    output = (
        run_dir / "evaluations"
        / f"epoch_{epoch:04d}.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output.exists():
        raise FileExistsError(output)

    report = {
        "schema_version": "1.0.0",
        "model_family": "nabra",
        "variant": "nano",
        "run_id": run_id,
        "epoch": epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        **result,
    }

    output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output
