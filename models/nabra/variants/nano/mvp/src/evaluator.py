
import json
from datetime import datetime, timezone

from .dataset import SpeechDataset
from .inference import transcribe
from .paths import get_run_dir


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
