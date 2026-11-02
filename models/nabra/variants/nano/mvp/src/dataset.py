
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .features import (
    FeatureConfig,
    LogMelExtractor,
    read_wav,
)
from .tokenizer import CharTokenizer


class SpeechDataset(Dataset):
    def __init__(self, records, tokenizer, config=None):
        self.records = records
        self.tokenizer = tokenizer
        self.config = config or FeatureConfig()
        self.extractor = LogMelExtractor(self.config)

        if not self.records:
            raise ValueError("No audio records")

        for record in records:
            self.tokenizer.encode(
                record["transcript"]
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        waveform = read_wav(
            record["audio"],
            self.config.sample_rate,
        )

        features = self.extractor(waveform)

        targets = torch.tensor(
            self.tokenizer.encode(
                record["transcript"]
            ),
            dtype=torch.long,
        )

        if len(targets) == 0:
            raise ValueError("Empty target")

        return features, targets, record["transcript"]


def collate_speech(batch):
    features, targets, transcripts = zip(*batch)

    feature_lengths = torch.tensor(
        [len(x) for x in features],
        dtype=torch.long,
    )

    target_lengths = torch.tensor(
        [len(y) for y in targets],
        dtype=torch.long,
    )

    padded_features = pad_sequence(
        features,
        batch_first=True,
        padding_value=0.0,
    )

    flat_targets = torch.cat(targets)

    return {
        "features": padded_features,
        "feature_lengths": feature_lengths,
        "targets": flat_targets,
        "target_lengths": target_lengths,
        "transcripts": transcripts,
    }


def build_datasets(train_records, val_records, config=None):
    tokenizer = CharTokenizer.from_transcripts(
        [r["transcript"] for r in train_records]
    )

    train_dataset = SpeechDataset(
        train_records, tokenizer, config
    )

    val_dataset = SpeechDataset(
        val_records, tokenizer, config
    )

    return train_dataset, val_dataset, tokenizer
