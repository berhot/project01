
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import wave

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0

    def __post_init__(self):
        if any(not isinstance(x, int) or x < 1 for x in
               (self.sample_rate, self.n_fft, self.hop_length, self.n_mels)):
            raise ValueError("Feature dimensions must be positive integers")
        if not 0 <= self.f_min < self.f_max <= self.sample_rate / 2:
            raise ValueError("Invalid frequency bounds")


def read_wav(path, sample_rate=16000):
    path = Path(path)

    with wave.open(str(path), "rb") as wav:
        if (
            wav.getnchannels() != 1
            or wav.getframerate() != sample_rate
            or wav.getsampwidth() != 2
            or wav.getcomptype() != "NONE"
        ):
            raise ValueError(
                "Expected 16 kHz mono PCM16 WAV"
            )

        frames = wav.getnframes()
        raw = wav.readframes(frames)
        if len(raw) != frames * 2:
            raise ValueError("Truncated WAV payload")

    audio = np.frombuffer(
        raw, dtype="<i2"
    ).astype(np.float32) / 32768.0

    if len(audio) == 0:
        raise ValueError("Empty audio")

    return torch.from_numpy(audio.copy())


def hz_to_mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(config):
    n_freq = config.n_fft // 2 + 1

    low = hz_to_mel(config.f_min)
    high = hz_to_mel(config.f_max)

    mel_points = torch.linspace(
        low, high, config.n_mels + 2
    )

    hz_points = torch.tensor([
        mel_to_hz(float(x))
        for x in mel_points
    ])

    frequencies = torch.linspace(
        0,
        config.sample_rate / 2,
        n_freq,
    )

    filters = torch.zeros(
        config.n_mels, n_freq
    )

    for i in range(config.n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        rising = (
            (frequencies - left)
            / (center - left).clamp_min(1e-8)
        )

        falling = (
            (right - frequencies)
            / (right - center).clamp_min(1e-8)
        )

        filters[i] = torch.minimum(
            rising, falling
        ).clamp(min=0)

    return filters


class LogMelExtractor:
    def __init__(self, config=None):
        self.config = config or FeatureConfig()

        self.window = torch.hann_window(
            self.config.n_fft
        )

        self.mel_filters = create_mel_filterbank(
            self.config
        )

    def __call__(self, waveform):
        config = self.config

        waveform = waveform.float()
        if waveform.ndim != 1 or waveform.numel() == 0 or not torch.isfinite(waveform).all():
            raise ValueError("Expected finite nonempty mono waveform")
        if len(waveform) <= config.n_fft // 2:
            waveform = F.pad(waveform, (0, config.n_fft // 2 + 1 - len(waveform)))

        # Centered STFT, fixed preprocessing.
        spectrum = torch.stft(
            waveform,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )

        power = spectrum.abs().square()

        mel = self.mel_filters @ power

        log_mel = torch.log(
            mel.clamp_min(1e-10)
        ).transpose(0, 1)

        # Per-recording normalization.
        mean = log_mel.mean(dim=0, keepdim=True)
        std = log_mel.std(
            dim=0,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-5)

        return (log_mel - mean) / std

    def to_dict(self):
        return asdict(self.config)


def main():
    from .paths import TRAIN_ROOT

    files = sorted(TRAIN_ROOT.rglob("*.wav"))

    if not files:
        raise FileNotFoundError(
            f"No WAV files in {TRAIN_ROOT}"
        )

    extractor = LogMelExtractor()

    waveform = read_wav(files[0])
    features = extractor(waveform)

    print("NABRA — Acoustic Features")
    print("File:", files[0])
    print("Waveform samples:", len(waveform))
    print("Features:", tuple(features.shape))
    print("Status: PASS")


if __name__ == "__main__":
    main()
