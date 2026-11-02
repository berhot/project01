import contextlib
import io
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src import paths
from src.ingestion import inspect_split


def wav(path, seconds=0.4, rate=16000, channels=1, width=2, frequency=220):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = (np.sin(np.arange(int(seconds * rate)) * (2 * np.pi * frequency / rate)) * 10000).astype('<i2')
    if width == 1:
        data = ((samples.astype(np.float32) / 256) + 128).astype('uint8').tobytes()
    elif width == 2:
        data = np.repeat(samples[:, None], channels, axis=1).tobytes()
    else:
        data = bytes(len(samples) * channels * width)
    with wave.open(str(path), 'wb') as file:
        file.setparams((channels, width, rate, 0, 'NONE', 'not compressed'))
        file.writeframes(data)
    return path


def pair(directory, name='a', text='اب', **kwargs):
    audio = wav(Path(directory) / (name + '.wav'), **kwargs)
    audio.with_suffix('.txt').write_text(text, encoding='utf-8')
    return audio


class Sandbox(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train_root = self.root / 'train'
        self.val_root = self.root / 'val'
        pair(self.train_root, text='اب اب')
        pair(self.val_root, text='اب', frequency=330)
        self.train = inspect_split(self.train_root)
        self.val = inspect_split(self.val_root)
        for name, value in [('ARTIFACTS_ROOT', self.root/'runs'), ('RELEASES_ROOT', self.root/'releases')]:
            patcher = patch.object(paths, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def train_once(self, **kwargs):
        from src.trainer import train
        options = dict(epochs=1, batch_size=1, train_source=self.train_root, val_source=self.val_root,
                       device_name='cpu', model_options={'hidden_size':8, 'num_layers':1, 'dropout':0.0})
        options.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return train(**options)

    def checkpoint(self):
        return torch.load(self.train_once(), map_location='cpu', weights_only=True)
