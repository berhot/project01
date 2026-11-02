import torch
from src.features import FeatureConfig, LogMelExtractor, read_wav
from tests.helpers import Sandbox, wav


class FeatureTests(Sandbox):
    def test_read_dtype_and_shape(self):
        x = read_wav(self.train_root/'a.wav')
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(x.shape, (6400,))

    def test_shape_finite_and_stable(self):
        x = read_wav(self.train_root/'a.wav'); extractor = LogMelExtractor()
        a = extractor(x); b = extractor(x)
        self.assertEqual(a.shape, (41,80))
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.equal(a,b))

    def test_short_long_and_empty(self):
        for length in (1, 100, 16000*21):
            with self.subTest(length=length):
                y = LogMelExtractor()(torch.zeros(length))
                self.assertTrue(torch.isfinite(y).all())
        for x in (torch.tensor([]), torch.tensor([float('nan')])):
            with self.assertRaises(ValueError): LogMelExtractor()(x)

    def test_invalid_config(self):
        for args in ({'hop_length':0}, {'f_max':9000}, {'n_mels':0}):
            with self.assertRaises(ValueError): FeatureConfig(**args)

    def test_config_roundtrip(self):
        extractor = LogMelExtractor()
        self.assertEqual(FeatureConfig(**extractor.to_dict()), extractor.config)
