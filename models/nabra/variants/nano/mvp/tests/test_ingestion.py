import json
import shutil
import wave

from src.ingestion import inspect_recording, inspect_split, sha256_file, write_manifest, read_records, validate_splits
from tests.helpers import Sandbox, pair, wav


class IngestionTests(Sandbox):
    def test_valid_and_utf8(self):
        record = self.train[0]
        self.assertEqual(record['transcript'], 'اب اب')
        self.assertEqual(record['sample_rate'], 16000)
        self.assertEqual(record['audio_sha256'], sha256_file(self.train_root/'a.wav'))

    def test_missing_audio(self):
        with self.assertRaises(FileNotFoundError):
            inspect_recording(self.root/'missing.wav', self.train_root/'a.txt')

    def test_missing_transcript(self):
        with self.assertRaises(FileNotFoundError):
            inspect_recording(self.train_root/'a.wav', self.root/'missing.txt')

    def test_empty_transcript(self):
        (self.train_root/'a.txt').write_text('  \n')
        with self.assertRaisesRegex(ValueError, 'Empty transcript'):
            inspect_split(self.train_root)

    def test_invalid_audio(self):
        (self.train_root/'a.wav').write_bytes(b'not a wave file')
        with self.assertRaises((wave.Error, EOFError, ValueError)):
            inspect_split(self.train_root)

    def test_audio_format_rejections(self):
        for kwargs in ({'rate':8000}, {'channels':2}, {'width':1}, {'width':3}, {'seconds':0.1}, {'seconds':20.1}):
            with self.subTest(kwargs=kwargs):
                pair(self.train_root, **kwargs)
                with self.assertRaises(ValueError):
                    inspect_split(self.train_root)

    def test_duplicate_recordings(self):
        shutil.copy(self.train_root/'a.wav', self.train_root/'b.wav')
        (self.train_root/'b.txt').write_text('اب')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            inspect_split(self.train_root)

    def test_cross_split_duplicates(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_splits(self.train, self.train)
        validate_splits(self.train, self.val)

    def test_unmatched_transcript(self):
        (self.train_root/'extra.txt').write_text('اب')
        with self.assertRaisesRegex(ValueError, 'Unmatched'):
            inspect_split(self.train_root)

    def test_manifest_provenance(self):
        target = self.root/'manifest.jsonl'
        write_manifest(self.train, target)
        self.assertEqual(read_records(target), self.train)
        value = json.loads(target.read_text())
        value['audio_sha256'] = '0'*64
        target.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            read_records(target)

    def test_no_audio(self):
        empty = self.root/'empty'; empty.mkdir()
        with self.assertRaises(ValueError): inspect_split(empty)

    def test_truncated_audio(self):
        p = self.train_root/'a.wav'; p.write_bytes(p.read_bytes()[:-100])
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            inspect_split(self.train_root)
