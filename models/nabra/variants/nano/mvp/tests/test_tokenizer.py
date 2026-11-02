import unittest
from src.tokenizer import CharTokenizer


class TokenizerTests(unittest.TestCase):
    def setUp(self): self.tokenizer = CharTokenizer.from_transcripts(['اب ب'])
    def test_encode_decode(self):
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode('اب ب')), 'اب ب')
    def test_blank(self): self.assertEqual(self.tokenizer.ctc_decode([0,0]), '')
    def test_repeats(self):
        a = self.tokenizer.encode('ا')[0]
        self.assertEqual(self.tokenizer.ctc_decode([a,a,0,a]), 'اا')
    def test_serialization(self):
        self.assertEqual(CharTokenizer.from_dict(self.tokenizer.to_dict()).to_dict(), self.tokenizer.to_dict())
    def test_oov(self):
        with self.assertRaises(ValueError): self.tokenizer.encode('z')
    def test_invalid_vocabulary(self):
        for change in ({'blank_id':1}, {'type':'other'}, {'vocab_size':99}, {'chars':['ا','ا']}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                CharTokenizer.from_dict({**self.tokenizer.to_dict(), **change})
