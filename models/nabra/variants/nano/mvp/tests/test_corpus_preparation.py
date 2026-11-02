"""Tests for prompt grouping and immutable source processing invariants."""
import tempfile
import unittest
import wave
from pathlib import Path
from src.corpus_preparation import prompt_key, split_groups, check_leakage, preflight
from src.ingestion import sha256_file

class CorpusPreparationTests(unittest.TestCase):
    def test_repeated_prompts_grouped(self):
        rows=[{"transcript":"مرحبا","audio_sha256":"a"},{"transcript":"مَرْحَبًا","audio_sha256":"b"},
              {"transcript":"صباح الخير","audio_sha256":"c"},{"transcript":"كيف حالك","audio_sha256":"d"}]
        train,val=split_groups(rows,0.34,42)
        self.assertTrue(check_leakage(train,val))
        self.assertEqual(len(train)+len(val),4)
        self.assertEqual(prompt_key("مرحبا"),prompt_key("مَرْحَبًا"))
    def test_duplicate_and_speaker_leakage(self):
        a=[{"transcript":"واحد","audio_sha256":"1","speaker_id":"same"}]
        b=[{"transcript":"اثنان","audio_sha256":"2","speaker_id":"same"}]
        self.assertRaises(ValueError,check_leakage,a,b,True)
        self.assertRaises(ValueError,check_leakage,a,[{"transcript":"ثلاثة","audio_sha256":"1"}])
    def test_single_prompt_split_rejected(self):
        self.assertRaises(ValueError,split_groups,[{"transcript":"نفسه"}]*3)
    def test_duplicate_original_audio_is_rejected_before_creation(self):
        from src.corpus_preparation import prepare
        import json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            original=root/"original.wav"
            original.write_bytes(b"fake source is never converted")
            index=root/"index.jsonl"
            lines=[
                {"audio":"original.wav","transcript":"مرحبا"},
                {"audio":"original.wav","transcript":"صباح الخير"},
            ]
            index.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in lines),encoding="utf8")
            destination=root/"prepared"
            with self.assertRaisesRegex(ValueError,"Duplicate original"):
                prepare(index,destination)
            self.assertFalse(destination.exists())

    def test_preflight_with_valid_wavs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            import json
            for name,text in (("train","مرحبا"),("val","صباح الخير")):
                wav=root/(name+".wav")
                with wave.open(str(wav),"wb") as w:
                    w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
                    w.writeframes((b"\x01\x00" if name=="train" else b"\x02\x00")*8000)
                record={"audio":name+".wav","transcript":text,"audio_sha256":sha256_file(wav)}
                (root/(name+".jsonl")).write_text(json.dumps(record,ensure_ascii=False)+"\n",encoding="utf8")
            report=preflight(root/"train.jsonl",root/"val.jsonl")
            self.assertEqual(report["train"],1)
            self.assertEqual(report["val"],1)

if __name__=="__main__":
    unittest.main()
