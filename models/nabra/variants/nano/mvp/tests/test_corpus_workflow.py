import contextlib
import io
import json
import shutil
import signal
from unittest.mock import patch

from src.corpus import prepare, preflight, grouped_split, prompt_key
from src.ingestion import read_records, sha256_file
from src.runtime import fingerprint
from src.training_control import TrainingControl
from src.tts import trainer
from tests.helpers import Sandbox, pair


class CorpusWorkflowTests(Sandbox):
    def setUp(self):
        super().setUp()
        self.raw=self.root/'raw'
        pair(self.raw,'one','اب اب',frequency=230)
        pair(self.raw,'two','با با',frequency=340)
        pair(self.raw,'three','اب با',frequency=450)
        self.output=self.root/'prepared'

    def test_portable_bundle_and_preflight(self):
        hashes={p:sha256_file(p) for p in self.raw.iterdir()}
        prepare(self.raw,self.output)
        train=read_records(self.output/'train.jsonl')
        self.assertTrue(all(not __import__('pathlib').Path(json.loads(line)['audio']).is_absolute()
                            for line in (self.output/'train.jsonl').read_text().splitlines()))
        moved=self.root/'moved';self.output.rename(moved)
        self.assertEqual(fingerprint(train),fingerprint(read_records(moved/'train.jsonl')))
        result=preflight(moved/'train.jsonl',moved/'val.jsonl')
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['warnings'],[])
        self.assertTrue(all(sha256_file(p)==digest for p,digest in hashes.items()))
        with self.assertRaises(FileExistsError):prepare(self.raw,moved)

    def test_grouping_and_order_independence(self):
        rows=[{'transcript':text,'audio_sha256':str(i),'speaker_id':'s'+str(i//2)}
              for i,text in enumerate(['اب',' اب  ','با','با'])]
        a=grouped_split(rows);b=grouped_split(list(reversed(rows)))
        self.assertEqual({r['audio_sha256'] for r in a['val']},{r['audio_sha256'] for r in b['val']})
        self.assertFalse({prompt_key(r['transcript']) for r in a['train']} & {prompt_key(r['transcript']) for r in a['val']})
        split=grouped_split(rows,split_by='speaker')
        self.assertFalse({r['speaker_id'] for r in split['train']} & {r['speaker_id'] for r in split['val']})
        with self.assertRaises(ValueError):grouped_split(rows,split_by='source')
        with self.assertRaises(ValueError):grouped_split(rows[:2])

    def test_metadata_preserved(self):
        meta={'speaker_id':'speaker01','source_id':'session01','provenance':{'source_type':'human-speech','license':'permission-record-01'}}
        for p in self.raw.glob('*.wav'):p.with_suffix('.json').write_text(json.dumps(meta))
        prepare(self.raw,self.output)
        rows=read_records(self.output/'train.jsonl')
        self.assertEqual(rows[0]['provenance']['license'],'permission-record-01')
        self.assertEqual(rows[0]['speaker_id'],'speaker01')
        result=preflight(self.output/'train.jsonl',self.output/'val.jsonl')
        self.assertIn('1 shared speakers across train/validation',result['warnings'])

    def test_synthetic_provenance_cannot_be_erased(self):
        samples=[{'audio_sha256':sha256_file(p)} for p in self.raw.glob('*.wav')]
        (self.raw/'synthetic-samples.json').write_text(json.dumps({'generator':'test generator','voice':'test voice','samples':samples}))
        for p in self.raw.glob('*.wav'):p.with_suffix('.json').write_text(json.dumps({'provenance':{'source_type':'human-speech'}}))
        prepare(self.raw,self.output)
        self.assertTrue(all(r['provenance']['source_type']=='synthetic-speech' for r in read_records(self.output/'train.jsonl')))

    def test_reject_partial_invalid_and_duplicate_data(self):
        (self.raw/'orphan.txt').write_text('اب')
        with self.assertRaises(ValueError):prepare(self.raw,self.output)
        self.assertFalse(self.output.exists());(self.raw/'orphan.txt').unlink()
        (self.raw/'two.wav').write_bytes((self.raw/'one.wav').read_bytes())
        with self.assertRaisesRegex(ValueError,'Duplicate'):prepare(self.raw,self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.prepare-*')))
        with self.assertRaises(ValueError):prepare(self.raw,self.raw/'out')

    def test_transcript_tampering_rejected(self):
        prepare(self.raw,self.output)
        path=self.output/'train.jsonl';row=json.loads(path.read_text().splitlines()[0])
        (self.output/row['transcript_file']).write_text('changed')
        with self.assertRaisesRegex(ValueError,'Transcript manifest hash'):read_records(path)

    def test_preflight_rejects_oov_and_impossible_ctc(self):
        pair(self.val_root,'a','ز',frequency=300)
        with self.assertRaises(ValueError):preflight(self.train_root,self.val_root,'asr')
        (self.val_root/'a.txt').write_text('ا'*100)
        with self.assertRaisesRegex(ValueError,'CTC alignment'):preflight(self.train_root,self.val_root,'asr')

    def test_unseen_tts_speaker(self):
        for i,p in enumerate(self.raw.glob('*.wav')):p.with_suffix('.json').write_text(json.dumps({'speaker_id':str(i)}))
        prepare(self.raw,self.output,split_by='speaker')
        with self.assertRaisesRegex(ValueError,'unseen speaker'):preflight(self.output/'train.jsonl',self.output/'val.jsonl','tts')

    @__import__('unittest').skipUnless(shutil.which('ffmpeg'),'ffmpeg unavailable')
    def test_real_conversion_preserves_sources(self):
        raw=self.root/'stereo'
        pair(raw,'one','اب اب',rate=44100,channels=2,frequency=230)
        pair(raw,'two','با با',rate=22050,channels=2,frequency=340)
        before={p:sha256_file(p) for p in raw.iterdir()}
        result=prepare(raw,self.output,convert=True)
        self.assertTrue(result['converted'])
        self.assertTrue(all(r['sample_rate']==16000 for r in read_records(self.output/'train.jsonl')))
        self.assertTrue(all(sha256_file(p)==digest for p,digest in before.items()))

    def test_control_restores_handlers(self):
        before=signal.getsignal(signal.SIGINT)
        with TrainingControl() as control:
            control.request_stop();self.assertEqual(control.reason(),'interrupted')
        self.assertEqual(signal.getsignal(signal.SIGINT),before)
        with self.assertRaises(ValueError):TrainingControl(float('nan'))

    def test_time_budget_checkpoint_and_resume(self):
        options={'acoustic':{'hidden_size':8},'vocoder':{'channels':8}}
        with patch.object(trainer,'TTS_RUN_ROOT',self.root/'tts'),contextlib.redirect_stdout(io.StringIO()):
            checkpoint=trainer.train(self.train_root,self.val_root,steps=4,max_seconds=1e-12,config=options)
            summary=json.loads((checkpoint.parent.parent/'logs/summary.json').read_text())
            self.assertEqual((summary['status'],summary['steps'],summary['target_steps']),('time_limit',1,4))
            new=trainer.train(self.train_root,self.val_root,steps=2,resume=checkpoint,config=options)
            self.assertEqual(json.loads((new.parent.parent/'logs/summary.json').read_text())['status'],'completed')

    def test_stop_request_saves_after_current_step(self):
        with patch.object(trainer,'TTS_RUN_ROOT',self.root/'tts'),patch.object(TrainingControl,'reason',return_value='interrupted'),contextlib.redirect_stdout(io.StringIO()):
            checkpoint=trainer.train(self.train_root,self.val_root,steps=4,config={'acoustic':{'hidden_size':8},'vocoder':{'channels':8}})
        summary=json.loads((checkpoint.parent.parent/'logs/summary.json').read_text())
        self.assertEqual(summary['status'],'interrupted');self.assertEqual(summary['steps'],1)
