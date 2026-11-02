import json
import shutil
import torch
from safetensors.torch import load_file
from src import paths
from src.evaluator import evaluate_asr, write_evaluation, edit_distance
from src.inference import load_checkpoint, load_release, transcribe
from src.packaging import package_release, sha256_file
from tests.helpers import Sandbox


class PipelineTests(Sandbox):
    def evaluated(self):
        path=self.train_once()
        ck=torch.load(path,map_location='cpu',weights_only=True)
        model,tok,features=load_checkpoint(path,'cpu')
        result=evaluate_asr(model,tok,features,self.val)
        report=write_evaluation(ck['run_id'],ck['epoch'],sha256_file(path),result)
        return path,ck,result,report

    def test_training_logging_selection(self):
        p=self.train_once(epochs=2); ck=torch.load(p,map_location='cpu',weights_only=True)
        run=p.parent.parent
        rows=[json.loads(x) for x in (run/'logs/metrics.jsonl').read_text().splitlines()]
        best=min(rows,key=lambda r:(r['validation']['cer'],r['validation']['loss']))
        self.assertEqual(ck['epoch'],best['epoch'])
        self.assertTrue(ck['optimizer_state_dict']['state'])
        self.assertIn('dataset_fingerprints',ck)

    def test_resume_optimizer(self):
        p=self.train_once(); old=torch.load(p,map_location='cpu',weights_only=True)
        self.train_once(resume=str(p),epochs=2)
        files=list((self.root/'runs').glob('*/checkpoints/epoch_0002.pt'))
        self.assertEqual(len(files),1)
        new=torch.load(files[0],map_location='cpu',weights_only=True)
        first=next(iter(old['optimizer_state_dict']['state']))
        self.assertGreater(new['optimizer_state_dict']['state'][first]['step'],old['optimizer_state_dict']['state'][first]['step'])

    def test_resume_rejections(self):
        p=self.train_once(); ck=torch.load(p,map_location='cpu',weights_only=True)
        for key,value in [('tokenizer',{**ck['tokenizer'],'vocab_size':99}),('feature_config',{**ck['feature_config'],'hop_length':80}),('model_config',{**ck['model_config'],'hidden_size':16}),('source_manifest',{'train':[],'val':self.val})]:
            with self.subTest(key=key):
                broken=self.root/'broken.pt';torch.save({**ck,key:value},broken)
                with self.assertRaises(ValueError):self.train_once(resume=str(broken),epochs=2)
        broken=self.root/'corrupt.pt';broken.write_bytes(b'corrupt')
        with self.assertRaises(Exception):load_checkpoint(broken,'cpu')

    def test_metrics_edit_distance(self):
        for ref,hyp,want in [('abc','',3),('abc','adc',1),('ab','abc',1),('abc','ac',1),('','',0)]:
            self.assertEqual(edit_distance(ref,hyp),want)
        self.assertEqual(edit_distance('one two'.split(),'one three four'.split()),2)

    def test_release_end_to_end(self):
        p,ck,result,report=self.evaluated()
        out=package_release(p,'0.1.0',result['cer'],True)
        a,t,f=load_checkpoint(p,'cpu');b,u,g=load_release('0.1.0','cpu')
        self.assertEqual(f,g)
        self.assertEqual(transcribe(a,t,f,self.val[0]['audio']),transcribe(b,u,g,self.val[0]['audio']))
        for name,tensor in load_file(str(out/'model.safetensors')).items():
            self.assertTrue(torch.equal(tensor,ck['model_state_dict'][name]))
        manifest=json.loads((out/'manifest.json').read_text())
        self.assertFalse(manifest['production_ready'])
        for name,digest in manifest['files'].items():self.assertEqual(sha256_file(out/name),digest)
        with self.assertRaises(FileExistsError):package_release(p,'0.1.0',result['cer'],True)
        with self.assertRaises(FileExistsError):write_evaluation(ck['run_id'],ck['epoch'],sha256_file(p),result)

    def test_gate_and_stale_report(self):
        p,ck,result,report=self.evaluated()
        with self.assertRaises(ValueError):package_release(p,'0.1.0',0,False)
        r=json.loads(report.read_text());r['cer']=1
        report.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'threshold'):package_release(p,'0.1.0',0.5,True)
        r['checkpoint_sha256']='bad';report.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'changed after evaluation'):package_release(p,'0.1.0',1,True)

    def test_release_integrity_and_paths(self):
        p,ck,result,_=self.evaluated();out=package_release(p,'0.1.0',result['cer'],True)
        manifest=json.loads((out/'manifest.json').read_text())
        for files in ({}, {**manifest['files'],'../escape':'bad'}):
            (out/'manifest.json').write_text(json.dumps({**manifest,'files':files}))
            with self.assertRaises(ValueError):load_release('0.1.0','cpu')
        (out/'manifest.json').write_text(json.dumps(manifest))
        (out/'config.json').write_text('{}')
        with self.assertRaises(ValueError):load_release('0.1.0','cpu')
        (out/'config.json').unlink()
        with self.assertRaises(FileNotFoundError):load_release('0.1.0','cpu')
        for version in ('../../escape','latest','1.0',''):
            with self.assertRaises(ValueError):paths.get_release_dir(version)

    def test_existing_speech_integration(self):
        # Actual paired speech WAVs; current corpus provenance is synthetic, not human.
        source=paths.BERHOT_ROOT/'intake/speech'
        if not (source/'train').is_dir(): self.skipTest('Shared speech corpus unavailable')
        from src.ingestion import inspect_split
        self.train_root=source/'train';self.val_root=source/'val'
        self.train=inspect_split(self.train_root);self.val=inspect_split(self.val_root)
        self.test_release_end_to_end()

    def test_empty_hypothesis_metrics(self):
        from unittest.mock import patch
        with patch('src.evaluator.transcribe',return_value=''):
            result=evaluate_asr(None,None,None,self.val)
        self.assertEqual(result['cer'],1.0)
        self.assertEqual(result['wer'],1.0)

    def test_evaluation_checkpoint_binding(self):
        from src.evaluator import evaluate_checkpoint
        p=self.train_once()
        result,report=evaluate_checkpoint(p,'cpu')
        original=report.read_bytes()
        second,_=evaluate_checkpoint(p,'cpu')
        self.assertEqual(result,second);self.assertEqual(original,report.read_bytes())
        record=json.loads(report.read_text());record['checkpoint_sha256']='stale'
        report.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'provenance'):evaluate_checkpoint(p,'cpu')

    def test_checkpoint_vocabulary_rejection(self):
        p=self.train_once();ck=torch.load(p,weights_only=True)
        from src.tokenizer import CharTokenizer
        ck['tokenizer']=CharTokenizer(['ا']).to_dict()
        bad=self.root/'bad-vocab.pt';torch.save(ck,bad)
        with self.assertRaisesRegex(ValueError,'vocabulary'):load_checkpoint(bad,'cpu')
