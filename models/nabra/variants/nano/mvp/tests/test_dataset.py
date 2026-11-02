import torch
from src.dataset import SpeechDataset, collate_speech
from src.tokenizer import CharTokenizer
from src.ingestion import inspect_split
from src.trainer import required_ctc_lengths, forward_loss
from src.model import NabraASR, ModelConfig
from tests.helpers import Sandbox, pair


class DatasetTests(Sandbox):
    def test_padding_lengths_targets(self):
        pair(self.train_root, 'b', 'ا', seconds=0.65, frequency=440)
        records = inspect_split(self.train_root)
        tok = CharTokenizer.from_transcripts([r['transcript'] for r in records])
        ds = SpeechDataset(records,tok)
        a,b = ds[0],ds[1]; batch=collate_speech([a,b])
        self.assertEqual(batch['feature_lengths'].tolist(),[len(a[0]),len(b[0])])
        self.assertEqual(batch['target_lengths'].tolist(),[5,1])
        self.assertEqual(batch['targets'].numel(),6)
        self.assertTrue(torch.equal(batch['features'][0,len(a[0]):],torch.zeros_like(batch['features'][0,len(a[0]):])))

    def test_alignment_feasibility(self):
        self.assertEqual(required_ctc_lengths(torch.tensor([1,1,2]),torch.tensor([3])),[4])
        model=NabraASR(ModelConfig(vocab_size=3,hidden_size=8,num_layers=1))
        batch={'features':torch.zeros(1,4,80),'feature_lengths':torch.tensor([4]),'targets':torch.tensor([1,1,2]),'target_lengths':torch.tensor([3])}
        with self.assertRaisesRegex(ValueError,'alignment impossible'): forward_loss(model,batch,'cpu')

    def test_empty_dataset(self):
        with self.assertRaises(ValueError): SpeechDataset([],CharTokenizer(['ا']))
