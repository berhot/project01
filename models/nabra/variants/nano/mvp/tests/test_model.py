import os
import unittest
import torch
from src.model import ModelConfig, NabraASR, ctc_loss


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7)
        self.model=NabraASR(ModelConfig(vocab_size=4,hidden_size=8,num_layers=1,dropout=0))
    def test_shapes_lengths_finite(self):
        logits,lengths=self.model(torch.randn(2,81,80),torch.tensor([81,60]))
        self.assertEqual(logits.shape,(2,21,4)); self.assertEqual(lengths.tolist(),[21,15])
        self.assertTrue(torch.isfinite(logits).all())
    def test_loss_backward_optimizer(self):
        before=self.model.output.weight.detach().clone()
        optim=torch.optim.AdamW(self.model.parameters(),lr=0.01)
        logits,lengths=self.model(torch.randn(1,40,80),torch.tensor([40]))
        loss=ctc_loss(logits,lengths,torch.tensor([1,2]),torch.tensor([2]))
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        self.assertGreater(self.model.output.weight.grad.abs().sum().item(),0)
        optim.step(); self.assertFalse(torch.equal(before,self.model.output.weight))
    def test_state_equivalence(self):
        other=NabraASR(self.model.config);other.load_state_dict(self.model.state_dict())
        x=torch.randn(1,40,80); n=torch.tensor([40])
        self.assertTrue(torch.equal(self.model(x,n)[0],other(x,n)[0]))
    @unittest.skipUnless(os.getenv('NABRA_TEST_MPS')=='1' and torch.backends.mps.is_available(),'Optional MPS test; set NABRA_TEST_MPS=1')
    def test_mps_backward(self):
        self.model.to('mps')
        logits,lengths=self.model(torch.randn(1,40,80,device='mps'),torch.tensor([40]))
        loss=ctc_loss(logits,lengths,torch.tensor([1,2]),torch.tensor([2]));loss.backward()
        self.assertTrue(torch.isfinite(loss))
