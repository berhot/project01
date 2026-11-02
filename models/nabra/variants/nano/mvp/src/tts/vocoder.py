"""Compact deterministic convolutional neural vocoder, trained with waveform/STFT losses.
Not a MelGAN reproduction: no adversarial discriminator is implemented.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class VocoderConfig:
    n_mels:int=80
    channels:int=32
    def __post_init__(self):
        if self.n_mels<1 or self.channels<8:raise ValueError('Invalid vocoder dimensions')


class Vocoder(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        c=config.channels
        layers=[nn.Conv1d(config.n_mels,c,7,padding=3)]
        for stride in (5,4,4,2):
            next_c=max(c//2,4)
            layers += [nn.LeakyReLU(0.1),nn.ConvTranspose1d(c,next_c,stride*2,stride=stride,
                       padding=(stride+1)//2,output_padding=stride%2),nn.LeakyReLU(0.1),
                       nn.Conv1d(next_c,next_c,3,padding=1)]
            c=next_c
        layers += [nn.LeakyReLU(0.1),nn.Conv1d(c,1,7,padding=3),nn.Tanh()]
        self.network=nn.Sequential(*layers)
    def forward(self,mel):return self.network(mel.transpose(1,2)).squeeze(1)


def reconstruction_loss(predicted,target):
    if predicted.shape!=target.shape:raise ValueError('Waveform shapes differ')
    # MPS lacks complete complex STFT support; CPU transfers retain autograd.
    if predicted.device.type == 'mps':
        predicted, target = predicted.cpu(), target.cpu()
    l1=F.l1_loss(predicted,target)
    spectral=predicted.new_zeros(())
    for size in (256,512,1024):
        window=torch.hann_window(size,device=predicted.device)
        a=torch.stft(predicted,n_fft=size,hop_length=size//4,window=window,return_complex=True,pad_mode='constant').abs().clamp_min(1e-5)
        b=torch.stft(target,n_fft=size,hop_length=size//4,window=window,return_complex=True,pad_mode='constant').abs().clamp_min(1e-5)
        spectral=spectral+F.l1_loss(a.log(),b.log())+(a-b).norm()/b.norm().clamp_min(1e-5)
    spectral=spectral/3
    return l1+spectral,{'waveform_l1':float(l1.detach()),'spectral_loss':float(spectral.detach())}
