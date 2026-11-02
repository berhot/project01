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
    min_channels:int=4
    residual:bool=False
    head:str="waveform"
    def __post_init__(self):
        if self.head not in {'waveform','spectral'}: raise ValueError('Invalid vocoder head')
        if self.n_mels<1 or self.channels<8 or not 4<=self.min_channels<=self.channels:raise ValueError('Invalid vocoder dimensions')


class ResidualBlock(nn.Module):
    """Dilated learned corrections expand context at each upsampling scale."""
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv1d(channels, channels, 7, dilation=d, padding=3*d)
                                     for d in (1, 3, 9)])
    def forward(self, value):
        for layer in self.layers:
            value = value + 0.1 * layer(F.leaky_relu(value, 0.1))
        return value


class Vocoder(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        c=config.channels
        if config.head == 'spectral':
            # Predict BOTH real and imaginary coefficients from Mel. No target phase,
            # stored waveform, Griffin-Lim, or Mel inversion is used in synthesis.
            self.network = nn.Sequential(nn.Conv1d(config.n_mels,c,7,padding=3),
                                         nn.LeakyReLU(0.1), ResidualBlock(c),
                                         nn.LeakyReLU(0.1), nn.Conv1d(c,402,1))
            return
        layers=[nn.Conv1d(config.n_mels,c,7,padding=3)]
        for stride in (5,4,4,2):
            next_c=max(c//2,config.min_channels)
            layers += [nn.LeakyReLU(0.1),nn.ConvTranspose1d(c,next_c,stride*2,stride=stride,
                       padding=(stride+1)//2,output_padding=stride%2),nn.LeakyReLU(0.1),
                       nn.Conv1d(next_c,next_c,3,padding=1)]
            if config.residual: layers.append(ResidualBlock(next_c))
            c=next_c
        layers += [nn.LeakyReLU(0.1),nn.Conv1d(c,1,7,padding=3),nn.Tanh()]
        self.network=nn.Sequential(*layers)
    def spectrum(self, mel):
        if self.config.head != 'spectral': raise ValueError('No spectral head')
        coefficients = self.network(mel.transpose(1,2))
        if coefficients.device.type == 'mps': coefficients = coefficients.cpu()
        real, imaginary = coefficients.chunk(2,dim=1)
        return torch.complex(real,imaginary)

    def forward(self, mel):
        if self.config.head == 'waveform':
            return self.network(mel.transpose(1,2)).squeeze(1)
        spectrum = self.spectrum(mel)
        waveform = torch.istft(spectrum, n_fft=400, hop_length=160,
                               window=torch.hann_window(400,device=spectrum.device),
                               length=mel.shape[1]*160)
        return waveform.to(mel.device)


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
