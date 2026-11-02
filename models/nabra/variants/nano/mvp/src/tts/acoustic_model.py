"""Small autoregressive attention acoustic model; learns alignment without forced durations."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AcousticConfig:
    vocab_size: int = 353
    n_mels: int = 80
    hidden_size: int = 64
    reduction: int = 4
    num_speakers: int = 1
    def __post_init__(self):
        if any(not isinstance(x,int) or x<1 for x in (self.vocab_size,self.n_mels,self.hidden_size,self.reduction,self.num_speakers)):
            raise ValueError('Invalid acoustic model dimensions')


class AcousticModel(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config;c=config;h=c.hidden_size
        self.embedding=nn.Embedding(c.vocab_size,h,padding_idx=0)
        self.encoder=nn.GRU(h,h,batch_first=True,bidirectional=True)
        self.speakers=nn.Embedding(c.num_speakers,h)
        self.key=nn.Linear(h*2,h);self.query=nn.Linear(h,h)
        self.energy=nn.Linear(h,1,bias=False)
        self.location=nn.Conv1d(1,h,7,padding=3,bias=False)
        self.initial=nn.Linear(h*2,h)
        self.prenet=nn.Sequential(nn.Linear(c.n_mels,h),nn.ReLU())
        self.decoder=nn.GRUCell(h*3,h)
        self.mel=nn.Linear(h*3,c.reduction*c.n_mels)
        self.stop=nn.Linear(h*3,1)

    def encode(self,tokens,lengths,speakers):
        x=self.embedding(tokens)+self.speakers(speakers)[:,None,:]
        packed=nn.utils.rnn.pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
        encoded,_=self.encoder(packed)
        memory,_=nn.utils.rnn.pad_packed_sequence(encoded,batch_first=True,total_length=tokens.shape[1])
        mask=torch.arange(tokens.shape[1],device=tokens.device)[None,:]>=lengths.to(tokens.device)[:,None]
        pooled=memory.sum(1)/lengths.to(memory.device)[:,None]
        return memory,self.key(memory),mask,torch.tanh(self.initial(pooled))

    def step(self,previous,state,memory,keys,mask,cumulative):
        energy=self.energy(torch.tanh(keys+self.query(state)[:,None,:]+self.location(cumulative[:,None,:]).transpose(1,2))).squeeze(-1)
        attention=energy.masked_fill(mask,-torch.inf).softmax(-1)
        context=torch.bmm(attention[:,None,:],memory).squeeze(1)
        state=self.decoder(torch.cat([self.prenet(previous),context],-1),state)
        joined=torch.cat([state,context],-1)
        mel=self.mel(joined).reshape(-1,self.config.reduction,self.config.n_mels)
        return mel,self.stop(joined).squeeze(-1),state,attention

    def forward(self,tokens,text_lengths,speakers,targets,teacher_forcing_ratio=1.0):
        if not 0 <= teacher_forcing_ratio <= 1: raise ValueError("Invalid teacher forcing ratio")
        memory,keys,mask,state=self.encode(tokens,text_lengths,speakers)
        previous=targets.new_zeros((len(tokens),self.config.n_mels))
        cumulative=targets.new_zeros(mask.shape)
        mels=[];stops=[];attentions=[]
        for start in range(0,targets.shape[1],self.config.reduction):
            mel,stop,state,attention=self.step(previous,state,memory,keys,mask,cumulative)
            mels.append(mel);stops.append(stop);attentions.append(attention)
            cumulative=cumulative+attention
            teacher=targets[:,min(start+self.config.reduction,targets.shape[1])-1]
            previous=teacher if teacher_forcing_ratio==1 else mel[:,-1]
            if 0 < teacher_forcing_ratio < 1:
                choose=torch.rand((len(tokens),1),device=targets.device)<teacher_forcing_ratio
                previous=torch.where(choose,teacher,mel[:,-1])
        return torch.cat(mels,1)[:,:targets.shape[1]],torch.stack(stops,1),torch.stack(attentions,1)

    @torch.inference_mode()
    def generate(self,tokens,speaker=0,max_frames=800,min_frames=20,return_attention=False):
        if tokens.ndim!=2 or tokens.shape[0]!=1: raise ValueError('Synthesis expects one text')
        if not 1<=min_frames<=max_frames<=4000: raise ValueError('Invalid synthesis frame limits')
        lengths=torch.tensor([tokens.shape[1]])
        memory,keys,mask,state=self.encode(tokens,lengths,torch.tensor([speaker],device=tokens.device))
        previous=memory.new_zeros((1,self.config.n_mels));cumulative=memory.new_zeros(mask.shape)
        frames=[];attentions=[];stopped=False
        for start in range(0,max_frames,self.config.reduction):
            mel,stop,state,attention=self.step(previous,state,memory,keys,mask,cumulative)
            cumulative=cumulative+attention;frames.append(mel);attentions.append(attention);previous=mel[:,-1]
            if start+self.config.reduction>=min_frames and stop.sigmoid().item()>0.5:
                stopped=True;break
        info={'stop_predicted':stopped,'frame_limit_reached':not stopped}
        if return_attention: info['attention']=torch.stack(attentions,1)
        return torch.cat(frames,1)[:,:max_frames],info


def acoustic_loss(model,batch,teacher_forcing_ratio=1.0,return_prediction=False):
    device=next(model.parameters()).device
    tokens=batch['tokens'].to(device);target=batch['mel'].to(device);lengths=batch['mel_lengths'].to(device)
    predicted,stop,attention=model(tokens,batch['text_lengths'],batch['speaker'].to(device),target,teacher_forcing_ratio=teacher_forcing_ratio)
    mask=torch.arange(target.shape[1],device=device)[None,:]<lengths[:,None]
    mel=((predicted-target).abs()*mask[:,:,None]).sum()/(mask.sum()*target.shape[-1])
    steps=(lengths+model.config.reduction-1)//model.config.reduction
    positions=torch.arange(stop.shape[1],device=device)[None,:]
    stop_target=(positions>=steps[:,None]-1).float()
    valid=positions<steps[:,None]
    stop_loss=(F.binary_cross_entropy_with_logits(stop,stop_target,reduction='none')*valid).sum()/valid.sum()
    text_lengths=batch['text_lengths'].to(device)
    ti=positions[:,:,None]/steps[:,None,None]
    ni=torch.arange(tokens.shape[1],device=device)[None,None,:]/text_lengths[:,None,None]
    guide=1-torch.exp(-((ti-ni)**2)/(2*0.2**2))
    valid_attention=valid[:,:,None] & (torch.arange(tokens.shape[1],device=device)[None,None,:]<text_lengths[:,None,None])
    guided=(attention*guide*valid_attention).sum()/valid.sum()
    total=mel+0.2*stop_loss+0.1*guided
    metrics={'mel_l1':float(mel.detach()),'stop_bce':float(stop_loss.detach()),'guided_attention':float(guided.detach())}
    return (total,metrics,predicted) if return_prediction else (total,metrics)
