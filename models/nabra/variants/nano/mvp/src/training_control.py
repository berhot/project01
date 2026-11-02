"""Cooperative stopping: finish an optimizer step, checkpoint, restore handlers."""
import math
import signal
import time

import torch


class TrainingControl:
    def __init__(self, max_seconds=None):
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds<=0):
            raise ValueError('max_seconds must be positive and finite')
        self.max_seconds=max_seconds
        self.requested=False
        self.previous={}

    def __enter__(self):
        self.started=time.monotonic()
        # Python only permits process signal handlers in the main thread.
        import threading
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT,signal.SIGTERM):
                self.previous[sig]=signal.getsignal(sig)
                signal.signal(sig,self.request_stop)
        return self

    def request_stop(self,*_):
        self.requested=True

    def reason(self):
        if self.requested:return 'interrupted'
        if self.max_seconds is not None and time.monotonic()-self.started>=self.max_seconds:return 'time_limit'
        return None

    def __exit__(self,*_):
        for sig,handler in self.previous.items():signal.signal(sig,handler)


def accelerator_rng(device):
    if device.type=='cuda':return {'cuda':torch.cuda.get_rng_state_all()}
    if device.type=='mps':return {'mps':torch.mps.get_rng_state()}
    return {}


def restore_accelerator_rng(state,device):
    if device.type=='cuda' and 'cuda' in state:torch.cuda.set_rng_state_all(state['cuda'])
    if device.type=='mps' and 'mps' in state:torch.mps.set_rng_state(state['mps'])
