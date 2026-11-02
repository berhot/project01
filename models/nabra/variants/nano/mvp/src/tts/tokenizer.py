import string
import unicodedata


def normalize(text):
    """NFC and whitespace normalization only; preserve Arabic letters and diacritics."""
    if not isinstance(text, str):
        raise ValueError('Text must be a string')
    return ' '.join(unicodedata.normalize('NFC', text).split())


class TextTokenizer:
    TYPE = 'nabra-tts-char-v1'
    def __init__(self, chars=None):
        # A fixed alphabet permits unseen inputs; their pronunciations are not learned by definition.
        self.chars = sorted(set(chars if chars is not None else
            ' ' + string.ascii_letters + string.digits + string.punctuation + ''.join(chr(i) for i in range(0x600,0x700))))
        if not self.chars or any(not isinstance(c,str) or len(c)!=1 for c in self.chars):
            raise ValueError('Invalid character vocabulary')
        self.stoi = {c:i+2 for i,c in enumerate(self.chars)}  # PAD=0, EOS=1
        self.vocab_size = len(self.chars)+2
    def encode(self,text):
        text=normalize(text)
        if not text: raise ValueError('Empty text')
        try: return [self.stoi[c] for c in text]+[1]
        except KeyError as e: raise ValueError(f'Unsupported character: {e.args[0]!r}') from e
    def to_dict(self):
        return {'type':self.TYPE,'chars':self.chars,'normalization':'NFC-whitespace-preserve-diacritics','pad_id':0,'eos_id':1}
    @classmethod
    def from_dict(cls,data):
        if data.get('type')!=cls.TYPE or data.get('pad_id')!=0 or data.get('eos_id')!=1 or data.get('normalization')!='NFC-whitespace-preserve-diacritics':
            raise ValueError('Invalid TTS tokenizer')
        if len(data['chars'])!=len(set(data['chars'])): raise ValueError('Duplicate vocabulary')
        return cls(data['chars'])
