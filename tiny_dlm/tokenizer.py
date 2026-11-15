"""A word-level tokenizer, built from the training data (no download, no Hugging Face)."""
import re

PAD, UNK, CLS, SEP, MASK = "<pad>", "<unk>", "<cls>", "<sep>", "<mask>"
SPECIALS = [PAD, UNK, CLS, SEP, MASK]  # always ids 0, 1, 2, 3, 4


class WordTokenizer:
    """
        Lowercases, then splits into words and the punctuation "." and "?" (sentence ends carry meaning:
        in bAbI the *last* sentence about a person is the one that counts).
        Words never seen in training become <unk>.

        One model input is a (context, candidate) pair:
            encode_pair("it is raining", "take an umbrella")
            -> [<cls>] it is raining [<sep>] take an umbrella
    """

    def __init__(self, itos):
        assert itos[:len(SPECIALS)] == SPECIALS, "the special tokens must come first"
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id, self.unk_id, self.cls_id, self.sep_id, self.mask_id = (self.stoi[t] for t in SPECIALS)

    @classmethod
    def from_texts(cls, texts):
        words = {w for text in texts for w in cls.split(text)}
        return cls(SPECIALS + sorted(words - set(SPECIALS)))  # sorted -> same ids every run

    @staticmethod
    def split(text):
        return re.findall(r"[a-z0-9']+|[.?]", text.lower())

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi.get(w, self.unk_id) for w in self.split(text)]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids if i != self.pad_id)

    def encode_pair(self, context, candidate):
        return [self.cls_id] + self.encode(context) + [self.sep_id] + self.encode(candidate)

    def to_dict(self):
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d):
        return cls(d["itos"])
