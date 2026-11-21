"""
Two tokenizers, both built from our own data (no download, no Hugging Face). Pick one with --tokenizer.

    bpe   (default) BPETokenizer: byte pair encoding, as in GPT-2. Learns sub-word pieces; never meets an unknown word.
    word            WTokenizer: one id per whole word; words never seen in training become <unk>.

BPETokenizer: a byte-level BPE tokenizer (byte pair encoding, as in GPT-2), trained from scratch on our own text.

    tok = BPETokenizer.train(texts, vocab_size=4096)   # learn the merges
    tok.encode("i lost my card")                       # -> token ids
    tok.tokens("i lost my card")                       # -> ['i', ' lost', ' my', ' card']

How it works, in three steps:
    1. split the text into chunks: a word with its leading space (" card"), a number, punctuation, spaces
    2. start with one token per byte (256 tokens: any text can be written, nothing is ever unknown)
    3. training: count every pair of neighbouring tokens inside the chunks, merge the most frequent pair into
       a new token, repeat until the vocabulary is full. Encoding replays the merges in the order they were learned.

So common words become one token (" card"), rare words are built from pieces.
"""
import json
import re
from collections import Counter, defaultdict

PAD, UNK, CLS, SEP, MASK = "<pad>", "<unk>", "<cls>", "<sep>", "<mask>"
SPECIALS = [PAD, UNK, CLS, SEP, MASK]  # always ids 0, 1, 2, 3, 4; then the 256 bytes; then the merges

# step 1: a word with its leading space | a number | punctuation | spaces (GPT-2's split, with Python's re)
SPLIT = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|_+|\s+(?!\S)|\s+")


class WTokenizer:
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
    def from_texts(cls, texts, min_freq=1, keep=()):
        """
        The vocabulary: every word seen at least `min_freq` times in `texts`, plus every word of `keep`
        (the option texts: a word that can be an answer must never become <unk>, however rare it is).
        """
        counts = Counter(w for text in texts for w in cls.split(text))
        words = {w for w, n in counts.items() if n >= min_freq} | {w for text in keep for w in cls.split(text)}
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
        return {"type": "word", "itos": self.itos}

    @classmethod
    def from_dict(cls, d):
        return cls(d["itos"])


class BPETokenizer:
    """
        merges: the learned merges in order, [(a, b), ...]: merge number r makes token id len(SPECIALS) + 256 + r.

        Text is lowercased first (our datasets are lowercase).
        One model input is a (context, candidate) pair:
            encode_pair("it is raining", "take an umbrella")
            -> <cls> it is raining <sep> take an umbrella
    """

    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self.first_byte = len(SPECIALS)  # token id of byte 0
        self.rank = {pair: r for r, pair in enumerate(self.merges)}  # pair -> merge order (lower = learned earlier)
        pieces = [s.encode() for s in SPECIALS] + [bytes([x]) for x in range(256)]  # the bytes of every token
        for a, b in self.merges:
            pieces.append(pieces[a] + pieces[b])
        self.pieces = pieces
        self.itos = [p.decode("utf-8", errors="replace") for p in pieces]  # every token, shown as text
        self.pad_id, self.unk_id, self.cls_id, self.sep_id, self.mask_id = range(len(SPECIALS))
        self.cache = {}  # chunk -> ids (the same words come back again and again)

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, texts, vocab_size=4096, verbose=False):
        """Learn merges until the vocabulary holds `vocab_size` tokens (or no pair appears twice)."""
        first = len(SPECIALS)
        chunk_counts = Counter(c for text in texts for c in cls.split(text))
        chunks = [[first + x for x in c.encode("utf-8")] for c in chunk_counts]  # each distinct chunk, as bytes
        freq = list(chunk_counts.values())  # how often each chunk appears

        # how often each pair appears, and in which chunks (so a merge only revisits those chunks)
        pairs, where = Counter(), defaultdict(set)
        for n, ids in enumerate(chunks):
            for pair in zip(ids, ids[1:]):
                pairs[pair] += freq[n]
                where[pair].add(n)

        merges = []
        while first + 256 + len(merges) < vocab_size and pairs:
            best = max(pairs, key=lambda p: (pairs[p], -p[0], -p[1]))  # the most frequent pair; ties -> smallest ids
            if pairs[best] < 2:
                break
            new = first + 256 + len(merges)
            merges.append(best)
            for n in where.pop(best):
                ids = chunks[n]
                for pair in zip(ids, ids[1:]):  # forget this chunk's old pairs...
                    pairs[pair] -= freq[n]
                    if pairs[pair] <= 0:
                        del pairs[pair]
                chunks[n] = ids = cls.merge(ids, best, new)
                for pair in zip(ids, ids[1:]):  # ...and count its new ones
                    pairs[pair] += freq[n]
                    where[pair].add(n)
            if verbose and len(merges) % 500 == 0:
                print(f"  {len(merges)} merges, vocab {first + 256 + len(merges)}")
        return cls(merges)

    @staticmethod
    def merge(ids, pair, new):
        """Replace every neighbour pair (a, b) in ids by the token `new`."""
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                out.append(new)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # ------------------------------------------------------------------ encoding
    @staticmethod
    def split(text):
        return SPLIT.findall(text.lower())

    def encode_chunk(self, chunk):
        """One chunk -> token ids: start from its bytes, apply the learned merges, earliest learned first."""
        if chunk not in self.cache:
            ids = [self.first_byte + x for x in chunk.encode("utf-8")]
            while len(ids) > 1:
                pair = min(zip(ids, ids[1:]), key=lambda p: self.rank.get(p, float("inf")))
                if pair not in self.rank:
                    break
                ids = self.merge(ids, pair, self.first_byte + 256 + self.rank[pair])
            self.cache[chunk] = ids
        return self.cache[chunk]

    def encode(self, text):
        return [i for chunk in self.split(text) for i in self.encode_chunk(chunk)]

    def encode_pair(self, context, candidate):
        return [self.cls_id] + self.encode(context) + [self.sep_id] + self.encode(candidate)

    def decode(self, ids):
        return b"".join(self.pieces[i] for i in ids if i != self.pad_id).decode("utf-8", errors="replace")

    def tokens(self, text):
        """The pieces a text is cut into, to look at them."""
        return [self.itos[i] for i in self.encode(text)]

    def __len__(self):
        return len(self.pieces)

    # ------------------------------------------------------------------ saving
    def to_dict(self):
        return {"type": "bpe", "merges": [list(m) for m in self.merges]}

    @classmethod
    def from_dict(cls, d):
        assert d.get("type") == "bpe", "this checkpoint was made with the old word tokenizer: train it again"
        return cls(d["merges"])

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


def tokenizer_from_dict(d):
    """Rebuild the tokenizer saved in a checkpoint (word or BPE)."""
    return WTokenizer.from_dict(d) if d.get("type", "word") == "word" else BPETokenizer.from_dict(d)


def build_tokenizer(kind, texts, vocab_size=4096, min_freq=1, keep=()):
    """
    A new tokenizer learned from `texts`.
        bpe:  `vocab_size` tokens; `keep` (the option texts) is added to the training text
        word: every word seen at least `min_freq` times, plus every word of `keep`
    """
    if kind == "bpe":
        return BPETokenizer.train(list(texts) + list(keep), vocab_size)
    if kind == "word":
        return WTokenizer.from_texts(texts, min_freq=min_freq, keep=keep)
    raise ValueError(f"unknown tokenizer {kind!r}: use 'bpe' or 'word'")
