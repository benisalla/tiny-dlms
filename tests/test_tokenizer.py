"""The two tokenizers: a few inputs and their expected outputs."""
from tiny_dlm import BPETokenizer, WTokenizer, build_tokenizer
from tiny_dlm.tokenizer import tokenizer_from_dict

TEXTS = ["the cat sat on the mat", "the cat ate", "a cat and a hat"]


def test_bpe_merges_the_most_frequent_pair_first():
    tok = BPETokenizer.train(["ab ab ab", "ab"], vocab_size=5 + 256 + 1)  # room for exactly one merge
    assert tok.itos[-1] == "ab"  # "a"+"b" appears 4 times: the first merge
    assert tok.tokens("ab ab") == ["ab", " ", "ab"]  # " ab": the space stays alone, "ab" is merged


def test_bpe_common_words_become_one_token():
    tok = BPETokenizer.train(TEXTS * 10, vocab_size=300)
    assert tok.tokens("the cat") == ["the", " cat"]
    assert len(tok.tokens("zebra")) > 1  # never seen: built from pieces, never <unk>


def test_bpe_encode_then_decode_gives_the_text_back():
    tok = BPETokenizer.train(TEXTS, vocab_size=300)
    for text in ["the cat sat", "café, 42 zebras!", "where is mary ?"]:
        assert tok.decode(tok.encode(text)) == text
    assert tok.unk_id not in tok.encode("anything at all ✓")


def test_pair_layout_is_the_same_for_both():
    for tok in (BPETokenizer.train(TEXTS, vocab_size=300), WTokenizer.from_texts(TEXTS)):
        ids = tok.encode_pair("the cat", "sat")
        assert ids[0] == tok.cls_id and tok.sep_id in ids
        assert tok.decode(ids[ids.index(tok.sep_id) + 1:]).strip() == "sat"


def test_save_and_load_both_kinds():
    for kind in ("bpe", "word"):
        tok = build_tokenizer(kind, TEXTS, vocab_size=300)
        back = tokenizer_from_dict(tok.to_dict())
        assert type(back) is type(tok) and back.encode("the cat sat") == tok.encode("the cat sat")
    assert isinstance(tokenizer_from_dict({"itos": WTokenizer.from_texts(TEXTS).itos}), WTokenizer)  # old checkpoints
