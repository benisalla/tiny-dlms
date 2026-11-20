import torch

from conftest import load_script
from tiny_dlm import Decision, WordTokenizer, collate, load_csv, make_chunks, mask_tokens


def test_tokenizer_round_trip_and_unknown_words(tokenizer):
    assert tokenizer.decode(tokenizer.encode("It is raining")) == "it is raining"
    assert tokenizer.encode("it is snowing")[-1] == tokenizer.unk_id
    assert WordTokenizer.from_dict(tokenizer.to_dict()).itos == tokenizer.itos


def test_pair_layout(tokenizer):
    ids = tokenizer.encode_pair("it is cold", "wear a coat")
    assert ids[0] == tokenizer.cls_id and ids[4] == tokenizer.sep_id
    assert tokenizer.decode(ids) == "<cls> it is cold <sep> wear a coat"


def test_collate_pads_and_builds_targets(tokenizer, decisions):
    idx, cand_mask, targets = collate(decisions, tokenizer)
    assert idx.shape[:2] == (3, 4)
    assert cand_mask.tolist() == [[True, True, False, False], [True] * 4, [True, True, True, False]]
    assert (idx[:, :, 0] == tokenizer.cls_id).all()  # every slot, even an empty one, starts with [CLS]
    assert torch.allclose(targets.sum(-1), torch.ones(3))
    assert targets[1].tolist() == [0.5, 0.0, 0.5, 0.0]  # two right answers -> 0.5 each


def test_csv_round_trip(data_dir, decisions):
    assert load_csv(data_dir / "val.csv") == decisions


# ---------------------------------------------------------------- bAbI -> decisions (offline)

STORY = {  # first story of bAbI task 1, as returned by the dataset viewer API
    "text": ["Mary moved to the bathroom.", "John went to the hallway.", "Where is Mary?",
             "Daniel went back to the hallway.", "Sandra moved to the garden.", "Where is Daniel?"],
    "type": [0, 0, 1, 0, 0, 1],
    "answer": ["", "", "bathroom", "", "", "hallway"],
}
PLACES = {"bathroom", "bedroom", "garden", "hallway", "kitchen", "office"}


def test_babi_story_becomes_one_decision_per_question():
    prep = load_script("prepare_babi")
    qs = list(prep.questions(STORY))
    assert [q[1:] for q in qs] == [("Where is Mary?", "bathroom"), ("Where is Daniel?", "hallway")]
    assert qs[1][0].count(".") == 4  # the second question sees all four sentences before it

    d = prep.to_decision(*qs[1], PLACES)
    assert d["context"].endswith("sandra moved to the garden . where is daniel ?")
    assert d["candidates"] == "bathroom | hallway | garden"  # places of the story, in order of mention
    assert d["answers"] == "hallway"


def test_babi_falls_back_to_all_answers():
    prep = load_script("prepare_babi")
    d = prep.to_decision("Mary moved to the bathroom.", "Where is Mary?", "bathroom", PLACES)
    assert d["candidates"].split(" | ") == sorted(PLACES)  # one place only -> every possible place
    yes_no = prep.to_decision("Mary moved to the bathroom.", "Is Mary in the garden?", "no", {"yes", "no"})
    assert yes_no["candidates"] == "no | yes"


# ---------------------------------------------------------------- pre-training data


def test_mask_tokens(tokenizer):
    import random
    chunks = [[tokenizer.cls_id] + tokenizer.encode("i am hungry and i am tired . go to sleep .")] * 50
    idx, targets = mask_tokens(chunks, tokenizer, mask_prob=0.15, rng=random.Random(0))
    picked = targets != -1
    assert (picked.sum(1) >= 1).all()  # at least one word to guess in every row
    assert not picked[:, 0].any()  # [CLS] is never masked
    assert (idx[picked] == tokenizer.mask_id).float().mean() > 0.6  # ~80% of picked words become [MASK]
    original = torch.tensor(chunks)
    assert (targets[picked] == original[picked]).all()  # targets are the original words
    assert (idx[~picked] == original[~picked]).all()  # words that were not picked are untouched
    assert 0.08 < picked[:, 1:].float().mean() < 0.25  # about 15% picked


def test_make_chunks_respects_block_size(tokenizer):
    text = " ".join(["it is raining ."] * 40)  # 160 tokens
    chunks = make_chunks([text], tokenizer, block_size=32)
    assert all(len(c) <= 32 and c[0] == tokenizer.cls_id for c in chunks)
    assert sum(len(c) - 1 for c in chunks) == 160  # nothing lost


def test_sample_candidates_keeps_the_answers_and_caps_the_options():
    import random
    from tiny_dlm.data import sample_candidates
    d = Decision("which intent", [f"intent {i}" for i in range(151)], ["intent 42"])
    for seed in range(20):
        s = sample_candidates(d, 16, random.Random(seed))
        assert len(s.candidates) == 16 and "intent 42" in s.candidates
        assert len(set(s.candidates)) == 16 and set(s.candidates) <= set(d.candidates)
    assert sample_candidates(d, 0) is d  # 0 = keep all


def test_multitask_batch_comes_from_one_dataset(tokenizer, decisions):
    import random
    from tiny_dlm.data import get_multitask_batch
    two = [d for d in decisions if len(d.candidates) == 2]
    many = [d for d in decisions if len(d.candidates) > 2]
    seen = set()
    for seed in range(30):
        idx, cand_mask, targets = get_multitask_batch({"two": two, "many": many}, 8, tokenizer, "cpu",
                                                      random.Random(seed))
        seen.add(idx.shape[1])
        assert torch.allclose(targets.sum(-1), torch.ones(len(targets)))
    assert seen == {2, 4}  # both datasets are used, and a batch is never padded to the other one's size


def test_tokenizer_min_freq_keeps_option_words():
    from tiny_dlm import WordTokenizer
    tok = WordTokenizer.from_texts(["the cat sat", "the dog sat"], min_freq=2, keep=["apple pay"])
    assert "the" in tok.stoi and "sat" in tok.stoi and "cat" not in tok.stoi  # seen once -> <unk>
    assert "apple" in tok.stoi and "pay" in tok.stoi  # option words are always kept
