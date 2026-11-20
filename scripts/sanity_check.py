"""
Sanity checks for the network and the whole pipeline. Fast (seconds, CPU), no real training.

Run this before a long training run (on Colab / Kaggle): if one check fails, training would waste hours.

    python scripts/sanity_check.py                       # on data/babi
    python scripts/sanity_check.py --data_dir data/babi --threads 2

The checks (the classic "make sure the network can learn at all" list):

    data      1. every answer is one of its candidates, every input fits in block_size
              2. no test / val context also appears in the training set (no leak)
              3. val / test words are known to the tokenizer (few <unk>)
              3b. look at the data: one input decoded back to words, exactly as the network gets it
    model     4. loss at init = loss of random guessing   (decisions: mean log k, words: log vocab)
              5. every parameter gets a finite, non-zero gradient   (nothing is disconnected)
              5b. activations stay healthy from block to block (no explosion, no dead layer)
              5c. one optimizer step moves the weights by a sane amount (learning rate not absurd)
              5d. the same seed gives the same loss
              6. decisions in a batch do not leak into each other (padding of k and s changes nothing)
              7. shuffling the candidates only shuffles the probabilities
              8. any number of candidates works: 2, 3, 7, 10 -> probabilities sum to 1
    learning  9. the model can memorise one small batch of decisions (loss -> ~0, 100% correct)
             10. the model can memorise one small batch of masked words (pre-training)
    pipeline 11. save -> load gives exactly the same outputs
"""
import argparse
import math
import os
import random
import tempfile

import torch

from tiny_dlm.checkpoint import load_checkpoint, save_checkpoint
from tiny_dlm.config import ModelConfig
from tiny_dlm.data import Decision, all_texts, collate, load_csv, load_text, make_chunks, mask_tokens
from tiny_dlm.model import TinyDLM
from tiny_dlm import sanity
from tiny_dlm.sanity import check
from tiny_dlm.tokenizer import WordTokenizer


def build_model(tokenizer, block_size, **overrides):
    args = dict(vocab_size=len(tokenizer), block_size=block_size, pad_id=tokenizer.pad_id,
                sep_id=tokenizer.sep_id, drop_rate=0.0)  # no dropout: checks must be deterministic
    args.update(overrides)
    return TinyDLM(ModelConfig(**args))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data/babi")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads used by the checks")
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    sanity.RESULTS.clear()

    # ------------------------------------------------------------------ data
    print("data")
    train = load_csv(os.path.join(args.data_dir, "train.csv"))
    val = load_csv(os.path.join(args.data_dir, "val.csv"))
    test_path = os.path.join(args.data_dir, "test.csv")
    test = load_csv(test_path) if os.path.exists(test_path) else []
    pre_path = os.path.join(args.data_dir, "pretrain_train.txt")
    texts = load_text(pre_path) if os.path.exists(pre_path) else []
    tokenizer = WordTokenizer.from_texts(list(all_texts(train)) + texts)

    every = train + val + test
    longest = max(len(tokenizer.encode_pair(d.context, c)) for d in every for c in d.candidates)
    check("1. answers are candidates, inputs fit in block_size",
          all(set(d.answers) <= set(d.candidates) for d in every) and longest <= args.block_size,
          f"{len(train)} train / {len(val)} val / {len(test)} test, longest input {longest} <= {args.block_size}")

    # bAbI's generator repeats some tiny 2-sentence stories, even across its official splits (~2%).
    # A real leak (e.g. building val from train) would share far more.
    seen = {d.context for d in train}
    leaks = sum(d.context in seen for d in val + test)
    share = leaks / max(1, len(val + test))
    check("2. val/test are not copies of the training set", share < 0.05,
          f"{leaks} shared contexts ({share:.1%}), all tiny stories repeated by bAbI itself" if leaks else "0 shared")

    words = [w for d in val + test for w in tokenizer.split(d.context + " " + " ".join(d.candidates))]
    unk = sum(w not in tokenizer.stoi for w in words) / max(1, len(words))
    check("3. val/test words are known to the tokenizer", unk < 0.01, f"{unk:.2%} unknown, vocab {len(tokenizer)}")
    shown = collate(train[:1], tokenizer)[0][0, 0].tolist()
    print(f"  [LOOK] 3b. one input exactly as the network gets it: {' '.join(tokenizer.itos[i] for i in shown)}")

    # ------------------------------------------------------------------ model
    print("model")
    model = build_model(tokenizer, args.block_size)
    model.eval()
    batch = collate(rng.sample(train, 64), tokenizer)
    with torch.no_grad():
        _, loss = model(*batch)
    uniform = batch[1].sum(-1).float().log().mean().item()  # random guessing: -log(1/k), averaged
    check("4a. decision loss at init = random guessing", abs(loss.item() - uniform) < 0.1,
          f"loss {loss.item():.3f}, mean log k {uniform:.3f}")
    if texts:
        chunks = make_chunks(texts[:64], tokenizer, args.block_size)
        mlm = mask_tokens(chunks[:32], tokenizer, rng=rng)
        with torch.no_grad():
            _, mlm_loss = model(mlm[0], None, mlm[1])
        check("4b. masked-word loss at init = random guessing", abs(mlm_loss.item() - math.log(len(tokenizer))) < 0.5,
              f"loss {mlm_loss.item():.3f}, log vocab {math.log(len(tokenizer)):.3f}")

    model.train()
    model.zero_grad()
    _, loss = model(*batch)
    loss.backward()
    if texts:  # mlm_head is only used by pre-training (it is tied to wte anyway)
        model(mlm[0], None, mlm[1])[1].backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    check("5. every parameter gets a gradient", not dead and finite, f"no gradient: {dead}" if dead else
          f"{sum(1 for _ in model.parameters())} tensors, {model.get_num_params() / 1e3:.1f}K params")
    model.zero_grad(set_to_none=True)
    run = lambda m: m(*batch)  # noqa: E731
    sanity.check_activations("5b. activations stay healthy", model, model.transformer.h, run)
    sanity.check_update_ratio("5c. one optimizer step moves the weights a sane amount", model, run)
    sanity.check_reproducible("5d. the same seed gives the same loss",
                              lambda: build_model(tokenizer, args.block_size), run)

    model.eval()
    one = [d for d in train if len(d.candidates) == 2][0]
    big = max(train[:500], key=lambda d: (len(d.candidates), len(d.context)))
    with torch.no_grad():
        alone = model(*collate([one], tokenizer)[:2])[0][0, :2]
        crowd = model(*collate([one, big], tokenizer)[:2])[0][0, :2]  # padded to big's k and s
    check("6. decisions in a batch do not leak into each other", torch.allclose(alone, crowd, atol=1e-5),
          f"max difference {(alone - crowd).abs().max().item():.1e}")

    d = big
    perm = list(range(len(d.candidates)))
    rng.shuffle(perm)
    shuffled = Decision(d.context, [d.candidates[i] for i in perm], d.answers)
    with torch.no_grad():
        p = torch.softmax(model(*collate([d], tokenizer)[:2])[0], -1)[0]
        q = torch.softmax(model(*collate([shuffled], tokenizer)[:2])[0], -1)[0]
    check("7. shuffling candidates only shuffles probabilities", torch.allclose(p[perm], q, atol=1e-5))

    places = ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]
    sums = []
    for k in (2, 3, 7, 10):
        cands = (places * 2)[:k]
        with torch.no_grad():
            probs = torch.softmax(model(*collate([Decision(one.context, cands, cands[:1])], tokenizer)[:2])[0], -1)
        sums.append(probs.sum().item())
    check("8. any number of candidates works (k = 2, 3, 7, 10)", all(abs(s - 1) < 1e-5 for s in sums),
          "probabilities sum to " + ", ".join(f"{s:.3f}" for s in sums))

    # ------------------------------------------------------------------ learning
    print("learning")
    small = rng.sample(train, 16)
    small_batch = collate(small, tokenizer)
    t = small_batch[2]
    floor = max(0.0, -(t * t.clamp(min=1e-12).log()).sum(-1).mean().item())  # best possible loss (> 0 with several right answers)
    for pool in ("candidate", "cls"):
        torch.manual_seed(args.seed)
        m = build_model(tokenizer, args.block_size, pool=pool)
        final, steps = sanity.overfit(m, lambda m: m(*small_batch), steps=300, target=floor + 0.05)
        m.eval()
        with torch.no_grad():
            choice = m(*small_batch[:2])[0].argmax(-1)
        acc = (small_batch[2][torch.arange(len(small)), choice] > 0).float().mean().item()
        name = f"9{'a' if pool == 'candidate' else 'b'}. memorise 16 decisions (pool={pool})"
        if pool == "candidate":
            check(name, final < floor + 0.05 and acc == 1.0,
                  f"loss {final:.3f} (best possible {floor:.3f}) after {steps} steps, {acc:.0%} correct")
        else:  # informative only: shows why "candidate" is the default
            print(f"  [INFO] {name}  (loss {final:.3f} after {steps} steps, {acc:.0%} correct)")

    if texts:
        torch.manual_seed(args.seed)
        m = build_model(tokenizer, args.block_size)
        fixed = mask_tokens(chunks[:8], tokenizer, rng=rng)
        final, steps = sanity.overfit(m, lambda m: m(fixed[0], None, fixed[1]), steps=300, target=0.05)
        check("10. memorise 8 masked-word examples", final < 0.05, f"loss {final:.3f} after {steps} steps")

    # ------------------------------------------------------------------ pipeline
    print("pipeline")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pt")
        save_checkpoint(path, model, extra={"tokenizer": tokenizer.to_dict()})
        loaded, tok2, _ = load_checkpoint(path)
        model.eval()
        with torch.no_grad():
            same = torch.equal(model(*batch[:2])[0], loaded(*batch[:2])[0])
    check("11. save -> load gives the same outputs", same and tok2.itos == tokenizer.itos)

    return sanity.summary()


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
