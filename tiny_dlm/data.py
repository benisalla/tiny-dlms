"""
Decision data: load the CSV files and build padded batches.

CSV columns (see scripts/make_toy_data.py):
    context     "i am hungry and it is raining"
    candidates  "take an umbrella | eat some food | go to sleep"
    answers     "take an umbrella | eat some food"
"""
import csv
import random
from dataclasses import dataclass

import torch

from tiny_dlm.tokenizer import SPECIALS


@dataclass
class Decision:
    context: str
    candidates: list
    answers: list


def load_csv(path):
    decisions = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            candidates = [c.strip() for c in row["candidates"].split("|")]
            answers = [a.strip() for a in row["answers"].split("|")]
            assert set(answers) <= set(candidates), f"an answer is not among the candidates: {row}"
            decisions.append(Decision(row["context"].strip(), candidates, answers))
    return decisions


def all_texts(decisions):
    """Every context and candidate, to build the tokenizer."""
    for d in decisions:
        yield d.context
        yield from d.candidates


def collate(decisions, tokenizer, device="cpu"):
    """
        Turn a list of decisions into padded tensors.

        Tensor          Type            Shape
        ===========================================================================
        idx             long            (batch_size, max_candidates, max_seq_len)
        cand_mask       bool            (batch_size, max_candidates)
        targets         float           (batch_size, max_candidates)
        ===========================================================================

        Empty candidate slots (a decision with fewer candidates than the largest one in the batch)
        hold a lone [CLS] token, never an all-padding row, which would leave attention nothing to look at.
        Targets are uniform over the correct answers: one right answer -> 1.0, two -> 0.5 each.
    """
    B = len(decisions)
    K = max(len(d.candidates) for d in decisions)
    seqs = [[tokenizer.encode_pair(d.context, c) for c in d.candidates] for d in decisions]
    T = max(len(s) for row in seqs for s in row)

    idx = torch.full((B, K, T), tokenizer.pad_id, dtype=torch.long)
    idx[:, :, 0] = tokenizer.cls_id
    cand_mask = torch.zeros(B, K, dtype=torch.bool)
    targets = torch.zeros(B, K)
    for b, d in enumerate(decisions):
        for k, (cand, seq) in enumerate(zip(d.candidates, seqs[b])):
            idx[b, k, :len(seq)] = torch.tensor(seq)
            cand_mask[b, k] = True
            targets[b, k] = float(cand in d.answers)
        targets[b] /= targets[b].sum()
    return idx.to(device), cand_mask.to(device), targets.to(device)


def sample_candidates(decision, max_candidates, rng=random):
    """
        Keep every right answer plus random wrong options, at most `max_candidates` in total (0 -> keep all).
        Used for training only: CLINC150 has 151 options, and each one is a full pass through the encoder.
        The model never sees a fixed list, so it still works with all 151 at test time (no layer depends on K).

            "translate | transfer | weather | ... (151)"   ->   "weather | translate | pay bill | ... (16)"
    """
    if not max_candidates or len(decision.candidates) <= max_candidates:
        return decision
    wrong = [c for c in decision.candidates if c not in decision.answers]
    n_wrong = max(1, max_candidates - len(decision.answers))
    candidates = list(decision.answers) + rng.sample(wrong, min(n_wrong, len(wrong)))
    rng.shuffle(candidates)  # the right answer must not always be first (the candidate layer ignores order anyway)
    return Decision(decision.context, candidates, decision.answers)


def get_batch(decisions, batch_size, tokenizer, device, rng=random, max_candidates=0):
    """A random batch of decisions: (idx, cand_mask, targets)."""
    batch = rng.sample(decisions, min(batch_size, len(decisions)))
    return collate([sample_candidates(d, max_candidates, rng) for d in batch], tokenizer, device)


def get_multitask_batch(datasets, batch_size, tokenizer, device, rng=random, max_candidates=0):
    """
        Several datasets, one model: pick ONE dataset (in proportion to its size), then a batch from it.
        One dataset per batch keeps the batch rectangle small (a bAbI decision padded to 151 options would waste
        ~30x the compute). Over many steps the model sees every dataset.

        datasets: {"clinc150": [Decision, ...], "csqa": [...], ...}
    """
    names = list(datasets)
    name = rng.choices(names, weights=[len(datasets[n]) for n in names])[0]
    return get_batch(datasets[name], batch_size, tokenizer, device, rng, max_candidates)


# ---------------------------------------------------------------- pre-training (masked words)


def load_text(path):
    """One text (a story) per line."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def make_chunks(texts, tokenizer, block_size):
    """
        Cut every text into pieces of at most block_size tokens, each starting with [CLS]
        (the same start as a decision input, so pre-training and fine-tuning see the same layout).
    """
    n = block_size - 1
    chunks = []
    for text in texts:
        ids = tokenizer.encode(text)
        chunks += [[tokenizer.cls_id] + ids[i:i + n] for i in range(0, len(ids), n)]
    return chunks


def mask_tokens(chunks, tokenizer, mask_prob=0.15, rng=random):
    """
        BERT masking. Pick ~mask_prob of the words (at least one per row); the model must guess them.
        Of the picked words: 80% become [MASK], 10% a random word, 10% stay the same
        (so the model can't just learn "only [MASK] positions matter").

        Tensor          Type            Shape
        ===========================================================================
        idx             long            (batch_size, max_len)   the input, with masks
        targets         long            (batch_size, max_len)   original id where picked, -1 elsewhere
        ===========================================================================
    """
    T = max(len(c) for c in chunks)
    idx = torch.full((len(chunks), T), tokenizer.pad_id, dtype=torch.long)
    targets = torch.full((len(chunks), T), -1, dtype=torch.long)
    first_word = len(SPECIALS)  # random replacements are real words, never special tokens
    for b, chunk in enumerate(chunks):
        idx[b, :len(chunk)] = torch.tensor(chunk)
        positions = list(range(1, len(chunk)))  # never mask [CLS]
        picked = [p for p in positions if rng.random() < mask_prob] or [rng.choice(positions)]
        for p in picked:
            targets[b, p] = chunk[p]
            r = rng.random()
            if r < 0.8:
                idx[b, p] = tokenizer.mask_id
            elif r < 0.9:
                idx[b, p] = rng.randrange(first_word, len(tokenizer))
    return idx, targets


def get_mlm_batch(chunks, batch_size, tokenizer, device, mask_prob=0.15, rng=random):
    """
    A random batch for pre-training: (idx, None, targets).
    The None is the cand_mask slot of model.forward(idx, cand_mask, targets): no candidates -> pre-training mode.
    """
    idx, targets = mask_tokens(rng.sample(chunks, min(batch_size, len(chunks))), tokenizer, mask_prob, rng)
    return idx.to(device), None, targets.to(device)
