"""
Build notebooks/Tiny_DLM(Training).ipynb: one self-contained notebook to train Tiny DLM on Colab / Kaggle.

The classes and functions are copied from the tiny_dlm package (so the notebook never drifts from the code),
and the training loop is written out in the notebook. Re-run this after changing the package:

    python scripts/build_notebook.py
"""
import ast
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "Tiny_DLM(Training).ipynb")


def source(path, *names):
    """The source code of the given top-level classes / functions of a file, in that order."""
    text = open(os.path.join(ROOT, path), encoding="utf-8").read()
    nodes = {n.name: n for n in ast.parse(text).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    parts = []
    for name in names:
        node = nodes[name]
        start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        parts.append("\n".join(text.splitlines()[start - 1:node.end_lineno]))
    return "\n\n\n".join(parts)


def constants(path, *names):
    """Top-level assignments (e.g. PAD, UNK, ... = ...) of a file."""
    text = open(os.path.join(ROOT, path), encoding="utf-8").read()
    out = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, (ast.Name, ast.Tuple)) and
                                                 any(n in ast.get_source_segment(text, t) for n in names)
                                                 for t in node.targets):
            out.append("\n".join(text.splitlines()[node.lineno - 1:node.end_lineno]))
    return "\n".join(out)


cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    text = re.sub(r"see scripts/train\.py and scripts/finetune\.py", "see the two training sections below", text)
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


# ====================================================================== imports
md("#Importing required libraries.")
code("""
import math
import inspect
import os
import re
import csv
import json
import time
import random
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
from torch.nn import functional as F
""")

md("#####Where to save (Colab / Kaggle / local)")
code("""
IN_COLAB = os.path.exists("/content")
IN_KAGGLE = os.path.exists("/kaggle/working")
USE_DRIVE = True  # Colab: keep data + checkpoints on Google Drive, so they survive a disconnect

if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    root_dir = '/content/drive/MyDrive/TINY_DLM_FROM_SCRATCH'
elif IN_COLAB:
    root_dir = '/content/TINY_DLM_FROM_SCRATCH'
elif IN_KAGGLE:
    root_dir = '/kaggle/working/TINY_DLM_FROM_SCRATCH'
else:
    root_dir = './TINY_DLM_FROM_SCRATCH'

data_dir = os.path.join(root_dir, 'data', 'babi')
checkpts_dir = os.path.join(root_dir, 'check_points')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(checkpts_dir, exist_ok=True)
print("saving everything in", root_dir)
""")

# ====================================================================== idea
md("""
#What is a Decision Model? 🤔

An LLM is a **writer**: it invents the answer word by word, choosing every word from the whole vocabulary.
A decision model is a **judge**: it is given the question **and the possible answers**, and gives every answer a score.

```
context:     mary moved to the bathroom . john went to the hallway . where is mary ?
candidates:  bathroom | hallway

  [CLS] context [SEP] bathroom   ->  Transformer  ->  one vector  ->  score_head  ->  4.1  ┐
  [CLS] context [SEP] hallway    ->  Transformer  ->  one vector  ->  score_head  ->  0.2  ┘ softmax -> 0.98 | 0.02
                                    (same weights)                    (same weights)
```

* The candidates are **plain text**, not special tokens.
* No layer has the number of candidates in its shape: `score_head` is `Linear(n_embd, 1)` and is used once per
  candidate. So the same model can choose among 2, 3 or 10 options.
* Two training stages:
  1. **pre-training**: guess hidden words (`[MASK]`), like BERT -> the model learns the language.
  2. **fine-tuning**: softmax over the candidates' scores, cross-entropy with the right answer -> the model learns to decide.
""")

# ====================================================================== model
md("#The Core of Tiny DLM 🤖")
md("""
```
for each candidate i:
    [CLS] context [SEP] candidate_i  ->  wte  ->  EBlock x n_layer (RoPE)  ->  ln_f  ->  mean of the candidate's tokens = z_i
all z_i of one decision              ->  EBlock x n_cand_layer (no positions: candidates are a set)  ->  ln_f
                                     ->  score_head  ->  logit_i  ->  softmax over candidates
pre-training only:                       every token vector  ->  mlm_head (tied to wte)  ->  the hidden word
```
""")
md("#####Configuration")
code(source("tiny_dlm/config.py", "ModelConfig"))
md("#####Layer Norm / RMS Norm (Normalization Layers)")
code(source("tiny_dlm/norm.py", "LayerNorm", "RMSNorm", "build_norm"))
md("#####RoPE (Rotary Position Embedding)")
code(source("tiny_dlm/rope.py", "rotate_half", "RotaryEmbedding"))
md("#####Bidirectional Self Attention")
code(source("tiny_dlm/attention.py", "SelfAttention"))
md("#####MLP (Multi-Layer Perceptron)")
code(constants("tiny_dlm/mlp.py", "ACTIVATIONS") + "\n\n\n" + source("tiny_dlm/mlp.py", "MLP"))
md("#####Block (communication + computation)")
code(source("tiny_dlm/block.py", "EBlock"))
md("#####Body of Tiny DLM")
code(source("tiny_dlm/model.py", "TinyDLM"))

# ====================================================================== data
md("#Data: bAbI 📚")
md("""
[bAbI](https://arxiv.org/abs/1502.05698) is a very simple question-answering benchmark (Facebook AI Research).
Every question becomes one decision: the candidates are the places mentioned in the story.

* **decisions (stage 2):** task 1, 10k version: 9,000 train / 1,000 val / 1,000 test
* **pre-training text (stage 1):** the training stories of all 20 tasks, without the answers (6,897 stories, ~594k words)

The download uses the Hugging Face dataset viewer API (plain JSON, no extra library) and is skipped if the files exist.
""")
md("#####Download and build the dataset")
code(constants("scripts/prepare_babi.py", "API", "MIRROR", "DATASET", "SPLITS", "PAGE") + "\n\n\n" +
     source("scripts/prepare_babi.py", "words", "fetch_stories", "fetch_mirror", "mirror_stories", "questions",
            "story_text", "to_decision", "write"))
code("""
def prepare_babi(out_dir, tasks=(1,), pretrain_tasks=range(1, 21), prefix="en-valid-10k"):
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    raw = {split: [] for split in SPLITS}  # split -> [(task, story text, question, answer)]
    for task in tasks:
        for split in SPLITS:
            for story in fetch_stories(f"{prefix}-qa{task}", split, raw_dir):
                raw[split] += [(task, *q) for q in questions(story)]

    # answer vocabulary per task, from the training split only
    answer_vocab = {}
    for task, _, _, answer in raw["train"]:
        answer_vocab.setdefault(task, set()).add(answer)

    for split, name in SPLITS.items():
        rows = [to_decision(s, q, a, answer_vocab[t]) for t, s, q, a in raw[split]]
        write(os.path.join(out_dir, f"{name}.csv"), rows)

    # pre-training text: training stories only (no val / test story is ever seen in stage 1)
    texts = [story_text(s) for t in tasks for s in fetch_stories(f"{prefix}-qa{t}", "train", raw_dir)]
    texts += mirror_stories(fetch_mirror(raw_dir), set(pretrain_tasks) - set(tasks))
    for name, part in (("pretrain_train", [x for i, x in enumerate(texts) if i % 20]),
                       ("pretrain_val", [x for i, x in enumerate(texts) if i % 20 == 0])):
        with open(os.path.join(out_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.writelines(x + "\\n" for x in part)
        print(f"wrote {len(part)} stories to {name}.txt")


if not os.path.exists(os.path.join(data_dir, "pretrain_val.txt")):
    prepare_babi(data_dir)
else:
    print("dataset already in", data_dir)
""")
md("#####Tokenizers: BPE (byte pair encoding, learned from our data) and the simple word tokenizer, to compare")
code(constants("tiny_dlm/tokenizer.py", "PAD", "SPECIALS", "SPLIT") + "\n\n\n" + source("tiny_dlm/tokenizer.py", "WTokenizer")
     + "\n\n\n" + source("tiny_dlm/tokenizer.py", "BPETokenizer"))
md("#####Data Loader")
code(source("tiny_dlm/data.py", "Decision", "load_csv", "all_texts", "collate", "sample_candidates", "get_batch"))
md("> pre-training batches: BERT masking (80% `[MASK]`, 10% random word, 10% unchanged)")
code(source("tiny_dlm/data.py", "load_text", "make_chunks", "mask_tokens", "get_mlm_batch"))
code("""
train_data = load_csv(os.path.join(data_dir, "train.csv"))
val_data = load_csv(os.path.join(data_dir, "val.csv"))
test_data = load_csv(os.path.join(data_dir, "test.csv"))
pretrain_train = load_text(os.path.join(data_dir, "pretrain_train.txt"))
pretrain_val = load_text(os.path.join(data_dir, "pretrain_val.txt"))

# one vocabulary for both stages: every word of the decisions and of the pre-training text
tokenizer = BPETokenizer.train(list(all_texts(train_data)) + pretrain_train, vocab_size=4096)  # or WTokenizer.from_texts(...)
print(tokenizer.tokens(train_data[0].context)[:12])
print(f"{len(train_data)} / {len(val_data)} / {len(test_data)} decisions | {len(pretrain_train)} stories | vocab {len(tokenizer)}")
print(train_data[0])
""")

# ====================================================================== config
md("#Model Configuration and Hyperparameters")
code("""
seed = 2000
eval_interval = 250
log_interval = 50
eval_iters = 20

# model
block_size = 128   # longest "[CLS] context [SEP] candidate" (bAbI task 1: 75 tokens)
n_layer = 4        # blocks that read each (context, candidate) pair
n_cand_layer = 1   # blocks where the candidates of one decision look at each other
n_head = 4
n_embd = 128
dropout = 0.1
bias = False
pool = "candidate"  # "candidate" (mean of the candidate's tokens) or "cls" (the [CLS] vector, like BERT)

# stage 1: pre-training (masked words)
pt_batch_size = 64
pt_max_iters = 3000
pt_learning_rate = 1e-3
mask_prob = 0.15

# stage 2: fine-tuning (decisions)
ft_batch_size = 64
ft_max_iters = 3000
ft_learning_rate = 5e-4

# adamw optimizer
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay (linear warmup -> cosine -> min_lr)
decay_lr = True
warmup_iters = 100
min_lr_ratio = 0.1  # min_lr = learning_rate * min_lr_ratio

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
""")

# ====================================================================== init
md("#Initialization")
md("""
> AUTOMATIC MIXED PRECISION: on the GPU the forward pass runs in float16 / bfloat16 (faster, less memory).
> With float16 a `GradScaler` scales the loss up so small gradients do not round to zero.
""")
code("""
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

model_args = dict(vocab_size=len(tokenizer), block_size=block_size, n_layer=n_layer, n_cand_layer=n_cand_layer,
                  n_head=n_head, n_embd=n_embd, drop_rate=dropout, bias=bias, pool=pool,
                  pad_id=tokenizer.pad_id, sep_id=tokenizer.sep_id)
print(f"device: {device} | dtype: {dtype}")
""")

# ====================================================================== utils
md("#Utils")
md("> Learning Rate ( Linear warmup -> Cosine Decay -> min_lr )")
code("""
# Learning Rate Update
def get_lr(it, learning_rate, max_iters):
    min_lr = learning_rate * min_lr_ratio
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # goes from 1 to 0
    return min_lr + coeff * (learning_rate - min_lr)


# Estimation of Loss
@torch.no_grad()
def estimate_loss(model, get_batch_fns):
    out = {}
    model.eval()
    for split, get_batch_fn in get_batch_fns.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            with ctx:
                _, loss = model(*get_batch_fn())
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# Accuracy: how often the most likely candidate is a right answer
@torch.no_grad()
def accuracy(model, decisions, batch_size=128):
    model.eval()
    correct = 0
    for i in range(0, len(decisions), batch_size):
        idx, cand_mask, targets = collate(decisions[i:i + batch_size], tokenizer, device)
        with ctx:
            choice, _ = model.decide(idx, cand_mask)
        correct += (targets[torch.arange(len(choice)), choice] > 0).sum().item()
    model.train()
    return correct / len(decisions)


# Save / Load Checkpoints (with the tokenizer, so a checkpoint is all you need to use the model)
def save_checkpts(ckpt_path, model, optimizer, iter_num, best_val_loss):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model.config.to_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'tokenizer': tokenizer.to_dict(),
    }
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path} :)")


def load_checkpts(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = TinyDLM(ModelConfig.from_dict(checkpoint['model_args']))
    model.load_state_dict(checkpoint['model'])
    return model.to(device), checkpoint
""")

md("#####Training Loop (used by both stages)")
code("""
def train_loop(model, get_batch_fns, learning_rate, max_iters, ckpt_path):
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16' and device == 'cuda'))
    best_val_loss = float('inf')
    iter_num = 0

    X = get_batch_fns['train']()
    t0 = time.time()
    while True:
        lr = get_lr(iter_num, learning_rate, max_iters) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluation (Train / val) + checkpoint when val loss improves
        if iter_num % eval_interval == 0 or iter_num == max_iters:
            losses = estimate_loss(model, get_batch_fns)
            print(f"Val-Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    save_checkpts(ckpt_path, model, optimizer, iter_num, best_val_loss)
        if iter_num >= max_iters:
            break

        # forward -> backward -> optimizing
        with ctx:
            _, loss = model(*X)
        X = get_batch_fns['train']()  # next batch while the GPU works
        scaler.scale(loss).backward()

        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt, t0 = t1 - t0, t1
        if iter_num % log_interval == 0:
            print(f"iter {iter_num}: loss {loss.item():.4f}, lr {lr:.2e}, time {dt * 1000:.2f}ms")
        iter_num += 1

    return best_val_loss
""")

# ====================================================================== sanity
md("#Sanity Checks (before spending GPU time)")
md("""
Three classic checks, a few seconds each:
1. **loss at init = random guessing**: `mean(log K)` for decisions (K = number of candidates).
2. **any number of candidates**: the same model gives probabilities that sum to 1 for K = 2, 3, 7, 10.
3. **memorise one small batch**: a working network drives the loss to ~0 on 16 fixed decisions.
""")
code("""
torch.manual_seed(seed)
model = TinyDLM(ModelConfig(**{**model_args, 'drop_rate': 0.0})).to(device)

# 1. loss at init
batch = collate(random.sample(train_data, 64), tokenizer, device)
with torch.no_grad():
    _, loss = model(*batch)
print(f"1. loss at init {loss.item():.3f} | random guessing {batch[1].sum(-1).float().log().mean().item():.3f}")

# 2. any number of candidates
places = ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]
for K in (2, 3, 7, 10):
    cands = (places * 2)[:K]
    idx, cand_mask, _ = collate([Decision(train_data[0].context, cands, cands[:1])], tokenizer, device)
    with torch.no_grad():
        probs = F.softmax(model(idx, cand_mask)[0], dim=-1)
    print(f"2. K = {K:2d}: probabilities sum to {probs.sum().item():.3f}")

# 3. memorise 16 decisions
small_batch = collate(random.sample(train_data, 16), tokenizer, device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
for step in range(300):
    _, loss = model(*small_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if loss.item() < 0.05:
        break
print(f"3. memorised 16 decisions: loss {loss.item():.3f} after {step + 1} steps")
assert loss.item() < 0.05, "the network cannot memorise 16 examples: something is broken, do not train"
del model, optimizer
""")

# ====================================================================== stage 1
md("#Stage 1: Pre-training (guess the hidden words)")
code("""
pt_chunks = {"train": make_chunks(pretrain_train, tokenizer, block_size),
             "val": make_chunks(pretrain_val, tokenizer, block_size)}
pt_batch_fns = {split: (lambda s=split: get_mlm_batch(pt_chunks[s], pt_batch_size, tokenizer, device, mask_prob))
                for split in pt_chunks}

torch.manual_seed(seed)
model = TinyDLM(ModelConfig(**model_args)).to(device)
print(f"number of parameters: {model.get_num_params() / 1e6:.2f}M")

pretrain_ckpt = os.path.join(checkpts_dir, "pretrain.pt")
best = train_loop(model, pt_batch_fns, pt_learning_rate, pt_max_iters, pretrain_ckpt)
print(f"pre-training done, best val loss {best:.4f}")
""")

# ====================================================================== stage 2
md("#Stage 2: Fine-tuning (learn to decide)")
md("""
Same model, new data and new loss. The encoder starts from the pre-trained weights; the candidate layer and
`score_head` learn now. Loss = `-log p(right candidate)`, the softmax being over the **candidates**, not the vocabulary.

Set `init_from = "scratch"` to fine-tune without pre-training and see what pre-training brings.
""")
code("""
init_from = "pretrain"  # "pretrain" or "scratch"

if init_from == "pretrain":
    model, _ = load_checkpts(pretrain_ckpt)
    print("starting from the pre-trained model", pretrain_ckpt)
else:
    torch.manual_seed(seed)
    model = TinyDLM(ModelConfig(**model_args)).to(device)
    print("starting from scratch (no pre-training)")

ft_batch_fns = {"train": lambda: get_batch(train_data, ft_batch_size, tokenizer, device),
                "val": lambda: get_batch(val_data, ft_batch_size, tokenizer, device)}

finetune_ckpt = os.path.join(checkpts_dir, f"finetune_{init_from}.pt")
best = train_loop(model, ft_batch_fns, ft_learning_rate, ft_max_iters, finetune_ckpt)

model, _ = load_checkpts(finetune_ckpt)  # the best checkpoint (lowest val loss)
print(f"fine-tuning done, best val loss {best:.4f}")
print(f"accuracy: train {accuracy(model, train_data):.1%} | val {accuracy(model, val_data):.1%} | test {accuracy(model, test_data):.1%}")
""")

# ====================================================================== ask
md("#Ask the Model 🗳️")
md("The same model, the same story, different numbers of candidates (it was trained with 2 to 6):")
code("""
def ask(context, candidates):
    idx, cand_mask, _ = collate([Decision(context, candidates, candidates[:1])], tokenizer, device)
    choice, probs = model.decide(idx, cand_mask)
    print(f"context: {context}")
    for cand, p in sorted(zip(candidates, probs[0].tolist()), key=lambda x: -x[1]):
        print(f"   {p:5.2f}  {cand}")
    print(f"decision: {candidates[choice.item()]}\\n")


story = "mary moved to the bathroom . john went to the hallway . mary went to the garden . where is mary ?"
ask(story, ["bathroom", "garden"])
ask(story, ["bathroom", "hallway", "garden"])
ask(story, ["bathroom", "hallway", "garden", "kitchen", "office"])
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"wrote {OUT} ({len(cells)} cells)")
