"""Training utilities shared by pre-training (scripts/train.py) and fine-tuning (scripts/finetune.py)."""
import json
import math
import os
import time

import torch

from tiny_dlm.checkpoint import save_checkpoint
from tiny_dlm.data import collate


def get_lr(it, cfg):
    """Linear warmup, then cosine decay down to min_lr."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # goes from 1 to 0
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, get_batch_fns, eval_iters):
    """Average loss over `eval_iters` batches for each split. get_batch_fns = {"train": fn, "val": fn}."""
    out = {}
    model.eval()
    for split, get_batch in get_batch_fns.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            _, loss = model(*get_batch())  # works for both stages: (idx, targets) or (idx, cand_mask, targets)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def accuracy(model, decisions, tokenizer, device, max_sequences=4096):
    """
    Top-1 accuracy with ALL the options of every decision: how often the most likely one is a correct answer.
    Batches hold at most `max_sequences` (context, option) sequences: 4096 = 64 bAbI decisions or 27 CLINC150 ones.
    """
    was_training = model.training
    model.eval()
    correct = 0
    batch_size = min(64, max(1, max_sequences // max(len(d.candidates) for d in decisions)))
    for i in range(0, len(decisions), batch_size):
        idx, cand_mask, targets = collate(decisions[i:i + batch_size], tokenizer, device)
        choice, _ = model.decide(idx, cand_mask)
        correct += (targets[torch.arange(len(choice)), choice] > 0).sum().item()
    model.train(was_training)
    return correct / len(decisions)


class MetricsLog:
    """
    Every number worth plotting, one JSON line per event, in `{out_dir}/{ckpt_name}.metrics.jsonl`:

        {"kind": "train", "step": 100, "loss": 1.21, "lr": 0.001, "grad_norm": 0.8, "ms_per_step": 95}
        {"kind": "eval",  "step": 500, "train_loss": 1.1, "val_loss": 1.2, "val_acc/clinc150": 0.41, ...}

    scripts/plot_metrics.py turns these files into graphs.
    """

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        open(path, "w").close()  # a new run starts a new file

    def write(self, **record):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


def train(model, optimizer, cfg, get_batch_fns, iter_num=0, best_val_loss=float('inf'), ckpt_extra=None,
          eval_fn=None):
    """
    The optimisation loop: gradient accumulation, grad clipping, lr schedule,
    periodic evaluation and checkpointing (best val loss -> `{out_dir}/{ckpt_name}`).
    `eval_fn(model)` -> {"val_acc/babi": 0.9, ...}: extra numbers measured at every evaluation (accuracy...).
    Everything is also written to `{out_dir}/{ckpt_name}.metrics.jsonl` (see MetricsLog).
    Returns the best validation loss.
    """
    ckpt_path = os.path.join(cfg.out_dir, cfg.ckpt_name)
    log = MetricsLog(ckpt_path + ".metrics.jsonl")
    trainable = [p for p in model.parameters() if p.requires_grad]
    model.train()

    batch = get_batch_fns['train']()
    t0 = time.time()
    while True:
        lr = get_lr(iter_num, cfg) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters:
            losses = estimate_loss(model, get_batch_fns, cfg.eval_iters)
            extra = eval_fn(model) if eval_fn else {}
            print(f"Val-Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                  + "".join(f", {k} {v:.3f}" for k, v in extra.items()))
            log.write(kind="eval", step=iter_num, train_loss=losses['train'], val_loss=losses['val'], **extra)
            if losses['val'] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = min(best_val_loss, losses['val'])
                if iter_num > 0:
                    save_checkpoint(ckpt_path, model, optimizer, model.config.to_dict(),
                                    iter_num, best_val_loss, extra=ckpt_extra)

        if iter_num >= cfg.max_iters:
            break

        # forward / backward with gradient accumulation (simulates a bigger batch)
        for _ in range(cfg.grad_accum_steps):
            _, loss = model(*batch)
            loss = loss / cfg.grad_accum_steps
            batch = get_batch_fns['train']()  # next batch
            loss.backward()

        # the gradient norm before clipping: a spike warns that training is getting unstable
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip if cfg.grad_clip else float('inf'))

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg.log_interval == 0:  # .item() waits for the GPU: only every log_interval steps
            lossf = loss.item() * cfg.grad_accum_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, lr {lr:.2e}, grad norm {grad_norm.item():.2f}, "
                  f"time {dt * 1000:.2f}ms")
            log.write(kind="train", step=iter_num, loss=lossf, lr=lr, grad_norm=grad_norm.item(), ms_per_step=dt * 1000)
        iter_num += 1

    if not os.path.exists(ckpt_path):  # the val loss never improved: still save the model we have
        save_checkpoint(ckpt_path, model, optimizer, model.config.to_dict(), iter_num, best_val_loss, extra=ckpt_extra)
    return best_val_loss


def build_optimizer(model, cfg, device):
    return model.configure_optimizers(cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2),
                                      torch.device(device).type)
