"""Training utilities shared by pre-training (scripts/train.py) and fine-tuning (scripts/finetune.py)."""
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


def train(model, optimizer, cfg, get_batch_fns, iter_num=0, best_val_loss=float('inf'), ckpt_extra=None):
    """
    The optimisation loop: gradient accumulation, grad clipping, lr schedule,
    periodic evaluation and checkpointing (best val loss -> `{out_dir}/{ckpt_name}`).
    Returns the best validation loss.
    """
    ckpt_path = os.path.join(cfg.out_dir, cfg.ckpt_name)
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
            print(f"Val-Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
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

        if cfg.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg.log_interval == 0:
            lossf = loss.item() * cfg.grad_accum_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, lr {lr:.2e}, time {dt * 1000:.2f}ms")
        iter_num += 1

    return best_val_loss


def build_optimizer(model, cfg, device):
    return model.configure_optimizers(cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2),
                                      torch.device(device).type)
