"""
Train ImageDLM to drive: from the camera frame, pick the teacher's option (behaviour cloning).

    python scripts/car_record.py          # data/car/{train,val}.pt
    python scripts/car_train.py           # -> checkpoints/car.pt
    python scripts/car_drive.py           # drive on tracks never seen in training

The loss is the same decision loss as the text model: a softmax over the 15 options, cross-entropy with
the teacher's option. Every config field is a flag (--n_layer 4 --n_embd 128 --max_iters 4000 ...).
"""
import os
import random

import torch

from tiny_dlm.config import ModelConfig, TrainConfig
from tiny_dlm.runtime import parse_config_args, resolve_device, setup_torch
from tiny_dlm.tokenizer import WordTokenizer
from tiny_dlm.training import build_optimizer, train
from tiny_dlm.vision import ImageDLM, encode_options

DEFAULTS = dict(data_dir="./data/car", ckpt_name="car.pt", batch_size=64, max_iters=3000, lr_decay_iters=3000,
                warmup_iters=100, eval_interval=250, eval_iters=10, log_interval=100,
                block_size=48)  # 36 patches + "[SEP] steer medium right and accelerate"


def load_split(path):
    data = torch.load(path, weights_only=True)
    return data["images"], data["labels"], data["option_texts"]


def make_batch_fn(images, labels, option_idx, batch_size, device, rng):
    """A random batch: (images, idx, cand_mask, targets). Every frame has the same 15 options."""
    k = option_idx.size(0)  # options per frame
    def get_batch():
        rows = torch.tensor(rng.sample(range(len(labels)), min(batch_size, len(labels))))
        b = len(rows)
        targets = torch.zeros(b, k)  # (b, k): 1 on the teacher's option
        targets[torch.arange(b), labels[rows]] = 1.0
        return (images[rows].to(device), option_idx.expand(b, -1, -1),
                torch.ones(b, k, dtype=torch.bool, device=device), targets.to(device))
    return get_batch


@torch.no_grad()
def option_accuracy(model, images, labels, option_idx, device, batch_size=256):
    """Top-1: how often the model picks exactly the teacher's option. Also 'close': the same steering +-1 notch."""
    model.eval()
    choices = []
    for s in range(0, len(labels), batch_size):
        img = images[s:s + batch_size].to(device)
        b, k = len(img), option_idx.size(0)
        choice, _ = model.decide(img, option_idx.expand(b, -1, -1), torch.ones(b, k, dtype=torch.bool, device=device))
        choices.append(choice.cpu())
    model.train()
    choice = torch.cat(choices)
    exact = (choice == labels).float().mean().item()
    # options 2s and 2s+1 share steering level s (accelerate / coast); the last option is "brake"
    brake = option_idx.size(0) - 1
    steer = lambda c: torch.where(c == brake, torch.full_like(c, -10), c // 2)  # noqa: E731 (brake is only close to brake)
    close = ((steer(choice) - steer(labels)).abs() <= 1).float().mean().item()
    return exact, close


def main(argv=None):
    cfg, model_cfg = parse_config_args(TrainConfig, ModelConfig, description=__doc__, argv=argv, defaults=DEFAULTS)
    setup_torch(cfg.seed)
    device = resolve_device(cfg.device)

    train_images, train_labels, option_texts = load_split(os.path.join(cfg.data_dir, "train.pt"))
    val_images, val_labels, _ = load_split(os.path.join(cfg.data_dir, "val.pt"))
    tokenizer = WordTokenizer.from_texts(option_texts)  # 12 words: steer, slight, left, accelerate, ...
    option_idx = encode_options(option_texts, tokenizer, device)  # (k = 15 options, s tokens)

    model_cfg.vocab_size, model_cfg.pad_id, model_cfg.sep_id = len(tokenizer), tokenizer.pad_id, tokenizer.sep_id
    model = ImageDLM(model_cfg).to(device)
    print(f"number of parameters: {model.get_num_params() / 1e3:.1f}K | frames: train {len(train_labels):,}, "
          f"val {len(val_labels):,} | {len(option_texts)} options | device: {device}")

    optimizer = build_optimizer(model, cfg, device)
    rng = random.Random(cfg.seed)
    get_batch_fns = {"train": make_batch_fn(train_images, train_labels, option_idx, cfg.batch_size, device, rng),
                     "val": make_batch_fn(val_images, val_labels, option_idx, cfg.batch_size, device, rng)}
    best = train(model, optimizer, cfg, get_batch_fns,
                 ckpt_extra={"stage": "car", "tokenizer": tokenizer.to_dict(), "option_texts": option_texts,
                             "image_args": model.image_args})

    exact, close = option_accuracy(model, val_images, val_labels, option_idx, device)
    print(f"done, best val loss {best:.4f} | val: teacher's exact option {exact:.1%}, "
          f"steering within one notch {close:.1%} -> {os.path.join(cfg.out_dir, cfg.ckpt_name)}")
    return {"best_val_loss": best, "val_exact": exact, "val_close": close}


if __name__ == "__main__":
    main()
