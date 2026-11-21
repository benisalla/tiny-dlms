"""
Stage 1 - pre-train the TinyDLM encoder on plain text (data_dir/pretrain_train.txt, pretrain_val.txt).

The encoder learns the language by guessing hidden words (masked-word prediction, like BERT).
It knows nothing about decisions yet; that is stage 2 (scripts/finetune.py).

    python scripts/prepare_babi.py           # builds data/babi/pretrain_*.txt and data/babi/*.csv
    python scripts/train.py                  # -> checkpoints/pretrain.pt
    python scripts/train.py --data_dir data/pretrain     # all datasets (scripts/prepare_data.py)
    python scripts/train.py --n_layer 4 --n_embd 128 --max_iters 3000
    python scripts/train.py --norm_type layernorm --pos_encoding learned --gated_mlp false --activation gelu_tanh
                                             # GPT-2 style blocks instead of Llama style
"""
import os
import random

import torch

from tiny_dlm.config import ModelConfig, TrainConfig
from tiny_dlm.data import get_mlm_batch, load_text, make_chunks, mask_tokens
from tiny_dlm.model import TinyDLM
from tiny_dlm.runtime import parse_config_args, resolve_device, setup_torch
from tiny_dlm.tokenizer import build_tokenizer
from tiny_dlm.training import build_optimizer, train


def main(argv=None):
    cfg, model_cfg = parse_config_args(TrainConfig, ModelConfig, description=__doc__, argv=argv)
    cfg.ckpt_name = cfg.ckpt_name or "pretrain.pt"
    setup_torch(cfg.seed)
    device = resolve_device(cfg.device)

    # connect to dataset
    train_texts = load_text(os.path.join(cfg.data_dir, "pretrain_train.txt"))
    val_texts = load_text(os.path.join(cfg.data_dir, "pretrain_val.txt"))
    # the tokenizer is learned here, from the pre-training text, and fine-tuning reuses it.
    # The option texts (options.txt) are added: an answer must always be written well (bpe) or never be <unk> (word).
    options_path = os.path.join(cfg.data_dir, "options.txt")
    options = load_text(options_path) if os.path.exists(options_path) else []
    tokenizer = build_tokenizer(cfg.tokenizer, train_texts, cfg.bpe_vocab_size, cfg.min_freq, keep=options)
    train_chunks = make_chunks(train_texts, tokenizer, model_cfg.block_size)
    val_chunks = make_chunks(val_texts, tokenizer, model_cfg.block_size)

    model_cfg.vocab_size = len(tokenizer)
    model_cfg.pad_id, model_cfg.sep_id = tokenizer.pad_id, tokenizer.sep_id
    model = TinyDLM(model_cfg).to(device)
    print(f"Rhoa | number of parameters: {model.get_num_params() / 1e3:.1f}K | {cfg.tokenizer} vocab: {len(tokenizer)} | "
          f"texts: {len(train_texts)} -> {len(train_chunks)} chunks | device: {device}")

    optimizer = build_optimizer(model, cfg, device)
    rng = random.Random(cfg.seed)
    get_batch_fns = {
        "train": lambda: get_mlm_batch(train_chunks, cfg.batch_size, tokenizer, device, cfg.mask_prob, rng),
        "val": lambda: get_mlm_batch(val_chunks, cfg.batch_size, tokenizer, device, cfg.mask_prob, rng),
    }
    # masked-word accuracy: how often the hidden word is guessed exactly (fixed validation batch of 256 texts)
    val_idx, val_targets = mask_tokens(val_chunks[:256], tokenizer, cfg.mask_prob, random.Random(0))
    val_idx, val_targets = val_idx.to(device), val_targets.to(device)

    @torch.no_grad()
    def eval_fn(m):
        m.eval()
        guess = m(val_idx)[0].argmax(-1)  # (b, s): the most likely word at every position
        m.train()
        hidden = val_targets != -1
        return {"val_acc/masked_word": (guess[hidden] == val_targets[hidden]).float().mean().item()}

    best = train(model, optimizer, cfg, get_batch_fns, eval_fn=eval_fn,
                 ckpt_extra={'stage': 'pretrain', 'tokenizer': tokenizer.to_dict()})
    print(f"done, best val loss {best:.4f} -> {os.path.join(cfg.out_dir, cfg.ckpt_name)}")
    return {"best_val_loss": best}


if __name__ == "__main__":
    main()
