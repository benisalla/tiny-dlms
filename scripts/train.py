"""
Stage 1 - pre-train the TinyDLM encoder on plain text (data_dir/pretrain_train.txt, pretrain_val.txt).

The encoder learns the language by guessing hidden words (masked-word prediction, like BERT).
It knows nothing about decisions yet; that is stage 2 (scripts/finetune.py).

    python scripts/prepare_babi.py           # builds data/babi/pretrain_*.txt and data/babi/*.csv
    python scripts/train.py                  # -> checkpoints/pretrain.pt
    python scripts/train.py --n_layer 4 --n_embd 128 --max_iters 3000
    python scripts/train.py --norm_type layernorm --pos_encoding learned --gated_mlp false --activation gelu_tanh
                                             # GPT-2 style blocks instead of Llama style
"""
import os
import random

from tiny_dlm.config import ModelConfig, TrainConfig
from tiny_dlm.data import get_mlm_batch, load_text, make_chunks
from tiny_dlm.model import TinyDLM
from tiny_dlm.runtime import parse_config_args, resolve_device, setup_torch
from tiny_dlm.tokenizer import WordTokenizer
from tiny_dlm.training import build_optimizer, train


def main(argv=None):
    cfg, model_cfg = parse_config_args(TrainConfig, ModelConfig, description=__doc__, argv=argv)
    cfg.ckpt_name = cfg.ckpt_name or "pretrain.pt"
    setup_torch(cfg.seed)
    device = resolve_device(cfg.device)

    # connect to dataset
    train_texts = load_text(os.path.join(cfg.data_dir, "pretrain_train.txt"))
    val_texts = load_text(os.path.join(cfg.data_dir, "pretrain_val.txt"))
    tokenizer = WordTokenizer.from_texts(train_texts)  # the vocabulary is fixed here, fine-tuning reuses it
    train_chunks = make_chunks(train_texts, tokenizer, model_cfg.block_size)
    val_chunks = make_chunks(val_texts, tokenizer, model_cfg.block_size)

    model_cfg.vocab_size = len(tokenizer)
    model_cfg.pad_id, model_cfg.sep_id = tokenizer.pad_id, tokenizer.sep_id
    model = TinyDLM(model_cfg).to(device)
    print(f"number of parameters: {model.get_num_params() / 1e3:.1f}K | vocab: {len(tokenizer)} | "
          f"texts: {len(train_texts)} -> {len(train_chunks)} chunks | device: {device}")

    optimizer = build_optimizer(model, cfg, device)
    rng = random.Random(cfg.seed)
    get_batch_fns = {
        "train": lambda: get_mlm_batch(train_chunks, cfg.batch_size, tokenizer, device, cfg.mask_prob, rng),
        "val": lambda: get_mlm_batch(val_chunks, cfg.batch_size, tokenizer, device, cfg.mask_prob, rng),
    }
    best = train(model, optimizer, cfg, get_batch_fns,
                 ckpt_extra={'stage': 'pretrain', 'tokenizer': tokenizer.to_dict()})
    print(f"done, best val loss {best:.4f} -> {os.path.join(cfg.out_dir, cfg.ckpt_name)}")
    return {"best_val_loss": best}


if __name__ == "__main__":
    main()
