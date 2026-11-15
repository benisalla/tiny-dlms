"""
Stage 2 - fine-tune TinyDLM to decide (data_dir/train.csv, val.csv, and test.csv if present).

Starts from the pre-trained encoder (stage 1) and trains the whole model with the decision loss:
a softmax over the candidates' scores, compared with the correct answer(s).

    python scripts/finetune.py                                         # from checkpoints/pretrain.pt
    python scripts/finetune.py --init_from scratch                     # no pre-training, to compare
    python scripts/finetune.py --init_from checkpoints/pretrain.pt --learning_rate 3e-4
"""
import os
import random

from tiny_dlm.checkpoint import load_checkpoint
from tiny_dlm.config import ModelConfig, TrainConfig
from tiny_dlm.data import all_texts, get_batch, load_csv
from tiny_dlm.model import TinyDLM
from tiny_dlm.runtime import parse_config_args, resolve_device, setup_torch
from tiny_dlm.tokenizer import WordTokenizer
from tiny_dlm.training import accuracy, build_optimizer, train


def main(argv=None):
    # fine-tuning starts from the pre-trained model unless told otherwise
    cfg, model_cfg = parse_config_args(TrainConfig, ModelConfig, description=__doc__, argv=argv,
                                       defaults={"init_from": "./checkpoints/pretrain.pt"})
    cfg.ckpt_name = cfg.ckpt_name or "finetune.pt"
    setup_torch(cfg.seed)
    device = resolve_device(cfg.device)

    # connect to dataset
    train_data = load_csv(os.path.join(cfg.data_dir, "train.csv"))
    val_data = load_csv(os.path.join(cfg.data_dir, "val.csv"))

    if cfg.init_from == "scratch":
        tokenizer = WordTokenizer.from_texts(all_texts(train_data))
        model_cfg.vocab_size, model_cfg.pad_id, model_cfg.sep_id = len(tokenizer), tokenizer.pad_id, tokenizer.sep_id
        model = TinyDLM(model_cfg).to(device)
        print("initialising a new model from scratch (no pre-training)")
    else:
        # the architecture and the vocabulary come from the pre-trained checkpoint
        model, tokenizer, _ = load_checkpoint(cfg.init_from, device)
        print(f"starting from the pre-trained model {cfg.init_from}")
    print(f"number of parameters: {model.get_num_params() / 1e3:.1f}K | vocab: {len(tokenizer)} | device: {device}")

    optimizer = build_optimizer(model, cfg, device)
    rng = random.Random(cfg.seed)
    get_batch_fns = {
        "train": lambda: get_batch(train_data, cfg.batch_size, tokenizer, device, rng),
        "val": lambda: get_batch(val_data, cfg.batch_size, tokenizer, device, rng),
    }
    best = train(model, optimizer, cfg, get_batch_fns,
                 ckpt_extra={'stage': 'finetune', 'tokenizer': tokenizer.to_dict()})

    # note: these use the model at the end of training; the checkpoint holds the best-val-loss one
    result = {"best_val_loss": best,
              "train_acc": accuracy(model, train_data, tokenizer, device),
              "val_acc": accuracy(model, val_data, tokenizer, device)}
    msg = (f"done, best val loss {best:.4f} | train accuracy {result['train_acc']:.1%} | "
           f"val accuracy {result['val_acc']:.1%}")
    test_path = os.path.join(cfg.data_dir, "test.csv")  # held out until the very end
    if os.path.exists(test_path):
        result["test_acc"] = accuracy(model, load_csv(test_path), tokenizer, device)
        msg += f" | test accuracy {result['test_acc']:.1%}"
    print(msg)
    return result


if __name__ == "__main__":
    main()
