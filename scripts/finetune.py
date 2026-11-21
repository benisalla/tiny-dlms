"""
Stage 2 - fine-tune TinyDLM to decide (train.csv, val.csv, and test.csv if present).

Starts from the pre-trained encoder (stage 1) and trains the whole model with the decision loss:
a softmax over the candidates' scores, compared with the correct answer(s).

    python scripts/finetune.py                                         # bAbI, from checkpoints/pretrain.pt
    python scripts/finetune.py --init_from scratch                     # no pre-training, to compare
    python scripts/finetune.py --init_from checkpoints/pretrain.pt --learning_rate 3e-4

One model, every dataset (scripts/prepare_data.py builds them, data/pretrain is the matching pre-training):

    python scripts/train.py --data_dir data/pretrain
    python scripts/finetune.py --tasks babi,clinc150,banking77,csqa

Each batch comes from one dataset (picked in proportion to its size). Training keeps at most
--max_train_candidates options per decision (the right one + random wrong ones); accuracy always uses all of them.
"""
import os
import random

from tiny_dlm.checkpoint import load_checkpoint
from tiny_dlm.config import ModelConfig, TrainConfig
from tiny_dlm.data import all_texts, get_multitask_batch, load_csv
from tiny_dlm.model import TinyDLM
from tiny_dlm.runtime import parse_config_args, resolve_device, setup_torch
from tiny_dlm.tokenizer import build_tokenizer
from tiny_dlm.training import accuracy, build_optimizer, train


def load_tasks(cfg):
    """{task name: {split: [Decision, ...]}}. One task (data_dir) unless --tasks lists several."""
    dirs = {os.path.basename(os.path.normpath(cfg.data_dir)): cfg.data_dir}
    if cfg.tasks:
        dirs = {name: os.path.join(cfg.data_root, name) for name in cfg.tasks.split(",")}
    return {name: {split: load_csv(os.path.join(folder, f"{split}.csv")) for split in ("train", "val", "test")
                   if os.path.exists(os.path.join(folder, f"{split}.csv"))}
            for name, folder in dirs.items()}


def main(argv=None):
    # fine-tuning starts from the pre-trained model unless told otherwise
    cfg, model_cfg = parse_config_args(TrainConfig, ModelConfig, description=__doc__, argv=argv,
                                       defaults={"init_from": "./checkpoints/pretrain.pt"})
    cfg.ckpt_name = cfg.ckpt_name or "finetune.pt"
    setup_torch(cfg.seed)
    device = resolve_device(cfg.device)

    # connect to dataset(s)
    tasks = load_tasks(cfg)
    for name, splits in tasks.items():
        k = max(len(d.candidates) for d in splits["train"])
        print(f"{name}: " + " | ".join(f"{s} {len(v):,}" for s, v in splits.items()) + f" | up to {k} options")

    if cfg.init_from == "scratch":
        texts = [t for splits in tasks.values() for t in all_texts(splits["train"])]
        tokenizer = build_tokenizer(cfg.tokenizer, texts, cfg.bpe_vocab_size, min_freq=1)  # learned from the training set
        model_cfg.vocab_size, model_cfg.pad_id, model_cfg.sep_id = len(tokenizer), tokenizer.pad_id, tokenizer.sep_id
        model = TinyDLM(model_cfg).to(device)
        print("initialising a new model from scratch (no pre-training)")
    else:
        # the architecture and the vocabulary come from the pre-trained checkpoint
        model, tokenizer, _ = load_checkpoint(cfg.init_from, device)
        print(f"starting from the pre-trained model {cfg.init_from}")
    print(f"Rhoa | number of parameters: {model.get_num_params() / 1e3:.1f}K | {type(tokenizer).__name__} vocab: {len(tokenizer)} | device: {device}")

    optimizer = build_optimizer(model, cfg, device)
    rng = random.Random(cfg.seed)
    get_batch_fns = {}
    for split in ("train", "val"):
        data = {name: splits[split] for name, splits in tasks.items()}
        get_batch_fns[split] = (lambda data=data: get_multitask_batch(data, cfg.batch_size, tokenizer, device, rng,
                                                                      cfg.max_train_candidates))
    # validation accuracy per dataset during training, on a fixed sample of 300 decisions, with ALL the options
    val_sample = {name: random.Random(0).sample(splits["val"], min(300, len(splits["val"]))) for name, splits in tasks.items()}
    eval_fn = lambda m: {f"val_acc/{n}": accuracy(m, d, tokenizer, device) for n, d in val_sample.items()}  # noqa: E731
    best = train(model, optimizer, cfg, get_batch_fns, eval_fn=eval_fn,
                 ckpt_extra={'stage': 'finetune', 'tokenizer': tokenizer.to_dict(), 'tasks': list(tasks)})

    # note: these use the model at the end of training; the checkpoint holds the best-val-loss one.
    # Accuracy uses ALL the options of each decision (151 for CLINC150), not the sampled training subset.
    result = {"best_val_loss": best}
    for name, splits in tasks.items():
        # train accuracy on a fixed sample of 1,000 (all of CLINC150's train set is 15,100 x 151 = 2.3M sequences)
        sample = {**splits, "train": random.Random(0).sample(splits["train"], min(1000, len(splits["train"])))}
        accs = {split: accuracy(model, data, tokenizer, device) for split, data in sample.items()}
        result[name] = accs
        print(f"{name:10s} " + " | ".join(f"{split} accuracy {a:.1%}" for split, a in accs.items()))
    print(f"done, best val loss {best:.4f}")
    if len(tasks) == 1:  # single dataset: also the flat keys (train_acc, val_acc, test_acc)
        result.update({f"{split}_acc": a for split, a in accs.items()})
    return result


if __name__ == "__main__":
    main()
