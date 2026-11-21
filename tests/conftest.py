import csv
import importlib.util
from pathlib import Path

import pytest
import torch

from tiny_dlm import Decision, ModelConfig, TinyDLM, WTokenizer

ROOT = Path(__file__).resolve().parents[1]


def tiny_config(**overrides):
    args = dict(vocab_size=50, block_size=32, n_layer=2, n_cand_layer=1, n_head=4, n_embd=32, drop_rate=0.0)
    args.update(overrides)
    return ModelConfig(**args)


def load_script(name):
    """Import scripts/<name>.py as a module (the scripts folder is not a package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DECISIONS = [
    Decision("it is raining", ["take an umbrella", "eat some food"], ["take an umbrella"]),
    Decision("i am hungry and i am tired", ["go to sleep", "wear a coat", "eat some food", "read a book"],
             ["eat some food", "go to sleep"]),
    Decision("it is cold", ["wear sunglasses", "wear a coat", "run to work"], ["wear a coat"]),
]


@pytest.fixture
def decisions():
    return list(DECISIONS)


@pytest.fixture
def tokenizer():
    texts = [t for d in DECISIONS for t in [d.context, *d.candidates]]
    return WTokenizer.from_texts(texts)


@pytest.fixture
def tiny_model(tokenizer):
    torch.manual_seed(0)
    return TinyDLM(tiny_config(vocab_size=len(tokenizer))).eval()


def write_csv(path, decisions):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["context", "candidates", "answers"])
        writer.writeheader()
        for d in decisions:
            writer.writerow(dict(context=d.context, candidates=" | ".join(d.candidates), answers=" | ".join(d.answers)))


@pytest.fixture
def data_dir(tmp_path):
    """A tiny train/val/test set in the same CSV format as data/babi (no download needed)."""
    write_csv(tmp_path / "train.csv", DECISIONS * 50)
    write_csv(tmp_path / "val.csv", DECISIONS)
    write_csv(tmp_path / "test.csv", DECISIONS)
    texts = [f"{d.context} . {c} ." for d in DECISIONS for c in d.candidates]  # pre-training text
    (tmp_path / "pretrain_train.txt").write_text("\n".join(texts * 10) + "\n")
    (tmp_path / "pretrain_val.txt").write_text("\n".join(texts) + "\n")
    return tmp_path
