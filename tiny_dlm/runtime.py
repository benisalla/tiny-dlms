"""Small helpers shared by the scripts: command-line config, seeding, device."""
import argparse
import random
from dataclasses import fields

import torch


def _parse_bool(s):
    return s.lower() in ("1", "true", "yes", "y")


def parse_config_args(train_cls, model_cls, description=None, argv=None, defaults=None):
    """
    Every field of TrainConfig and ModelConfig becomes a --flag (e.g. --n_layer 4 --max_iters 300).
    `defaults` lets a script change a default value (e.g. fine-tuning starts from the pre-trained model).
    """
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    for cls in (train_cls, model_cls):
        for f in fields(cls):
            kind = type(f.default)
            parser.add_argument(f"--{f.name}", type=_parse_bool if kind is bool else kind,
                                default=defaults.get(f.name, f.default))
    args = vars(parser.parse_args(argv))
    pick = lambda cls: cls(**{f.name: args[f.name] for f in fields(cls)})  # noqa: E731
    return pick(train_cls), pick(model_cls)


def setup_torch(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
