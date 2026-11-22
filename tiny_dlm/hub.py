"""
Hugging Face Hub format: a model is a folder (or a Hub repo) with

    model.safetensors   the weights
    config.json         {"library", "kind" ("rhoa" or "melon"), "model_args", "image_args", "option_texts", ...}
    tokenizer.json      the tokenizer (byte-level BPE for Rhoa, the option words for Mēlon)
    README.md           the model card

    save_pretrained("checkpoints/rhoa_web2.pt", "hf/tiny-dlms-rhoa")   # a training checkpoint -> that folder
    model, tokenizer, config = from_pretrained("benisalla/tiny-dlms-rhoa")   # a Hub id or a local folder

For Mēlon, config["option_texts"] holds the options the model was trained to pick from (encode them with
tiny_dlm.vision.encode_options).
"""
import json
import os

import torch

from tiny_dlm.config import ModelConfig
from tiny_dlm.model import TinyDLM
from tiny_dlm.tokenizer import tokenizer_from_dict
from tiny_dlm.vision import ImageDLM

LIBRARY = "tiny-dlms"
FILES = ["model.safetensors", "config.json", "tokenizer.json"]


def _build(config):
    model_config = ModelConfig.from_dict(config["model_args"])
    if config["kind"] == "melon":
        return ImageDLM(model_config, **config["image_args"])
    return TinyDLM(model_config)


def save_pretrained(checkpoint_path, out_dir, extra=None):
    """A training checkpoint (.pt) -> out_dir/{model.safetensors, config.json, tokenizer.json}. -> the config dict"""
    from safetensors.torch import save_model
    ckpt = torch.load(os.path.expanduser(checkpoint_path), map_location="cpu", weights_only=True)
    config = {"library": LIBRARY, "kind": "melon" if "image_args" in ckpt else "rhoa",
              "model_args": ckpt["model_args"], "stage": ckpt.get("stage"), "training_steps": ckpt.get("iter_num")}
    for key in ("image_args", "option_texts", "tasks"):
        if key in ckpt:
            config[key] = ckpt[key]
    config.update(extra or {})
    model = _build(config)
    model.load_state_dict(ckpt["model"])
    os.makedirs(out_dir, exist_ok=True)
    # save_model writes shared (tied) tensors once
    save_model(model, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt", "library": LIBRARY})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        json.dump(ckpt["tokenizer"], f, ensure_ascii=False)
    return config


def resolve(name_or_path, revision=None):
    """A local folder as is, else a Hub repo id downloaded to the Hugging Face cache. -> local folder"""
    if os.path.isdir(os.path.expanduser(name_or_path)):
        return os.path.expanduser(name_or_path)
    from huggingface_hub import snapshot_download
    return snapshot_download(name_or_path, revision=revision, allow_patterns=FILES)


def from_pretrained(name_or_path, device="cpu", revision=None):
    """-> (model in eval mode on `device`, tokenizer, config dict)"""
    from safetensors.torch import load_model
    folder = resolve(name_or_path, revision)
    with open(os.path.join(folder, "config.json")) as f:
        config = json.load(f)
    assert config.get("library") == LIBRARY, f"{name_or_path} is a {config.get('library')} model, not {LIBRARY}"
    with open(os.path.join(folder, "tokenizer.json")) as f:
        tokenizer = tokenizer_from_dict(json.load(f))
    model = _build(config)
    load_model(model, os.path.join(folder, "model.safetensors"))
    return model.to(device).eval(), tokenizer, config
