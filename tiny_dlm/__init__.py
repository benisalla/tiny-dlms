"""Tiny DLM: a small Jev-style decision model built from scratch in PyTorch."""
from tiny_dlm.checkpoint import load_checkpoint, save_checkpoint
from tiny_dlm.config import ModelConfig, TrainConfig
from tiny_dlm.data import Decision, collate, get_batch, get_mlm_batch, load_csv, load_text, make_chunks, mask_tokens
from tiny_dlm.model import TinyDLM
from tiny_dlm.tokenizer import BPETokenizer, WTokenizer, build_tokenizer
from tiny_dlm.vision import ImageDLM

# The models' names: Greek fruits.
Rhoa = TinyDLM  # ῥόα, pomegranate: one fruit full of seeds, like a decision full of options. Reads text.
Melon = ImageDLM  # μῆλον, the golden apple of the Judgment of Paris, who looked and picked one. Reads images.

__version__ = "0.1.0"

__all__ = ["ModelConfig", "TrainConfig", "TinyDLM", "ImageDLM", "Rhoa", "Melon", "BPETokenizer", "WTokenizer", "build_tokenizer", "Decision",
           "collate", "get_batch", "get_mlm_batch", "load_csv", "load_text", "make_chunks", "mask_tokens",
           "load_checkpoint", "save_checkpoint"]
