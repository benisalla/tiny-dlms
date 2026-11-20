"""
Mēlon (μῆλον, the golden apple): ImageDLM, a decision model that looks at an IMAGE and picks one option
written in words.

It is TinyDLM with the context made of image patches instead of words:

    text  (TinyDLM):   [CLS] mary went to the garden . where is mary ? [SEP] garden
    image (ImageDLM):  [p1 p2 ... p36]                                 [SEP] steer slight left and accelerate
                        48x48 frame cut into 36 patches of 8x8

Everything after the context is the same as TinyDLM: the same EBlock, the option's own tokens are averaged,
the candidate layer lets the options of one decision look at each other, `score_head` gives one logit per
option and a softmax over the options gives one probability each.

    for each option k:
        patches(image) + pos  ⊕  wte([SEP] option_k)  ->  EBlock x n_layer  ->  ln_f  ->  mean of option tokens = z_k
    all z_k                   ->  EBlock x n_cand_layer (no positions)       ->  ln_f  ->  score_head  ->  logit_k

The image is cut into patches ONCE per decision, then shared by its K options (expand, no copy).

        b images in the batch, k options per image, p patches per image, s tokens per "[SEP] option",
        e embedding size

    Tensor          Type            Shape
    ===========================================================================
    images          uint8           (b, 3, height, width)   pixels 0..255
    idx             long            (b, k, s)               "[SEP] option" token ids
    cand_mask       bool            (b, k)                  True = real option
    targets         float           (b, k)                  target probabilities
    ---------------------------------------------------------------------------
    logits          float           (b, k)                  -inf on empty slots
    ===========================================================================
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from tiny_dlm.block import EBlock
from tiny_dlm.model import TinyDLM
from tiny_dlm.norm import build_norm
from tiny_dlm.rope import RotaryEmbedding


class PatchEmbedding(nn.Module):
    """
        Cut the image into non-overlapping patches and project each one to a vector (ViT, Dosovitskiy et al. 2020).

            (b, 3, 48, 48)  -> unfold 8x8 ->  (b, p = 36, 3*8*8 = 192)  -> Linear ->  (b, p, e)  + position

        Written with `unfold` + `Linear` so every step is visible (same as a Conv2d with stride = kernel = patch).
        The position embedding is learned: one vector per patch location.
    """

    def __init__(self, image_size, patch_size, channels, n_embd):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be a multiple of patch_size"
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Linear(channels * patch_size * patch_size, n_embd)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, n_embd))

    def forward(self, images):
        x = images.float() / 127.5 - 1.0  # 0..255 -> -1..1
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)  # (b, 3*8*8, p)
        return self.proj(patches.transpose(1, 2)) + self.pos  # (b, p, e)


class ImageDLM(nn.Module):
    """
    Args:
        config: ModelConfig (vocab_size for the option words, n_layer, n_cand_layer, n_head, n_embd, ...)
        image_size, patch_size, channels: the image (48x48 colour, 8x8 patches -> 36 patches by default)

    block_size must hold n_patches + the longest "[SEP] option" (36 + 8 for the car).
    """

    def __init__(self, config, image_size=48, patch_size=8, channels=3):
        super().__init__()
        assert config.vocab_size > 0, "set config.vocab_size from the tokenizer"
        self.config = config
        self.image_args = dict(image_size=image_size, patch_size=patch_size, channels=channels)

        self.patches = PatchEmbedding(image_size, patch_size, channels, config.n_embd)
        rotary = None
        if config.pos_encoding == "rope":
            rotary = RotaryEmbedding(config.head_size, config.block_size, config.rope_base)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # option words
            drop=nn.Dropout(config.drop_rate),
            h=nn.ModuleList([EBlock(config, rotary) for _ in range(config.n_layer)]),
            ln_f=build_norm(config),
        ))
        self.candidates = nn.ModuleDict(dict(
            h=nn.ModuleList([EBlock(config, use_positions=False) for _ in range(config.n_cand_layer)]),
            ln_f=build_norm(config),
        ))
        # no bias: adding the same number to every option's score does not change the softmax, it could never learn
        self.score_head = nn.Linear(config.n_embd, 1, bias=False)

        self.apply(self._init_weights)
        n_residual_layers = config.n_layer + config.n_cand_layer
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_residual_layers))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, images, idx, cand_mask, targets=None):
        b, k, s = idx.size()  # b images, k options each, s tokens per "[SEP] option"
        p = self.patches.n_patches
        assert p + s <= self.config.block_size, f"{p} patches + {s} option tokens > block_size {self.config.block_size}"

        # 1. the image, cut into patches once, then shared by its k options (expand: no copy)
        patches = self.patches(images)  # (b, p, e)
        patches = patches.unsqueeze(1).expand(b, k, p, -1).reshape(b * k, p, -1)  # (b*k, p, e)

        # 2. one sequence per option: [patch_1 .. patch_p] [SEP] option words
        options = idx.reshape(b * k, s)  # (b*k, s)
        X = torch.cat([patches, self.transformer.wte(options)], dim=1)  # (b*k, p+s, e)
        keep = torch.cat([torch.ones(b * k, p, dtype=torch.bool, device=idx.device),
                          options != self.config.pad_id], dim=1)  # (b*k, p+s): padding is never attended to
        X = self.transformer.drop(X)
        for block in self.transformer.h:
            X = block(X, keep)  # (b*k, p+s, e)
        H = self.transformer.ln_f(X)[:, p:]  # (b*k, s, e): keep the option part only

        # 3. pool: the mean of the option's own words (everything after [SEP], no padding)
        is_option = (options != self.config.pad_id) & (options != self.config.sep_id)  # (b*k, s)
        w = is_option.unsqueeze(-1).to(H.dtype)  # (b*k, s, 1)
        Z = ((H * w).sum(1) / w.sum(1).clamp(min=1)).view(b, k, -1)  # (b, k, e)

        # 4. the options look at each other, then one logit each (same as TinyDLM)
        for block in self.candidates.h:
            Z = block(Z, cand_mask)  # (b, k, e)
        Z = self.candidates.ln_f(Z)
        logits = self.score_head(Z).squeeze(-1).masked_fill(~cand_mask, -float('inf'))  # (b, k)

        loss = None
        if targets is not None:
            log_probs = F.log_softmax(logits, dim=-1).masked_fill(~cand_mask, 0.0)
            loss = -(targets * log_probs).sum(dim=-1).mean()
        return logits, loss

    @torch.no_grad()
    def decide(self, images, idx, cand_mask, temperature=0.0):
        """Pick one option per image. temperature 0 -> the most likely; > 0 -> sample."""
        logits, _ = self(images, idx, cand_mask)
        if temperature == 0:
            probs = F.softmax(logits, dim=-1)
            return probs.argmax(dim=-1), probs
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1), probs

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """AdamW, weight decay on matrices only (the same rule as TinyDLM)."""
        return TinyDLM.configure_optimizers(self, weight_decay, learning_rate, betas, device_type)


def encode_options(texts, tokenizer, device="cpu"):
    """
        Option texts -> "[SEP] option" token ids, padded: (k, s) long (k options, s tokens).
        The same K options are used for every frame of the car, so this is built once.
    """
    seqs = [[tokenizer.sep_id] + tokenizer.encode(t) for t in texts]
    s = max(len(seq) for seq in seqs)
    idx = torch.full((len(texts), s), tokenizer.pad_id, dtype=torch.long)
    for j, seq in enumerate(seqs):
        idx[j, :len(seq)] = torch.tensor(seq)
    return idx.to(device)


def load_image_dlm(path, device="cpu"):
    """Return (model, tokenizer, option_texts, checkpoint) from a checkpoint written by scripts/car_train.py."""
    from tiny_dlm.config import ModelConfig
    from tiny_dlm.tokenizer import WordTokenizer
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = ImageDLM(ModelConfig.from_dict(ckpt["model_args"]), **ckpt["image_args"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), WordTokenizer.from_dict(ckpt["tokenizer"]), ckpt["option_texts"], ckpt
