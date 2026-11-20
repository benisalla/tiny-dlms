"""
ImageDLM: a decision model that looks at an IMAGE and picks one option written in words.

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

    Tensor          Type            Shape
    ===========================================================================
    images          uint8 / float   (B, 3, H, W)        0..255
    idx             long            (B, K, L)           "[SEP] option" token ids
    cand_mask       bool            (B, K)              True = real option
    targets         float           (B, K)              target probabilities
    ---------------------------------------------------------------------------
    logits          float           (B, K)              -inf on empty slots
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

            (B, 3, 48, 48)  -> unfold 8x8 ->  (B, 36, 3*8*8 = 192)  -> Linear ->  (B, 36, n_embd)  + position

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
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)  # (B, 3*p*p, n_patches)
        return self.proj(patches.transpose(1, 2)) + self.pos  # (B, n_patches, n_embd)


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
            rotary = RotaryEmbedding(config.head_size, config.block_size, config.rope_theta)
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
        B, K, L = idx.size()
        P = self.patches.n_patches
        assert P + L <= self.config.block_size, f"{P} patches + {L} option tokens > block_size {self.config.block_size}"

        # 1. the image, once per decision, shared by its K options
        img = self.patches(images)  # (B, P, n_embd)
        img = img.unsqueeze(1).expand(B, K, P, -1).reshape(B * K, P, -1)

        # 2. [patches] [SEP] option_k -> one sequence per option
        opt_idx = idx.reshape(B * K, L)
        x = torch.cat([img, self.transformer.wte(opt_idx)], dim=1)  # (B*K, P+L, n_embd)
        keep = torch.cat([torch.ones(B * K, P, dtype=torch.bool, device=idx.device),
                          opt_idx != self.config.pad_id], dim=1)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, keep)
        h = self.transformer.ln_f(x)[:, P:]  # the option part only: (B*K, L, n_embd)

        # 3. pool: the mean of the option's own words (everything after [SEP], no padding)
        is_option = (opt_idx != self.config.pad_id) & (opt_idx != self.config.sep_id)
        w = is_option.unsqueeze(-1).to(h.dtype)
        z = ((h * w).sum(1) / w.sum(1).clamp(min=1)).view(B, K, -1)  # (B, K, n_embd)

        # 4. the options look at each other, then one logit each (same as TinyDLM)
        for block in self.candidates.h:
            z = block(z, cand_mask)
        z = self.candidates.ln_f(z)
        logits = self.score_head(z).squeeze(-1).masked_fill(~cand_mask, -float('inf'))

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
        Option texts -> "[SEP] option" token ids, padded: (K, L) long.
        The same K options are used for every frame of the car, so this is built once.
    """
    seqs = [[tokenizer.sep_id] + tokenizer.encode(t) for t in texts]
    L = max(len(s) for s in seqs)
    idx = torch.full((len(texts), L), tokenizer.pad_id, dtype=torch.long)
    for k, s in enumerate(seqs):
        idx[k, :len(s)] = torch.tensor(s)
    return idx.to(device)


def load_image_dlm(path, device="cpu"):
    """Return (model, tokenizer, option_texts, checkpoint) from a checkpoint written by scripts/car_train.py."""
    from tiny_dlm.config import ModelConfig
    from tiny_dlm.tokenizer import WordTokenizer
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = ImageDLM(ModelConfig.from_dict(ckpt["model_args"]), **ckpt["image_args"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), WordTokenizer.from_dict(ckpt["tokenizer"]), ckpt["option_texts"], ckpt
