import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from tiny_dlm.block import EBlock
from tiny_dlm.norm import build_norm
from tiny_dlm.rope import RotaryEmbedding


class TinyDLM(nn.Module):
    """
    Tiny DLM: a Jev-style decision model.

    An LLM *generates* the answer token by token. A decision model gets a list of candidate answers
    and *scores all of them in one forward pass*, then a softmax over the candidates gives one
    probability per candidate. It can only ever pick one of the given candidates.

        for each candidate i:
            [CLS] context [SEP] candidate_i  ->  wte  ->  EBlock x n_layer  ->  ln_f  ->  pool = z_i
        all z_i of one decision              ->  EBlock x n_cand_layer (no positions)  ->  ln_f
                                             ->  score_head  ->  logit_i  ->  softmax over candidates

    pool: the mean of the candidate's own tokens (default) or the vector at [CLS] (see `pool`).
    The first part reads each (context, candidate) pair (the NanoJev layout). The second part lets
    the candidates of the same decision look at each other before they are scored.

    Training has two stages (see scripts/train.py and scripts/finetune.py):
        1. pre-training: the encoder learns the language by guessing hidden words ([MASK]), like BERT.
           Bidirectional encoder -> masked-word prediction, not next-word prediction like an LLM.
        2. fine-tuning: the same encoder + the candidate layers + score_head learn to decide.

    Args:
        config: ModelConfig (vocab_size, block_size, n_layer, n_cand_layer, n_head, n_embd, ...)

    Attributes:
        transformer: ModuleDict that reads each pair (Embeddings, Dropout, Blocks, Norm).
        candidates: ModuleDict that mixes the candidates of one decision (Blocks, Norm).
        score_head: Linear layer, one vector -> one logit (fine-tuning).
        mlm_head: Linear layer, one vector -> a score for every word of the vocabulary (pre-training).
        config: Model configuration.

        Tensor          Type            Shape
        ===========================================================================
        idx             long            (batch_size, n_candidates, seq_len)
        cand_mask       bool            (batch_size, n_candidates)   True = real candidate
        targets         float           (batch_size, n_candidates)   target probabilities
        ---------------------------------------------------------------------------
        logits          float           (batch_size, n_candidates)   -inf on empty slots
        ===========================================================================
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size > 0, "set config.vocab_size from the tokenizer"
        assert config.block_size is not None
        self.config = config

        rotary = None
        if config.pos_encoding == "rope":  # one cos/sin table shared by every layer
            rotary = RotaryEmbedding(config.head_size, config.block_size, config.rope_theta)
        layers = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
            drop=nn.Dropout(config.drop_rate),
            h=nn.ModuleList([EBlock(config, rotary) for _ in range(config.n_layer)]),  # Block X N
            ln_f=build_norm(config),
        )
        if config.pos_encoding == "learned":
            layers["wpe"] = nn.Embedding(config.block_size, config.n_embd)  # positional embeddings
        self.transformer = nn.ModuleDict(layers)

        # no positions here: shuffling the candidates must only shuffle their scores
        self.candidates = nn.ModuleDict(dict(
            h=nn.ModuleList([EBlock(config, use_positions=False) for _ in range(config.n_cand_layer)]),
            ln_f=build_norm(config),
        ))
        # no bias: adding the same number to every option's score does not change the softmax, it could never learn
        self.score_head = nn.Linear(config.n_embd, 1, bias=False)

        # pre-training head: predict the hidden (masked) words, like BERT
        self.mlm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.transformer.wte.weight = self.mlm_head.weight  # weight tying (as tiny-llm's lm_head)
        # scaled init of residual projections (GPT-2 paper)
        n_residual_layers = config.n_layer + config.n_cand_layer
        for param_name, param in self.named_parameters():
            if param_name.endswith('c_proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_residual_layers))

    def get_num_params(self, non_embedding=True):
        """Number of params. Position embeddings are excluded by default."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and "wpe" in self.transformer:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, idx):
        """
        The shared Transformer encoder, used by both stages.
        idx:     (N, T) token ids
        returns: (N, T, n_embd) one vector per token
        """
        N, T = idx.size()
        assert T <= self.config.block_size, f"Max sequence length is {self.config.block_size}, got {T}"
        keep = idx != self.config.pad_id  # padding tokens are never attended to
        x = self.transformer.wte(idx)  # (N, T, n_embd)
        if "wpe" in self.transformer:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = x + self.transformer.wpe(pos)  # (T, n_embd) broadcast over the batch

        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, keep)
        return self.transformer.ln_f(x)

    def pool(self, idx, h):
        """
        One vector per (context, candidate) sequence.
        idx: (N, T) token ids, h: (N, T, n_embd) encoder output -> (N, n_embd)

        "cls":       the vector at [CLS] (position 0), like BERT.
        "candidate": the mean of the vectors of the candidate's own tokens (everything after [SEP]).
                     Why: at the start of training [CLS] looks at all ~70 tokens equally, so the one word
                     that differs between candidates is only ~1/70 of its input and training gets stuck
                     at random guessing. A candidate token starts from its own word and only has to
                     collect the evidence about it from the context.
        """
        if self.config.pool == "cls":
            return h[:, 0]
        after_sep = (idx == self.config.sep_id).cumsum(dim=-1) > 0
        is_cand = after_sep & (idx != self.config.sep_id) & (idx != self.config.pad_id)  # (N, T)
        w = is_cand.unsqueeze(-1).to(h.dtype)  # (N, T, 1)
        return (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1)  # empty slots (no candidate) -> zeros

    def forward(self, idx, cand_mask=None, targets=None):
        """
        Two stages, one model:

        pre-training (cand_mask is None): masked-word prediction, like BERT
            idx:     (B, T) token ids where some words were replaced by [MASK]
            targets: (B, T) the original id at masked positions, -1 elsewhere (not part of the loss)
            returns: logits (B, T, vocab_size) and the loss

        fine-tuning / deciding (cand_mask given):
            idx:       (B, K, T) token ids, one "[CLS] context [SEP] candidate" sequence per candidate
            cand_mask: (B, K) True for real candidates, False for empty slots (decisions have different K)
            targets:   (B, K) target probabilities (sum to 1 over the real candidates), optional
            returns:   logits (B, K) and the loss (None when targets are not given)
        """
        if cand_mask is None:
            logits = self.mlm_head(self.encode(idx))  # (B, T, vocab_size)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss

        B, K, T = idx.size()

        # 1. read every (context, candidate) pair as its own sequence -> one vector per candidate
        z = self.pool(idx.view(B * K, T), self.encode(idx.view(B * K, T))).view(B, K, -1)  # (B, K, n_embd)

        # 2. the candidates of one decision look at each other (empty slots are hidden)
        for block in self.candidates.h:
            z = block(z, cand_mask)
        z = self.candidates.ln_f(z)

        # 3. one logit per candidate; empty slots can never be chosen
        logits = self.score_head(z).squeeze(-1)  # (B, K)
        logits = logits.masked_fill(~cand_mask, -float('Inf'))

        loss = None
        if targets is not None:
            # cross-entropy with soft targets: -sum_i target_i * log p_i.
            # empty slots have log p = -inf and target 0, and 0 * -inf = NaN, so they are zeroed first.
            log_probs = F.log_softmax(logits, dim=-1).masked_fill(~cand_mask, 0.0)
            loss = -(targets * log_probs).sum(dim=-1).mean()

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # 2D params (matmuls, embeddings) are decayed; biases and norms are not
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    @torch.no_grad()
    def decide(self, idx, cand_mask, temperature=0.0):
        """
        Pick one candidate per decision (the decision-model version of `generate`).
        temperature = 0 -> take the most likely candidate; > 0 -> sample (useful to explore).
        returns: choice (B,) candidate indices, probs (B, K)
        """
        logits, _ = self(idx, cand_mask)
        if temperature == 0:
            probs = F.softmax(logits, dim=-1)
            return probs.argmax(dim=-1), probs
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1), probs
