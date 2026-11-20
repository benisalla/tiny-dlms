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
    Rhoa (ῥόα, pomegranate): the text decision model. A Jev-style decision model.

    An LLM *generates* the answer token by token. A decision model gets a list of candidate answers
    and *scores all of them in one forward pass*, then a softmax over the candidates gives one
    probability per candidate. It can only ever pick one of the given candidates.

        for each option j:
            [CLS] context [SEP] option_j  ->  wte  ->  EBlock x n_layer  ->  ln_f  ->  pool = z_j
        all z_j of one decision           ->  EBlock x n_cand_layer (no positions)  ->  ln_f
                                          ->  score_head  ->  logit_j  ->  softmax over the k options

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

            b decisions in the batch, k options per decision, s tokens per sequence, e embedding size

        Tensor          Type            Shape
        ===========================================================================
        idx             long            (b, k, s)       one "[CLS] context [SEP] option" per option
        cand_mask       bool            (b, k)          True = real option, False = empty slot
        targets         float           (b, k)          target probabilities
        ---------------------------------------------------------------------------
        logits          float           (b, k)          -inf on empty slots
        ===========================================================================
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size > 0, "set config.vocab_size from the tokenizer"
        assert config.block_size is not None
        self.config = config

        rotary = None
        if config.pos_encoding == "rope":  # one cos/sin table shared by every layer
            rotary = RotaryEmbedding(config.head_size, config.block_size, config.rope_base)
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
        idx:     (b, s) token ids            b sequences, s tokens each
        returns: (b, s, e) one vector of size e per token
        """
        b, s = idx.size()
        assert s <= self.config.block_size, f"Max sequence length is {self.config.block_size}, got {s}"
        keep = idx != self.config.pad_id  # (b, s): padding tokens are never attended to
        X = self.transformer.wte(idx)  # (b, s, e)
        if "wpe" in self.transformer:
            m = torch.arange(s, device=idx.device)  # token positions 0 .. s-1
            X = X + self.transformer.wpe(m)  # (s, e) broadcast over the batch

        X = self.transformer.drop(X)
        for block in self.transformer.h:
            X = block(X, keep)  # (b, s, e)
        return self.transformer.ln_f(X)

    def pool(self, idx, H):
        """
        One vector per (context, option) sequence.
        idx: (b, s) token ids, H: (b, s, e) encoder output -> (b, e)

        "cls":       the vector at [CLS] (position 0), like BERT.
        "candidate": the mean of the vectors of the candidate's own tokens (everything after [SEP]).
                     Why: at the start of training [CLS] looks at all ~70 tokens equally, so the one word
                     that differs between candidates is only ~1/70 of its input and training gets stuck
                     at random guessing. A candidate token starts from its own word and only has to
                     collect the evidence about it from the context.
        """
        if self.config.pool == "cls":
            return H[:, 0]  # (b, e): the vector at position 0
        after_sep = (idx == self.config.sep_id).long().cumsum(dim=-1) > 0  # (b, s): True from [SEP] on
        is_option = after_sep & (idx != self.config.sep_id) & (idx != self.config.pad_id)  # (b, s)
        w = is_option.unsqueeze(-1).to(H.dtype)  # (b, s, 1): 1 on the option's own tokens
        return (H * w).sum(dim=1) / w.sum(dim=1).clamp(min=1)  # (b, e) mean; empty slots -> zeros

    def forward(self, idx, cand_mask=None, targets=None):
        """
        Two stages, one model:

        pre-training (cand_mask is None): masked-word prediction, like BERT
            idx:     (b, s) token ids where some words were replaced by [MASK]
            targets: (b, s) the original id at masked positions, -1 elsewhere (not part of the loss)
            returns: logits (b, s, vocab_size) and the loss

        fine-tuning / deciding (cand_mask given):
            idx:       (b, k, s) token ids, one "[CLS] context [SEP] option" sequence per option
            cand_mask: (b, k) True for real options, False for empty slots (decisions have different k)
            targets:   (b, k) target probabilities (sum to 1 over the real options), optional
            returns:   logits (b, k) and the loss (None when targets are not given)
        """
        if cand_mask is None:
            logits = self.mlm_head(self.encode(idx))  # (b, s, vocab_size)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss

        b, k, s = idx.size()  # b decisions, k options each, s tokens per sequence

        # 1. read every (context, option) pair as its own sequence -> one vector per option
        pairs = idx.view(b * k, s)  # all options of all decisions in one big batch
        Z = self.pool(pairs, self.encode(pairs)).view(b, k, -1)  # (b, k, e)

        # 2. the options of one decision look at each other (empty slots are hidden)
        for block in self.candidates.h:
            Z = block(Z, cand_mask)  # (b, k, e)
        Z = self.candidates.ln_f(Z)

        # 3. one logit per option; empty slots can never be chosen
        logits = self.score_head(Z).squeeze(-1)  # (b, k)
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
        returns: choice (b,) the chosen option of each decision, probs (b, k)
        """
        logits, _ = self(idx, cand_mask)
        if temperature == 0:
            probs = F.softmax(logits, dim=-1)
            return probs.argmax(dim=-1), probs
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1), probs
