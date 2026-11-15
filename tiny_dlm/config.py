from dataclasses import asdict, dataclass, fields


@dataclass
class ModelConfig:
    """
    Architecture of the model (everything needed to rebuild it from a checkpoint).

    The defaults describe a small Llama-style encoder: RMSNorm, RoPE, SwiGLU, no biases.
    The switches in the second block let you go back to the GPT-2 choices
    (LayerNorm, learned positions, GELU MLP, biases) and compare.
    """

    # ---- size ----
    block_size: int = 128  # longest "[CLS] context [SEP] candidate" sequence, in tokens
    vocab_size: int = 0  # set from the tokenizer before building the model
    n_layer: int = 2  # blocks that read each (context, candidate) pair
    n_cand_layer: int = 1  # blocks that let the candidates of one decision look at each other
    n_head: int = 4
    n_embd: int = 64
    drop_rate: float = 0.1
    bias: bool = False  # biases in Linear layers and LayerNorms
    pad_id: int = 0  # padding token id, ignored by attention
    sep_id: int = 3  # [SEP] token id, the candidate starts right after it
    pool: str = "candidate"  # one vector per candidate: "candidate" = mean of the candidate's own tokens, "cls" = [CLS]

    # ---- design choices (defaults = Llama-style) ----
    n_kv_head: int = 0  # 0 -> same as n_head. Fewer KV heads = grouped-query attention (Llama, Qwen2)
    head_dim: int = 0  # 0 -> n_embd // n_head
    norm_type: str = "rmsnorm"  # "rmsnorm" (Llama, Qwen2) or "layernorm" (GPT-2)
    norm_eps: float = 1e-6
    pos_encoding: str = "rope"  # "rope" rotary (Llama, Qwen2) or "learned" absolute positions (GPT-2)
    rope_theta: float = 10000.0  # RoPE base frequency
    activation: str = "silu"  # "silu" (Llama, inside SwiGLU), "gelu_tanh" (GPT-2), "gelu", "relu"
    gated_mlp: bool = True  # SwiGLU MLP: down(act(gate(x)) * up(x)) (Llama, Qwen2)
    intermediate_size: int = 0  # MLP hidden size, 0 -> 4 * n_embd

    def __post_init__(self):
        assert self.head_dim or self.n_embd % self.n_head == 0, "n_embd % n_head should be 0."
        assert self.n_head % self.kv_heads == 0, "n_head must be a multiple of n_kv_head"
        assert self.norm_type in ("layernorm", "rmsnorm"), self.norm_type
        assert self.pos_encoding in ("learned", "rope"), self.pos_encoding
        assert self.pool in ("candidate", "cls"), self.pool
        assert self.activation in ("gelu_tanh", "gelu", "silu", "relu"), self.activation

    # derived sizes
    @property
    def kv_heads(self):
        return self.n_kv_head or self.n_head

    @property
    def head_size(self):
        return self.head_dim or self.n_embd // self.n_head

    @property
    def hidden_size(self):
        return self.intermediate_size or 4 * self.n_embd

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Build a config from a dict, ignoring unknown keys (e.g. old checkpoints)."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class TrainConfig:
    """Everything about the optimisation loop."""

    # "scratch" or the path of a checkpoint to start from (fine-tuning starts from the pre-trained model)
    init_from: str = "scratch"

    # data
    data_dir: str = "./data/babi"  # pretrain_{train,val}.txt for pre-training, {train,val,test}.csv for fine-tuning
    mask_prob: float = 0.15  # pre-training: fraction of words hidden for the model to guess
    batch_size: int = 32  # decisions per micro-batch
    grad_accum_steps: int = 1  # micro-batches per optimizer step

    # optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # lr schedule (linear warmup + cosine decay)
    decay_lr: bool = True
    warmup_iters: int = 50
    lr_decay_iters: int = 600
    min_lr: float = 1e-4
    max_iters: int = 600

    # evaluation / logging / checkpointing
    eval_interval: int = 100
    eval_iters: int = 10
    log_interval: int = 50
    always_save_checkpoint: bool = False  # if False, only save when val loss improves
    out_dir: str = "./checkpoints"
    ckpt_name: str = ""  # checkpoints are written to out_dir/ckpt_name ("" -> pretrain.pt / finetune.pt)

    # system
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    seed: int = 2000
