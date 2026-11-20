from torch import nn

ACTIVATIONS = {
    "gelu_tanh": lambda: nn.GELU(approximate="tanh"),  # GPT-2 ("gelu_new")
    "gelu": nn.GELU,
    "silu": nn.SiLU,  # Llama, Qwen2 (inside SwiGLU)
    "relu": nn.ReLU,
}


class MLP(nn.Module):
    """
        Multi-Layer Perceptron (MLP): computation layer.
        Non-linearity = complex mappings + feature extraction + ... + processing data (reasoning)

            classic (GPT-2):           c_proj( act(c_fc(X)) )
            gated   (Llama, Qwen2):    c_proj( act(c_gate(X)) * c_fc(X) )      a.k.a. SwiGLU with act = SiLU

        Tensor          Type            Shape
        ===========================================================================
        input           float           (b, s, e)      b sequences, s tokens, e embedding size
        ---------------------------------------------------------------------------
        output          float           (b, s, e)      b sequences, s tokens, e embedding size
        ===========================================================================
    """

    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=config.bias)  # "up" projection
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=config.bias) if config.gated_mlp else None
        self.act = ACTIVATIONS[config.activation]()
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)  # "down" projection
        self.dropout = nn.Dropout(config.drop_rate)

    def forward(self, X):
        if self.c_gate is not None:
            hidden = self.act(self.c_gate(X)) * self.c_fc(X)  # (b, s, hidden_size): 4e by default
        else:
            hidden = self.act(self.c_fc(X))  # (b, s, hidden_size): 4e by default
        return self.dropout(self.c_proj(hidden))  # (b, s, e)
