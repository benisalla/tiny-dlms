# Rhoa · the model that reads

[← README](../README.md) · [Mēlon →](melon.md) · [Data](02-data.md)

Rhoa takes a text and its options, and gives every option a score. The highest score is the decision.

```mermaid
flowchart TB
    Q["<b>question</b><br/>i lost my card yesterday,<br/>what should i do?"]:::input
    O["<b>options</b><br/>lost or stolen card<br/>change pin<br/>card arrival<br/>… 77 in all"]:::input

    subgraph PAIRS["one sequence per option"]
        direction LR
        P1["&lt;cls&gt; question &lt;sep&gt; lost or stolen card"]
        P2["&lt;cls&gt; question &lt;sep&gt; change pin"]
        P3["&lt;cls&gt; question &lt;sep&gt; …"]
    end

    subgraph ENC["encoder · 4 blocks × 128"]
        direction LR
        T["BPE tokens → embeddings"] --> B["RMSNorm → attention + RoPE<br/>RMSNorm → SwiGLU"]
        B --> PL["mean of the option's tokens"]
    end

    C["<b>candidate layer</b><br/>options look at each other"]:::step
    S["<b>score head</b><br/>one number per option"]:::step
    D["<b>softmax</b><br/>lost or stolen card <b>99.7%</b><br/>lost or stolen phone 0.2%<br/>card linking 0.1%"]:::output

    Q --> PAIRS
    O --> PAIRS
    PAIRS --> ENC --> C --> S --> D

    classDef input fill:#1f2937,stroke:#60a5fa,color:#f9fafb
    classDef step fill:#1f2937,stroke:#f59e0b,color:#f9fafb
    classDef output fill:#064e3b,stroke:#34d399,color:#f9fafb
    style PAIRS fill:#111827,stroke:#374151,color:#d1d5db
    style ENC fill:#111827,stroke:#f59e0b,color:#d1d5db
```

## How it was trained

```mermaid
flowchart LR
    W["<b>2 GB of web text</b><br/>FineWeb-Edu<br/>479M tokens"]:::data --> PT["<b>pre-training</b><br/>guess the 15% hidden tokens<br/>40,000 steps"]:::step
    PT --> FT["<b>fine-tuning</b><br/>pick the right option<br/>16,000 steps"]:::step
    D4["<b>4 datasets at once</b><br/>bAbI · CLINC150<br/>Banking77 · CommonsenseQA"]:::data --> FT
    FT --> R["<b>Rhoa</b><br/>100% bAbI<br/>72.6% CLINC150<br/>84.4% Banking77"]:::output

    classDef data fill:#1f2937,stroke:#60a5fa,color:#f9fafb
    classDef step fill:#1f2937,stroke:#f59e0b,color:#f9fafb
    classDef output fill:#064e3b,stroke:#34d399,color:#f9fafb
```

## Specs (the best Rhoa, `rhoa_web2`)

| | |
|---|---|
| Weights | 2.36 M |
| Encoder | 4 blocks, 128 wide, 4 attention heads |
| Block | RMSNorm, self-attention with rotary positions (RoPE), SwiGLU |
| Candidate layer | 1 block over the options, no positions |
| Tokenizer | byte-level BPE, 8,192 tokens, trained on the web text (`tiny_dlm/tokenizer.py`) |
| Input | up to 128 tokens per `<cls> context <sep> option` |
| Options | any number: 2 for bAbI, 151 for CLINC150 |
| Pre-training | masked tokens, 128 windows of 127 tokens per step, 40,000 steps |
| Fine-tuning | 4 datasets mixed, at most 16 options per question in training, all options at test |
| Code | `tiny_dlm/model.py` (`TinyDLM`, also `Rhoa`) |

Also trained, for the benchmark: a 2 × 64 version, a version with GPT-2 blocks (LayerNorm, learned positions, GELU),
word-tokenizer versions, and a twin of each without pre-training. Results are in the [README](../README.md#results).
