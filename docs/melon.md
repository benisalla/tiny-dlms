# Mēlon · the model that sees

[← README](../README.md) · [← Rhoa](rhoa.md) · [Data](02-data.md)

Mēlon is the same judge as Rhoa, with a picture instead of a question. It cuts the image into small squares,
reads them with the option's words, and gives every option a score.

```mermaid
flowchart TB
    I["<b>camera frame</b><br/>48 × 48 pixels"]:::input --> P["<b>36 patches</b><br/>8 × 8 pixels each<br/>+ position"]:::step

    O["<b>15 options</b><br/>7 steering amounts<br/>× gas or coast<br/>+ brake"]:::input

    subgraph SEQ["one sequence per option"]
        direction LR
        S1["patches &lt;sep&gt; steer slight left and accelerate"]
        S2["patches &lt;sep&gt; go straight and accelerate"]
        S3["patches &lt;sep&gt; brake"]
    end

    subgraph ENC["encoder · 4 blocks × 128"]
        direction LR
        B["RMSNorm → attention + RoPE<br/>RMSNorm → SwiGLU"] --> PL["mean of the option's tokens"]
    end

    C["<b>candidate layer</b><br/>options look at each other"]:::step
    D["<b>softmax</b><br/>go straight and coast <b>96%</b>"]:::output
    A["<b>car</b><br/>steer · gas · brake"]:::output

    P --> SEQ
    O --> SEQ
    SEQ --> ENC --> C --> D --> A

    classDef input fill:#1f2937,stroke:#60a5fa,color:#f9fafb
    classDef step fill:#1f2937,stroke:#f59e0b,color:#f9fafb
    classDef output fill:#064e3b,stroke:#34d399,color:#f9fafb
    style SEQ fill:#111827,stroke:#374151,color:#d1d5db
    style ENC fill:#111827,stroke:#f59e0b,color:#d1d5db
```

## Two games

```mermaid
flowchart LR
    subgraph CAR["CarRacing"]
        direction TB
        CT["<b>teacher</b><br/>pure-pursuit driver,<br/>wheel shaken while recording"]:::data --> CM["<b>Mēlon 4 × 128</b><br/>36 patches · 15 options"]:::step
        CM --> CR["<b>913</b> on new tracks<br/>= the teacher"]:::output
    end
    subgraph MAZE["Maze"]
        direction TB
        MT["<b>teacher</b><br/>shortest path (BFS),<br/>new mazes every batch"]:::data --> MM["<b>Mēlon 8 × 128</b><br/>64 patches, one per cell · 4 moves"]:::step
        MM --> MR["<b>97%</b> of new mazes solved<br/>on the shortest path"]:::output
    end

    classDef data fill:#1f2937,stroke:#60a5fa,color:#f9fafb
    classDef step fill:#1f2937,stroke:#f59e0b,color:#f9fafb
    classDef output fill:#064e3b,stroke:#34d399,color:#f9fafb
    style CAR fill:#111827,stroke:#374151,color:#d1d5db
    style MAZE fill:#111827,stroke:#374151,color:#d1d5db
```

## Specs

| | Car (`melon`) | Maze (`melon_maze`) |
|---|---|---|
| Weights | 1.34 M (0.21 M for the 2 × 64 version) | 2.39 M |
| Encoder | 4 blocks × 128 | 8 blocks × 128 |
| Image | 48 × 48, 36 patches of 8 × 8 | 48 × 48, 64 patches of 6 × 6 |
| Options | 15 (steering, gas, brake) | 4 (up, down, left, right) |
| Learns from | a teacher driver: ~35,000 frames from 40 tracks | a shortest-path teacher on endless new mazes |
| Training | 4,000 steps | 30,000 steps, lr 3e-4 |
| Result | 913 on 5 new tracks (teacher 913, straight −26) | 97% of 200 new mazes (greedy 74%, 4 layers 50%) |

Code: `tiny_dlm/vision.py` (`ImageDLM`, also `Melon`), `tiny_dlm/car.py`, `tiny_dlm/maze.py`.
