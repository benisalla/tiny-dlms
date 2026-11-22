"""
Publish the main models to the Hugging Face Hub, one repo per model, with a model card, and group them in a
collection. The other checkpoints (ablations, twins) stay in the GitHub release (scripts/publish_models.sh).

    hf auth login                                  # once (a token with write access)
    python scripts/publish_hf.py --public          # every model below -> benisalla/tiny-dlms-<name>
    python scripts/publish_hf.py --names rhoa --dry_run   # only write the folder in hf/, upload nothing

Each repo gets model.safetensors, config.json, tokenizer.json and README.md (the card: what it does, usage,
model, data, results from docs/assets/results.json). Load one with tiny_dlm.hub.from_pretrained.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from tiny_dlm.hub import save_pretrained  # noqa: E402

GITHUB = "https://github.com/benisalla/tiny-dlms"
SITE = "https://benisalla.github.io/tiny-dlms/"
TASKS = [("babi", "bAbI"), ("clinc150", "CLINC150"), ("banking77", "Banking77"), ("csqa", "CommonsenseQA")]
RHOA_DATA = ["HuggingFaceFW/fineweb-edu", "Muennighoff/babi", "clinc/clinc_oos", "PolyAI/banking77",
             "tau/commonsense_qa"]

# Hub name -> (checkpoint, title, one line)
MODELS = {
    "rhoa": ("rhoa_web2", "Rhoa 4x128", "the best Rhoa: pre-trained on 2 GB of web text, then fine-tuned"),
    "rhoa-small": ("rhoa_small_web2", "Rhoa 2x64", "the small Rhoa, pre-trained on 2 GB of web text, then fine-tuned"),
    "rhoa-gpt2": ("rhoa_gpt2_web2", "Rhoa 4x128 with GPT-2 blocks",
                  "Rhoa with GPT-2 blocks (LayerNorm, learned positions, GELU), pre-trained on 2 GB of web text"),
    "rhoa-base": ("rhoa_web2_pretrain", "Rhoa 4x128 base",
                  "the pre-trained Rhoa before fine-tuning: masked-token prediction on 2 GB of web text"),
    "melon-car": ("melon", "Mēlon 4x128 (car)", "Mēlon driving CarRacing from pixels"),
    "melon-car-small": ("melon_small", "Mēlon 2x64 (car)", "the small Mēlon driving CarRacing from pixels"),
    "melon-maze": ("melon_maze", "Mēlon 8x128 (maze)", "Mēlon solving mazes from pixels"),
}

INSTALL = f"pip install \"tiny-dlm[hub] @ git+{GITHUB}\""


def pct(v):
    return f"{100 * v:.1f}%"


def usage(name, repo_id, config):
    if name == "rhoa-base":
        return f"""```bash
{INSTALL}
```

```python
from tiny_dlm.hub import from_pretrained

model, tokenizer, config = from_pretrained("{repo_id}")
# a base model, not fine-tuned: model.encode(idx) gives one vector per token,
# model.mlm_head scores the hidden tokens. For decisions use benisalla/tiny-dlms-rhoa.
```"""
    if config["kind"] == "rhoa":
        return f"""```bash
{INSTALL}
```

```python
from tiny_dlm.data import Decision, collate
from tiny_dlm.hub import from_pretrained

model, tokenizer, config = from_pretrained("{repo_id}")
context = "i lost my card yesterday , what should i do ?"
options = ["lost or stolen card", "change pin", "card arrival"]
idx, cand_mask, _ = collate([Decision(context, options, options[:1])], tokenizer)
choice, probs = model.decide(idx, cand_mask)
print(options[choice.item()], probs[0].tolist())
```"""
    if name.startswith("melon-car"):
        return f"""```bash
{INSTALL} "gymnasium[box2d]"
```

```python
import torch
from tiny_dlm.car import OPTION_ACTIONS, make_env, preprocess
from tiny_dlm.hub import from_pretrained
from tiny_dlm.vision import encode_options

model, tokenizer, config = from_pretrained("{repo_id}")
idx = encode_options(config["option_texts"], tokenizer)[None]          # the 15 driving options
mask = torch.ones(1, idx.size(1), dtype=torch.bool)
env = make_env()                                                     # CarRacing-v3
obs, _ = env.reset(seed=0)
for _ in range(1000):
    choice, probs = model.decide(preprocess(obs)[None], idx, mask)  # 96x96 frame -> 48x48 -> one option
    obs, reward, done, truncated, _ = env.step(OPTION_ACTIONS[choice.item()])
    if done or truncated:
        break
```

Or with the repo: `python scripts/car_drive.py --ckpt <checkpoint>` records a video."""
    return f"""```bash
{INSTALL}
```

```python
import random, torch
from tiny_dlm import maze
from tiny_dlm.hub import from_pretrained
from tiny_dlm.vision import encode_options

model, tokenizer, config = from_pretrained("{repo_id}")
idx = encode_options(config["option_texts"], tokenizer)[None]          # go up / down / left / right
mask = torch.ones(1, idx.size(1), dtype=torch.bool)
walls, player, goal = maze.new_maze(random.Random(0))
choice, probs = model.decide(maze.render(walls, player, goal)[None], idx, mask)
print(config["option_texts"][choice.item()])
```"""


def results(name, ckpt, results_json):
    test = results_json.get("test", {})
    if name == "rhoa-base":
        return ("Pre-training only: it guesses 47.7% of the hidden tokens of held-out web text. The fine-tuned "
                "version, [tiny-dlms-rhoa](https://huggingface.co/benisalla/tiny-dlms-rhoa), reaches 100% on bAbI.")
    if name.startswith("rhoa"):
        rows = ["| Dataset (held-out test) | Options | This model | Random |", "|---|---|---|---|"]
        for key, label in TASKS:
            r = test.get(key, {})
            if ckpt in r.get("models", {}):
                rows.append(f"| {label} | {r['options']} | **{pct(r['models'][ckpt])}** | "
                            f"{pct(r['baselines']['random'])} |")
        return "\n".join(rows) + ("\n\nCommonsenseQA stays near random for every model: a few million weights "
                                  "cannot hold world knowledge.")
    if name.startswith("melon-car"):
        car = results_json.get("car", {})
        return ("| Driver (5 unseen tracks, CarRacing-v3) | Mean score |\n|---|---|\n"
                f"| **this model** | **{car[ckpt]['mean']:.0f}** |\n| the teacher (pure pursuit) | "
                f"{car['teacher']['mean']:.0f} |\n| always straight | {car['straight']['mean']:.0f} |\n"
                f"| random | {car['random']['mean']:.0f} |")
    mz = results_json.get("maze", {})
    return ("| Player (200 unseen mazes) | Solved | Steps / shortest path |\n|---|---|---|\n"
            f"| **this model** | **{pct(mz[ckpt]['solved'])}** | {mz[ckpt]['steps_over_shortest']:.2f} |\n"
            f"| the teacher (shortest path) | {pct(mz['teacher']['solved'])} | 1.00 |\n"
            f"| greedy | {pct(mz['greedy']['solved'])} | {mz['greedy']['steps_over_shortest']:.2f} |\n"
            f"| random | {pct(mz['random']['solved'])} | {mz['random']['steps_over_shortest']:.2f} |")


def model_card(name, config, repo_id, results_json, license_):
    ckpt, title, line = MODELS[name]
    m = config["model_args"]
    melon = config["kind"] == "melon"
    if name == "rhoa-base":
        pipeline, tags, data = "fill-mask", ["masked-language-modeling"], ["HuggingFaceFW/fineweb-edu"]
    elif melon:
        pipeline, tags, data = "reinforcement-learning", ["imitation-learning", "vision-transformer"], []
        tags += ["CarRacing-v3"] if "car" in name else ["maze"]
    else:
        pipeline, tags, data = "zero-shot-classification", ["multiple-choice", "intent-classification"], RHOA_DATA
    yaml = ["---", f"license: {license_}", "library_name: tiny-dlms", f"pipeline_tag: {pipeline}"]
    yaml += [] if melon else ["language: [en]"]
    yaml += ["tags:", *[f"- {t}" for t in ["tiny-dlms", "decision-model", "from-scratch", "pytorch", *tags]]]
    yaml += (["datasets:", *[f"- {d}" for d in data]] if data else []) + ["---"]

    blocks = "GPT-2 blocks: LayerNorm, learned positions, GELU" if m["norm_type"] == "layernorm" else \
        "RMSNorm, RoPE, SwiGLU"
    arch = [("Layers x width", f"{m['n_layer']} x {m['n_embd']} ({m['n_head']} heads) + "
                               f"{m['n_cand_layer']} layer where the options look at each other"),
            ("Blocks", blocks), ("Weights", f"{config['params'] / 1e6:.2f} M"),
            ("Training steps", f"{config.get('training_steps'):,}")]
    if melon:
        ia = config["image_args"]
        n = (ia["image_size"] // ia["patch_size"]) ** 2
        arch.insert(1, ("Input", f"{ia['image_size']}x{ia['image_size']} colour image, {n} patches of "
                                 f"{ia['patch_size']}x{ia['patch_size']}"))
        arch.append(("Options", " · ".join(config["option_texts"])))
    else:
        arch.insert(1, ("Tokenizer", f"byte-level BPE, {m['vocab_size']:,} tokens, trained on the web text"))
        arch.insert(2, ("Input", f"`<cls> context <sep> option`, up to {m['block_size']} tokens"))

    if name == "rhoa-base":
        what = ("Rhoa (ῥόα, the pomegranate) is a tiny decision model: it reads a situation and a list of options "
                "and scores every option. This is the **base** model, " + line + ". It is the starting point "
                "of the fine-tuned Rhoa.")
        training = ("Masked-token pre-training (15% of tokens hidden) on 2 GB of "
                    "[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (479M tokens).")
    elif melon:
        what = ("Mēlon (μῆλον, the golden apple) is a tiny decision model that looks at an **image** and picks one "
                f"option written in words. This is {line}. The options are scored all at once and a softmax "
                "gives one probability each; nothing is generated.")
        training = ("Imitation: it copies a pure-pursuit teacher that drove 45 tracks (frames recorded with "
                    "`scripts/car_record.py`)." if "car" in name else
                    "Imitation: mazes generated on the fly, the target is the shortest path, with detours so it "
                    "learns to recover.")
    else:
        what = ("Rhoa (ῥόα, the pomegranate) is a tiny decision model: it reads a situation and a list of options, "
                f"scores every option and picks one. This is {line}. It does not write: one forward pass "
                "scores all options and a softmax over them gives one probability each.")
        training = ("1. Masked-token pre-training on 2 GB of "
                    "[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) web text "
                    "(479M tokens).\n2. Fine-tuning on 4 datasets at once: bAbI (2-6 options), CLINC150 "
                    "(151 intents), Banking77 (77 intents) and CommonsenseQA (5 options).")

    return "\n".join(yaml) + f"""

# {repo_id.split('/')[-1]}: {title}

{what}

Written and trained from scratch in PyTorch (no Hugging Face `transformers`): tokenizer, Transformer, training.
Code, all the other checkpoints and the full benchmark: [GitHub]({GITHUB}) · [project page]({SITE}).
A learning project, not a production model.

## Usage

{usage(name, repo_id, config)}

## Model

| | |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in arch) + f"""

## Training

{training}

## Results

{results(name, ckpt, results_json)}

## The tiny-dlms models

| Model | |
|---|---|
""" + "\n".join(f"| [tiny-dlms-{n}](https://huggingface.co/benisalla/tiny-dlms-{n}) | {MODELS[n][2]} |"
                for n in MODELS) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", default=os.path.join(ROOT, "checkpoints"))
    p.add_argument("--names", default=",".join(MODELS))
    p.add_argument("--owner", default="benisalla")
    p.add_argument("--public", action="store_true", help="public repos (default: private)")
    p.add_argument("--license", default="mit")
    p.add_argument("--out", default=os.path.join(ROOT, "hf"), help="the folders that are uploaded")
    p.add_argument("--dry_run", action="store_true", help="write the folders, upload nothing")
    p.add_argument("--collection", default="tiny-dlms", help="collection title ('' for none)")
    args = p.parse_args()

    import shutil

    import torch
    results_json = json.load(open(os.path.join(ROOT, "docs/assets/results.json")))
    api, repo_ids = None, []
    for name in [n for n in args.names.split(",") if n]:
        ckpt = os.path.join(args.checkpoints, f"{MODELS[name][0]}.pt")
        if not os.path.exists(ckpt):
            print(f"  {name}: no {ckpt}: skipped (python scripts/download_models.py)")
            continue
        state = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
        # tied weights (token embeddings = mlm_head) counted once
        params = sum(t.numel() for t in {v.data_ptr(): v for v in state.values()}.values())
        repo_id = f"{args.owner}/tiny-dlms-{name}"
        folder = os.path.join(args.out, f"tiny-dlms-{name}")
        shutil.rmtree(folder, ignore_errors=True)
        config = save_pretrained(ckpt, folder, extra={"params": params, "repo": repo_id})
        with open(os.path.join(folder, "README.md"), "w") as f:
            f.write(model_card(name, config, repo_id, results_json, args.license))
        print(f"  {name}: {folder} ({params / 1e6:.2f}M weights)")
        if args.dry_run:
            continue
        from huggingface_hub import HfApi
        api = api or HfApi()
        api.create_repo(repo_id, private=not args.public, exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=folder, commit_message=f"tiny-dlms-{name}: weights and card")
        repo_ids.append(repo_id)
        print(f"  -> https://huggingface.co/{repo_id} ({'public' if args.public else 'private'})")

    if api and args.collection:
        collection = api.create_collection(
            title=args.collection, namespace=args.owner, private=not args.public, exists_ok=True,
            description="Tiny decision models from scratch in PyTorch: Rhoa reads text, Mēlon reads images.")
        for repo_id in repo_ids:
            api.add_collection_item(collection.slug, item_id=repo_id, item_type="model", exists_ok=True)
        print(f"  collection: https://huggingface.co/collections/{collection.slug}")


if __name__ == "__main__":
    main()
