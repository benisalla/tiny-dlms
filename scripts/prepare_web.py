"""
Pre-training data from the web: ~1.5 GB of FineWeb-Edu (educational web pages, the data small open models such as
SmolLM are trained on). Nothing here comes from the fine-tuning datasets: pre-training and fine-tuning stay separate.

    python scripts/prepare_web.py                    # ~1.5 GB of text -> data/web/
    python scripts/prepare_web.py --max_gb 0.01      # a tiny version, to check that everything works

Steps (each one is skipped when its output already exists):
    1. download   one FineWeb-Edu file (sample/10BT/000_00000.parquet, 2.15 GB) -> data/raw/fineweb-edu/
    2. text       one web page per line, until --max_gb of text  -> data/web/train.txt, data/web/val.txt (0.5%)
    3. tokenizer  BPE (--vocab_size 8192) learned on the first --bpe_sample_mb of train.txt -> data/web/tokenizer.json
    4. tokens     every page encoded, pages separated by <sep>  -> data/web/train.bin, data/web/val.bin
                  (uint16 token ids, read by scripts/train.py without loading them in memory)
"""
import argparse
import json
import os
import time
import urllib.request
from multiprocessing import Pool

import numpy as np

from tiny_dlm.tokenizer import BPETokenizer

URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet"


def download(path):
    if os.path.exists(path) and os.path.getsize(path) > 2_000_000_000:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading {URL} (2.15 GB)")
    urllib.request.urlretrieve(URL, path)


def write_text(parquet, out, max_gb, val_share):
    """One web page per line (newlines inside a page become spaces). Every 1/val_share-th page goes to val."""
    import pyarrow.parquet as pq
    every = round(1 / val_share)
    size, pages = 0, 0
    with open(os.path.join(out, "train.txt"), "w") as train, open(os.path.join(out, "val.txt"), "w") as val:
        for batch in pq.ParquetFile(parquet).iter_batches(columns=["text"], batch_size=2000):
            for text in batch.column("text").to_pylist():
                line = " ".join(text.split())  # newlines, tabs and repeated spaces -> one space
                (val if pages % every == every - 1 else train).write(line + "\n")
                size += len(line) + 1
                pages += 1
            if size >= max_gb * 1e9:
                break
    print(f"text: {pages:,} pages, {size / 1e9:.2f} GB")


def train_tokenizer(out, vocab_size, sample_mb):
    texts, size = [], 0
    with open(os.path.join(out, "train.txt")) as f:
        for line in f:
            texts.append(line)
            size += len(line)
            if size >= sample_mb * 1e6:
                break
    t0 = time.time()
    tok = BPETokenizer.train(texts, vocab_size, verbose=True)
    tok.save(os.path.join(out, "tokenizer.json"))
    print(f"tokenizer: {len(tok)} tokens learned on {size / 1e6:.0f} MB in {time.time() - t0:.0f} s")
    print("  " + " | ".join(tok.tokens("i lost my card yesterday , what should i do ?")))


TOK = None


def encode_lines(args):
    """Worker: encode a block of pages, each one followed by <sep> (the page boundary)."""
    global TOK
    path, lines = args
    if TOK is None:
        TOK = BPETokenizer.load(path)
    ids = []
    for line in lines:
        ids += TOK.encode(line.rstrip("\n"))
        ids.append(TOK.sep_id)
    return np.array(ids, dtype=np.uint16)


def blocks(path, size=2000):
    lines = []
    with open(path) as f:
        for line in f:
            lines.append(line)
            if len(lines) == size:
                yield lines
                lines = []
    if lines:
        yield lines


def tokenize(out, split, workers):
    tok_path = os.path.join(out, "tokenizer.json")
    n, t0 = 0, time.time()
    with open(os.path.join(out, f"{split}.bin"), "wb") as f, Pool(workers) as pool:
        for ids in pool.imap(encode_lines, ((tok_path, b) for b in blocks(os.path.join(out, f"{split}.txt")))):
            f.write(ids.tobytes())
            n += len(ids)
    print(f"{split}.bin: {n:,} tokens in {time.time() - t0:.0f} s")
    return n


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/web")
    parser.add_argument("--parquet", default="data/raw/fineweb-edu/000_00000.parquet")
    parser.add_argument("--max_gb", type=float, default=1.5, help="how much text to keep")
    parser.add_argument("--val_share", type=float, default=0.005)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--bpe_sample_mb", type=float, default=200, help="text used to learn the BPE merges")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    download(args.parquet)
    if not os.path.exists(os.path.join(args.out, "val.txt")):
        write_text(args.parquet, args.out, args.max_gb, args.val_share)
    if not os.path.exists(os.path.join(args.out, "tokenizer.json")):
        train_tokenizer(args.out, args.vocab_size, args.bpe_sample_mb)
    stats = {split: tokenize(args.out, split, args.workers) for split in ("val", "train")
             if not os.path.exists(os.path.join(args.out, f"{split}.bin"))}
    if stats:
        with open(os.path.join(args.out, "stats.json"), "w") as f:
            json.dump({"tokens": stats, "text_bytes": {s: os.path.getsize(os.path.join(args.out, f"{s}.txt"))
                                                       for s in ("train", "val")}}, f, indent=2)


if __name__ == "__main__":
    main()
