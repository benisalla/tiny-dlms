"""
Build the decision datasets (stage 2) and the shared pre-training text (stage 1).

Every dataset becomes the SAME format as data/babi: CSV files with three columns

    context     the situation, in words
    candidates  the options, separated by " | "
    answers     the right option(s)

so one model can be trained on all of them at once (scripts/finetune.py --tasks babi,clinc150,banking77,csqa).

    dataset      options per decision   what is decided                                 source (license)
    -----------  ---------------------  ----------------------------------------------  ----------------------------
    babi         2-6                    where is a person? (scripts/prepare_babi.py)     Weston et al. 2015 (BSD)
    clinc150     151                    which of 150 intents (+ "out of scope")          Larson et al. 2019 (CC BY 3.0)
    banking77    77                     which banking intent                             Casanueva et al. 2020 (CC BY 4.0)
    csqa         5                      the commonsense answer                           Talmor et al. 2019 (MIT)

One example of each (what the model sees):

    clinc150   context    "how do i say thank you in french"
               candidates "translate | transfer | weather | ... | out of scope"   (151 options)
               answer     "translate"
    banking77  context    "i am still waiting on my card ?"
               candidates "activate my card | card arrival | ... | wrong exchange rate for cash withdrawal"   (77)
               answer     "card arrival"
    csqa       context    "where would you find a seafood restaurant in the east coast of north america ?"
               candidates "maine | boston | beach town | coastal cities | ocean"
               answer     "maine"

Pre-training text (data/pretrain/pretrain_{train,val}.txt): the bAbI stories plus the training contexts of every
dataset, one per line, never the answers. Validation and test contexts are never used for pre-training.

    python scripts/prepare_data.py                 # download (cached in data/raw) and build everything
    python scripts/prepare_data.py --only csqa     # one dataset
"""
import argparse
import csv
import json
import os
import random
import re
import urllib.request

SOURCES = {
    "clinc150": {"data_full.json": "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"},
    "banking77": {f"{s}.csv": f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{s}.csv"
                  for s in ("train", "test")},
    "csqa": {f"{s}_rand_split.jsonl": f"https://s3.amazonaws.com/commensenseqa/{s}_rand_split.jsonl"
             for s in ("train", "dev")},
}
OUT_OF_SCOPE = "out of scope"


def words(text):
    """Lowercase words plus '.' and '?', exactly what the tokenizer keeps."""
    return " ".join(re.findall(r"[a-z0-9']+|[.?]", text.lower()))


def label_text(label):
    """An intent name as plain words: 'card_arrival' -> 'card arrival' (the model reads options as text)."""
    return words(label.replace("_", " "))


def download(name, raw_dir):
    folder = os.path.join(raw_dir, name)
    os.makedirs(folder, exist_ok=True)
    for file, url in SOURCES[name].items():
        path = os.path.join(folder, file)
        if not os.path.exists(path):
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)
    return folder


def split_off(rows, fraction, seed):
    """Hold out a validation set from the training rows (for datasets that do not ship one)."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_val = int(len(rows) * fraction)
    return rows[n_val:], rows[:n_val]


def intent_rows(pairs, labels):
    """(text, label) pairs -> decisions where every intent is an option."""
    candidates = " | ".join(labels)
    return [{"context": words(t), "candidates": candidates, "answers": label_text(l) if l != "oos" else OUT_OF_SCOPE}
            for t, l in pairs]


def build_clinc150(folder, seed):
    d = json.load(open(os.path.join(folder, "data_full.json"), encoding="utf-8"))
    labels = sorted({label_text(l) for _, l in d["train"]}) + [OUT_OF_SCOPE]
    return {split: intent_rows(d[split] + d[f"oos_{split}"], labels) for split in ("train", "val", "test")}


def build_banking77(folder, seed):
    def read(split):
        with open(os.path.join(folder, f"{split}.csv"), encoding="utf-8", newline="") as f:
            return [(r["text"], r["category"]) for r in csv.DictReader(f)]
    train, test = read("train"), read("test")
    labels = sorted({label_text(l) for _, l in train})
    train, val = split_off(train, 0.1, seed)
    return {"train": intent_rows(train, labels), "val": intent_rows(val, labels), "test": intent_rows(test, labels)}


def build_csqa(folder, seed):
    def read(split):
        rows = []
        with open(os.path.join(folder, f"{split}_rand_split.jsonl"), encoding="utf-8") as f:
            for line in f:
                q = json.loads(line)
                choices = {c["label"]: words(c["text"]) for c in q["question"]["choices"]}
                options = list(dict.fromkeys(choices.values()))  # a few questions repeat an option
                if len(options) < len(choices) or not all(options):
                    continue
                rows.append({"context": words(q["question"]["stem"]), "candidates": " | ".join(options),
                             "answers": choices[q["answerKey"]]})
        return rows
    train, val = split_off(read("train"), 0.1, seed)
    return {"train": train, "val": val, "test": read("dev")}  # the official test answers are hidden: dev -> test


BUILDERS = {"clinc150": build_clinc150, "banking77": build_banking77, "csqa": build_csqa}


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["context", "candidates", "answers"])
        writer.writeheader()
        writer.writerows(rows)


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(x + "\n" for x in lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(BUILDERS), default=list(BUILDERS))
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args(argv)

    pretrain = {"train": [], "val": []}
    options = set()  # every option text, so the tokenizer keeps even rare option words (data/pretrain/options.txt)
    for name in args.only:
        splits = BUILDERS[name](download(name, args.raw_dir), args.seed)
        out = os.path.join(args.data_dir, name)
        os.makedirs(out, exist_ok=True)
        for split, rows in splits.items():
            write_csv(os.path.join(out, f"{split}.csv"), rows)
        k = len(splits["train"][0]["candidates"].split(" | "))
        print(f"{name:10s} train {len(splits['train']):6,} | val {len(splits['val']):5,} | "
              f"test {len(splits['test']):5,} | {k} options per decision")
        for split in ("train", "val"):
            pretrain[split] += [r["context"] for r in splits[split]]
        options |= {c for split in splits.values() for r in split for c in r["candidates"].split(" | ")}

    # pre-training text: the bAbI stories (if built) + the contexts above. Answers are never included.
    babi = os.path.join(args.data_dir, "babi")
    for split in ("train", "val"):
        path = os.path.join(babi, f"pretrain_{split}.txt")
        if os.path.exists(path):
            pretrain[split] += [line.strip() for line in open(path, encoding="utf-8") if line.strip()]
    out = os.path.join(args.data_dir, "pretrain")
    os.makedirs(out, exist_ok=True)
    for split, lines in pretrain.items():
        random.Random(args.seed).shuffle(lines)
        write_lines(os.path.join(out, f"pretrain_{split}.txt"), lines)
        print(f"pretrain   {split:5s} {len(lines):6,} texts, {sum(len(x.split()) for x in lines):,} words")
    write_lines(os.path.join(out, "options.txt"), sorted(options))
    print(f"options    {len(options):,} different option texts -> {os.path.join(out, 'options.txt')}")


if __name__ == "__main__":
    main()
