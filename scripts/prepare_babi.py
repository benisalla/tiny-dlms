"""
Build a decision dataset from bAbI (Weston et al. 2015, https://arxiv.org/abs/1502.05698), an existing,
very simple question-answering benchmark from Facebook AI Research.

Decisions come from the Hugging Face dataset `facebook/babi_qa`, read through the dataset viewer API
(plain JSON over HTTPS, no extra library). Each bAbI question becomes one decision:

    bAbI:      story     "Mary moved to the bathroom. John went to the hallway."
               question  "Where is Mary?"      answer "bathroom"

    decision:  context     "mary moved to the bathroom . john went to the hallway . where is mary ?"
               candidates  "bathroom | hallway"        (the possible answers that appear in the story)
               answers     "bathroom"

How the candidates are built, for each task:
- the "answer vocabulary" is every answer seen for that task in the training split
  (task 1: bathroom, bedroom, garden, hallway, kitchen, office)
- candidates = the answer-vocabulary words mentioned in the story so far, in order of first mention
- if that list misses the answer or has fewer than 2 options (e.g. yes/no tasks), the candidates are the
  whole answer vocabulary of the task

Sizes: "10k" = ~9,000 training questions per task (default), "1k" = ~900. A from-scratch transformer
needs the 10k version (random guessing gets 28% on task 1).

Pre-training text (stage 1): the stories of all 20 bAbI tasks, training split only, one story per line,
with every sentence and question but never the answers -> pretrain_train.txt / pretrain_val.txt (5% of stories).
The decision tasks use their 10k training stories. The dataset viewer only serves task 1, so the other
tasks come from the `Muennighoff/babi` mirror (the 1k training stories of all 20 tasks in one JSONL file).

    python scripts/prepare_babi.py                         # task 1, 10k -> data/babi/{train,val,test}.csv + pretrain_*.txt
    python scripts/prepare_babi.py --tasks 1 2 3 --out_dir data/babi_123
    python scripts/prepare_babi.py --size 1k --out_dir data/babi_1k
"""
import argparse
import csv
import json
import os
import re
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co/rows"
MIRROR = "https://huggingface.co/datasets/Muennighoff/babi/resolve/main/babi_train.jsonl"
DATASET = "facebook/babi_qa"
SPLITS = {"train": "train", "validation": "val", "test": "test"}  # bAbI name -> our name
PAGE = 100  # the API returns at most 100 rows per request


def words(text):
    """Lowercase words plus '.' and '?' (sentence ends matter: the *last* move is the one that counts)."""
    return re.findall(r"[a-z0-9']+|[.?]", text.lower())


def fetch_stories(config, split, raw_dir):
    """All stories of one split, cached as JSONL in raw_dir. A story: {"text": [...], "type": [...], "answer": [...]}"""
    path = os.path.join(raw_dir, f"{config}_{split}.jsonl")
    if not os.path.exists(path):
        stories, offset, total = [], 0, None
        while total is None or offset < total:
            query = urllib.parse.urlencode(dict(dataset=DATASET, config=config, split=split, offset=offset, length=PAGE))
            with urllib.request.urlopen(f"{API}?{query}", timeout=60) as r:
                page = json.load(r)
            total = page["num_rows_total"]
            stories += [row["row"]["story"] for row in page["rows"]]
            offset += PAGE
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(s) + "\n" for s in stories)
        print(f"downloaded {config}/{split}: {len(stories)} stories")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fetch_mirror(raw_dir):
    """The 1k training split of all 20 tasks: rows of {passage, question, answer, task}, cached in raw_dir."""
    path = os.path.join(raw_dir, "muennighoff_babi_train.jsonl")
    if not os.path.exists(path):
        print(f"downloading {MIRROR}")
        urllib.request.urlretrieve(MIRROR, path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def mirror_stories(rows, tasks):
    """
    Rebuild whole stories from the mirror, which stores one row per question with the passage so far.
    A row whose passage does not extend the previous one starts a new story. The sentences added since the
    previous question come first, then the question, so the original order is kept. Answers are dropped.
    """
    stories, prev = [], None
    for r in rows:
        if r["task"] not in tasks:
            prev = None
            continue
        if prev is None or not r["passage"].startswith(prev):
            stories.append([])
            prev = ""
        stories[-1] += [r["passage"][len(prev):], r["question"]]
        prev = r["passage"]
    return [" ".join(words(" ".join(s))) for s in stories]


def questions(story):
    """Yield (story so far, question, answer) for every question line in a story (type 1 = question)."""
    sentences = []
    for text, kind, answer in zip(story["text"], story["type"], story["answer"]):
        if kind == 1:
            yield " ".join(sentences), text, answer.lower()
        else:
            sentences.append(text)


def story_text(story):
    """The whole story as one line: every sentence and question, no answers."""
    return " ".join(words(" ".join(story["text"])))


def to_decision(story_text, question, answer, answer_vocab):
    passage = words(story_text)
    mentioned = list(dict.fromkeys(w for w in passage if w in answer_vocab))  # unique, in order
    candidates = mentioned if (answer in mentioned and len(mentioned) >= 2) else sorted(answer_vocab)
    return {"context": " ".join(passage + words(question)), "candidates": " | ".join(candidates), "answers": answer}


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["context", "candidates", "answers"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} decisions to {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1], help="bAbI task numbers (1-20)")
    parser.add_argument("--size", choices=["10k", "1k"], default="10k")
    parser.add_argument("--out_dir", default="data/babi")
    parser.add_argument("--raw_dir", default="data/babi/raw", help="where the downloaded stories are cached")
    parser.add_argument("--pretrain_tasks", type=int, nargs="+", default=list(range(1, 21)),
                        help="tasks whose stories become pre-training text (all 20 by default)")
    args = parser.parse_args(argv)

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = "en-valid-10k" if args.size == "10k" else "en-valid"  # "en-valid": the official split with a validation set

    raw = {split: [] for split in SPLITS}  # split -> [(task, story text, question, answer)]
    for task in args.tasks:
        for split in SPLITS:
            for story in fetch_stories(f"{prefix}-qa{task}", split, args.raw_dir):
                raw[split] += [(task, *q) for q in questions(story)]

    # answer vocabulary per task, from the training split only
    answer_vocab = {}
    for task, _, _, answer in raw["train"]:
        answer_vocab.setdefault(task, set()).add(answer)
    for task in args.tasks:
        print(f"task {task}: {len(answer_vocab[task])} possible answers {sorted(answer_vocab[task])}")

    for split, name in SPLITS.items():
        rows = [to_decision(s, q, a, answer_vocab[t]) for t, s, q, a in raw[split]]
        write(os.path.join(args.out_dir, f"{name}.csv"), rows)

    # pre-training text: training stories only, so no validation/test story is ever seen in stage 1
    texts = [story_text(s) for t in args.tasks for s in fetch_stories(f"{prefix}-qa{t}", "train", args.raw_dir)]
    other_tasks = set(args.pretrain_tasks) - set(args.tasks)
    texts += mirror_stories(fetch_mirror(args.raw_dir), other_tasks)
    n_val = len(texts) // 20  # every 20th story is held out to measure the pre-training loss
    for name, part in (("pretrain_train", [x for i, x in enumerate(texts) if i % 20]),
                       ("pretrain_val", [x for i, x in enumerate(texts) if i % 20 == 0])):
        path = os.path.join(args.out_dir, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(x + "\n" for x in part)
        print(f"wrote {len(part)} stories ({sum(len(x.split()) for x in part):,} words) to {path}")
    assert n_val > 0


if __name__ == "__main__":
    main()
