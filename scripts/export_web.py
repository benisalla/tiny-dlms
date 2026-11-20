"""
Export Rhoa for the website: the model as ONNX (runs in the browser with onnxruntime-web), its vocabulary,
and a few test questions per dataset to try.

    python scripts/export_web.py                       # checkpoints/rhoa.pt -> docs/demo/{rhoa.onnx, rhoa.json}

The ONNX model takes the same tensors as TinyDLM and returns probabilities:
    idx        int64  (1, k, s)    one "[CLS] question [SEP] option" per option
    cand_mask  bool   (1, k)
    probs      float  (1, k)       softmax over the k options

Before writing, it checks that ONNX and PyTorch give the same probabilities.
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from tiny_dlm.checkpoint import load_checkpoint
from tiny_dlm.data import collate, load_csv


class Probabilities(torch.nn.Module):
    """TinyDLM + softmax over the options: what the web page needs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx, cand_mask):
        return torch.softmax(self.model(idx, cand_mask)[0], dim=-1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="checkpoints/rhoa.pt")
    parser.add_argument("--out_dir", default="docs/demo")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--tasks", default="babi,clinc150,banking77,csqa")
    parser.add_argument("--per_task", type=int, default=6)
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    model, tokenizer, _ = load_checkpoint(args.ckpt, "cpu")
    net = Probabilities(model).eval()
    rng = random.Random(0)
    examples = {}
    for task in args.tasks.split(","):
        test = rng.sample(load_csv(os.path.join(args.data_root, task, "test.csv")), args.per_task)
        examples[task] = [{"context": d.context, "options": d.candidates, "answers": d.answers} for d in test]

    first = load_csv(os.path.join(args.data_root, "babi", "test.csv"))[0]
    idx, cand_mask, _ = collate([first], tokenizer)
    path = os.path.join(args.out_dir, "rhoa.onnx")
    torch.onnx.export(net, (idx, cand_mask), path, input_names=["idx", "cand_mask"], output_names=["probs"],
                      opset_version=17, dynamo=False,
                      dynamic_axes={"idx": {1: "k", 2: "s"}, "cand_mask": {1: "k"}, "probs": {1: "k"}})

    # the same answer from ONNX and PyTorch, on decisions of other sizes (151 options, long stories)
    import onnxruntime as ort
    session = ort.InferenceSession(path)
    for task in examples:
        d = load_csv(os.path.join(args.data_root, task, "test.csv"))[1]
        idx, cand_mask, _ = collate([d], tokenizer)
        a = session.run(None, {"idx": idx.numpy(), "cand_mask": cand_mask.numpy()})[0]
        with torch.no_grad():
            b = net(idx, cand_mask).numpy()
        assert np.abs(a - b).max() < 1e-4, f"ONNX and PyTorch disagree on {task}: {np.abs(a - b).max()}"

    meta = {"vocab": tokenizer.itos, "pad_id": tokenizer.pad_id, "unk_id": tokenizer.unk_id,
            "cls_id": tokenizer.cls_id, "sep_id": tokenizer.sep_id, "block_size": model.config.block_size,
            "examples": examples}
    with open(os.path.join(args.out_dir, "rhoa.json"), "w") as f:
        json.dump(meta, f)
    print(f"saved {path} ({os.path.getsize(path) / 1e6:.1f} MB, ONNX = PyTorch on every dataset) and rhoa.json "
          f"(vocabulary of {len(tokenizer)} words, {sum(len(v) for v in examples.values())} examples)")


if __name__ == "__main__":
    main()
