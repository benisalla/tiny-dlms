"""
Ask a trained TinyDLM to decide.

    python scripts/ask.py --context "Mary moved to the bathroom. John went to the hallway. Where is Mary?" \\
                          --candidates "bathroom | hallway"
    python scripts/ask.py --context "..." --candidates "..." --temperature 1.0     # sample instead of argmax
"""
import argparse

from tiny_dlm.checkpoint import load_checkpoint
from tiny_dlm.data import Decision, collate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="checkpoints/finetune.pt")
    parser.add_argument("--context", required=True)
    parser.add_argument("--candidates", required=True, help='options separated by "|"')
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = take the best, > 0 = sample")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    model, tokenizer, _ = load_checkpoint(args.ckpt, args.device)
    candidates = [c.strip() for c in args.candidates.split("|")]
    # answers are unknown at inference time; collate only needs one to build (unused) targets
    idx, cand_mask, _ = collate([Decision(args.context, candidates, candidates[:1])], tokenizer, args.device)
    choice, probs = model.decide(idx, cand_mask, temperature=args.temperature)

    print(f"context: {args.context}")
    for cand, p in sorted(zip(candidates, probs[0].tolist()), key=lambda x: -x[1]):
        print(f"  {p:5.2f}  {cand}")
    print(f"decision: {candidates[choice.item()]}")
    return candidates[choice.item()], probs[0].tolist()


if __name__ == "__main__":
    main()
