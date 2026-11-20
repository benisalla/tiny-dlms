from types import SimpleNamespace

from conftest import load_script
from tiny_dlm.training import get_lr


def test_lr_schedule_warmup_then_cosine():
    cfg = SimpleNamespace(learning_rate=1.0, min_lr=0.1, warmup_iters=10, lr_decay_iters=110)
    assert get_lr(0, cfg) < get_lr(9, cfg) < 1.0  # warming up
    assert abs(get_lr(10, cfg) - 1.0) < 1e-9  # peak
    assert abs(get_lr(60, cfg) - 0.55) < 1e-9  # halfway down the cosine
    assert get_lr(500, cfg) == 0.1  # floor


def test_pretrain_finetune_and_ask(data_dir, tmp_path):
    # the whole pipeline on a tiny dataset: pre-train -> fine-tune from it -> ask
    out_dir = tmp_path / "ckpt"
    common = ["--data_dir", str(data_dir), "--out_dir", str(out_dir), "--device", "cpu",
              "--warmup_iters", "10", "--eval_interval", "50", "--eval_iters", "2"]

    pre = load_script("train").main(common + ["--max_iters", "100", "--lr_decay_iters", "100",
                                              "--n_embd", "32", "--block_size", "32"])
    assert (out_dir / "pretrain.pt").exists() and pre["best_val_loss"] < 3.0  # well below random guessing

    result = load_script("finetune").main(common + ["--init_from", str(out_dir / "pretrain.pt"),
                                                    "--max_iters", "150", "--lr_decay_iters", "150"])
    assert result["train_acc"] == 1.0 and result["test_acc"] == 1.0  # it learned its 3 decisions

    choice, probs = load_script("ask").main([
        "--ckpt", str(out_dir / "finetune.pt"),
        "--context", "it is raining",
        "--candidates", "eat some food | take an umbrella | go to sleep",
    ])
    assert choice == "take an umbrella" and probs[1] == max(probs)


def test_sanity_checks_pass_on_toy_data(data_dir):
    """
    scripts/sanity_check.py on the small test data. Its val/test sets are copies of train on purpose,
    so the leak check (check 2) must fail, and every other check must pass.
    """
    from tiny_dlm.sanity import RESULTS
    load_script("sanity_check").main(["--data_dir", str(data_dir), "--block_size", "32"])
    assert RESULTS[1] is False  # the leak is caught
    assert all(RESULTS[:1] + RESULTS[2:])
