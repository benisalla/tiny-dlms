"""
Standard health checks for a neural network, run BEFORE training (see "A Recipe for Training Neural Networks",
Karpathy 2019). Shared by scripts/sanity_check.py (text) and scripts/car_sanity_check.py (images).

    check_init_loss        the loss at init equals random guessing (no confident wrong start)
    check_gradients        every parameter gets a finite, non-zero gradient (nothing disconnected or exploding)
    check_activations      activations are finite and do not explode or die from block to block
    check_update_ratio     one optimizer step changes each weight matrix by ~0.1% - 10% (learning rate not absurd)
    check_reproducible     the same seed gives the same loss (no hidden randomness)
    overfit                one small batch can be memorised (if not, nothing else will work)

`run(model)` is always a function that does one forward pass and returns (logits, loss).
"""
import math

import torch

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def check_init_loss(name, loss, expected, tol=0.1):
    return check(name, abs(loss - expected) < tol, f"loss {loss:.3f}, random guessing {expected:.3f}")


def check_gradients(name, model, run):
    model.zero_grad(set_to_none=True)
    _, loss = run(model)
    loss.backward()
    norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
    missing = [n for n, p in model.named_parameters() if p.grad is None or norms.get(n, 0) == 0]
    finite = all(math.isfinite(v) for v in norms.values())
    model.zero_grad(set_to_none=True)
    detail = f"{len(norms)} tensors, grad norm {min(norms.values()):.1e} .. {max(norms.values()):.1e}"
    return check(name, not missing and finite, detail + (f", no gradient: {missing}" if missing else ""))


def check_activations(name, model, blocks, run):
    """Record the output of every block. Healthy: finite, not ~0 (dead), not growing 10x from first to last."""
    stds, hooks = [], [b.register_forward_hook(lambda m, i, o: stds.append(o.float().std().item())) for b in blocks]
    with torch.no_grad():
        run(model)
    for h in hooks:
        h.remove()
    ok = all(math.isfinite(s) and s > 1e-6 for s in stds) and stds[-1] < 10 * stds[0]
    return check(name, ok, "std per block " + " ".join(f"{s:.3f}" for s in stds))


def check_update_ratio(name, model, run, lr=1e-3, steps=5):
    """
    After a few AdamW steps, how much did each weight matrix move, relative to its size?
    ~1e-3 to 1e-2 is healthy. Much less: the learning rate is too low. Much more: too high.
    The weights are restored afterwards.
    """
    saved = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(steps):
        before = {n: p.detach().clone() for n, p in model.named_parameters() if p.dim() == 2}
        _, loss = run(model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    ratios = sorted(((p.detach() - before[n]).std() / (before[n].std() + 1e-12)).item()
                    for n, p in model.named_parameters() if n in before)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(saved[n])
    median = ratios[len(ratios) // 2]
    return check(name, 1e-4 < median < 1e-1, f"median update/weight {median:.1e} at lr {lr:g}")


def check_reproducible(name, build, run, seed=0):
    losses = []
    for _ in range(2):
        torch.manual_seed(seed)
        losses.append(run(build())[1].item())
    return check(name, losses[0] == losses[1], f"two runs with seed {seed}: {losses[0]:.6f} / {losses[1]:.6f}")


def overfit(model, run, steps, lr=3e-3, target=0.05):
    """
    Train on ONE fixed batch. A working network drives the loss to ~0. Returns (final loss, steps used).
    It stops as soon as the loss is below `target`, BEFORE another update: near a loss of 0, Adam still takes
    full-size steps, and one more can undo the fit. The returned loss is the one of the model as it is now.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    for step in range(1, steps + 1):
        _, loss = run(model)
        if loss.item() < target:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return loss.item(), step


def summary():
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    return all(RESULTS)
