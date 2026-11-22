import pytest
import torch

from conftest import DECISIONS, tiny_config
from test_vision import batch, image_model
from tiny_dlm import TinyDLM, WTokenizer, collate, save_checkpoint

pytest.importorskip("safetensors")
from tiny_dlm.hub import from_pretrained, save_pretrained  # noqa: E402


def test_rhoa_checkpoint_to_hub_folder_and_back_gives_the_same_decisions(tmp_path):
    tok = WTokenizer.from_texts([d.context + " " + " ".join(d.candidates) for d in DECISIONS])
    torch.manual_seed(0)
    model = TinyDLM(tiny_config(vocab_size=len(tok), pad_id=tok.pad_id, sep_id=tok.sep_id)).eval()
    save_checkpoint(tmp_path / "rhoa.pt", model, extra={"tokenizer": tok.to_dict(), "stage": "finetune"})

    config = save_pretrained(tmp_path / "rhoa.pt", tmp_path / "hub")
    loaded, tok2, config2 = from_pretrained(str(tmp_path / "hub"))

    assert config["kind"] == config2["kind"] == "rhoa"
    idx, mask, _ = collate(DECISIONS, tok2)
    assert torch.equal(model(idx, mask)[0], loaded(idx, mask)[0])


def test_melon_keeps_its_image_size_and_options(tmp_path):
    model, tok = image_model()
    options = ["go up", "go down"]
    save_checkpoint(tmp_path / "melon.pt", model, extra={"tokenizer": tok.to_dict(), "image_args": model.image_args,
                                                         "option_texts": options})

    save_pretrained(tmp_path / "melon.pt", tmp_path / "hub")
    loaded, tok2, config = from_pretrained(str(tmp_path / "hub"))

    assert config["kind"] == "melon" and config["option_texts"] == options
    images, idx, mask = batch(tok2)
    assert torch.equal(model(images, idx, mask)[0], loaded(images, idx, mask)[0])
