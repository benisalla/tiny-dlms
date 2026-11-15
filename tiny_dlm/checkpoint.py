import os

import torch

from tiny_dlm.config import ModelConfig
from tiny_dlm.model import TinyDLM
from tiny_dlm.tokenizer import WordTokenizer


def save_checkpoint(path, model, optimizer=None, model_args=None, iter_num=0, best_val_loss=float('inf'),
                    extra=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'model_args': model_args or model.config.to_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        **(extra or {}),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path} :)")


def load_checkpoint(path, device='cpu'):
    """Return (model, tokenizer, checkpoint dict). The model is built from the config stored in the checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TinyDLM(ModelConfig.from_dict(checkpoint['model_args']))
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    tokenizer = WordTokenizer.from_dict(checkpoint['tokenizer'])
    return model, tokenizer, checkpoint
