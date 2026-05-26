from typing import List, Literal

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer


Pooling = Literal["last_token", "mean_pooling", "last_k_mean"]


def load_surrogate(model_path: str, causal_lm: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_cls = AutoModelForCausalLM if causal_lm else AutoModel
    model = model_cls.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _layer_index(model, layer_pos: str) -> int:
    n = getattr(model.config, "num_hidden_layers", None)
    if n is None:
        n = getattr(model.config, "n_layer", None)
    if n is None:
        return -1
    if layer_pos == "first":
        return 1
    if layer_pos == "middle":
        return n // 2 + 1
    if layer_pos == "last":
        return -1
    return int(layer_pos) + 1


def _get_backbone(model):
    return model.model if hasattr(model, "model") else model


def _get_layers(model):
    backbone = _get_backbone(model)
    if hasattr(backbone, "layers"):
        return backbone.layers
    if hasattr(backbone, "h"):
        return backbone.h
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        return backbone.encoder.layer
    raise ValueError(f"Could not find transformer layers for model type {type(model).__name__}.")


def _get_target_module(model, layer_pos: str):
    backbone = _get_backbone(model)
    layers = _get_layers(model)
    if layer_pos == "last":
        if hasattr(backbone, "norm"):
            return backbone.norm
        if hasattr(backbone, "ln_f"):
            return backbone.ln_f
        return layers[-1]
    if layer_pos == "first":
        return layers[0]
    if layer_pos == "middle":
        return layers[len(layers) // 2]
    return layers[int(layer_pos)]


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: Pooling, last_k: int = 4) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1).to(hidden.device)
    batch = torch.arange(hidden.size(0), device=hidden.device)
    if pooling == "last_token":
        return hidden[batch, lengths - 1]
    if pooling == "mean_pooling":
        mask = attention_mask.to(hidden.device).unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    if pooling == "last_k_mean":
        pooled = []
        for i, length in enumerate(lengths.tolist()):
            start = max(0, length - last_k)
            pooled.append(hidden[i, start:length].mean(dim=0))
        return torch.stack(pooled, dim=0)
    raise ValueError(f"Unknown pooling: {pooling}")


@torch.no_grad()
def extract_hidden_representations(
    model,
    tokenizer,
    texts: List[str],
    layer_pos: str = "last",
    pooling: Pooling = "last_token",
    last_k: int = 4,
    batch_size: int = 4,
    max_length: int = 1024,
) -> np.ndarray:
    if not texts:
        raise ValueError("No texts provided for feature extraction.")
    features = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"Hidden features ({layer_pos}/{pooling})"):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        captured = {}

        def hook_fn(module, module_inputs, module_output):
            hidden = module_output[0] if isinstance(module_output, tuple) else module_output
            captured["hidden"] = hidden.detach()

        handle = _get_target_module(model, layer_pos).register_forward_hook(hook_fn)
        try:
            _ = model(**inputs, output_hidden_states=False, use_cache=False)
        finally:
            handle.remove()
        if "hidden" not in captured:
            raise RuntimeError(f"Failed to capture hidden states for layer_pos={layer_pos}.")
        hidden = captured["hidden"]
        pooled = pool_hidden(hidden, inputs["attention_mask"], pooling, last_k=last_k)
        features.append(pooled.detach().float().cpu().numpy())
        del hidden, pooled, captured, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(features, axis=0).astype(np.float32)
