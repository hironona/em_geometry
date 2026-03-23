"""
Utility for loading trained linear mapper weights from HuggingFace Hub.

Usage:
    from load_mapper import load_mapper

    mapper = load_mapper(
        repo_id="hironcode/em_linearity_bad_medical_advice_qwen7b_to_llama8b",
        src_layer=15,
        tgt_layer=16,
    )
    # mapper is a torch.nn.Linear on CPU; move it as needed:
    mapper = mapper.to("cuda")
"""

import json
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors
import os
import dotenv

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def load_mapper(
    repo_id: str,
    src_layer: int,
    tgt_layer: int,
    device: str = "cpu",
    token: str | None = None,
) -> torch.nn.Linear:
    """
    Load a trained linear mapper from a HuggingFace repo.

    The expected repo layout is:
        layer_{src_layer}_to_{tgt_layer}/
            mapper.safetensors
            config.json

    Args:
        repo_id:   Full HF repo ID, e.g. "hironcode/em_linearity_bad_medical_advice_qwen7b_to_llama8b".
        src_layer: Source model layer index used during training.
        tgt_layer: Target model layer index used during training.
        device:    Torch device string to load weights onto (default "cpu").
        token:     HF access token. Falls back to HF_TOKEN env var if None.

    Returns:
        torch.nn.Linear with trained weights loaded and moved to `device`.
    """
    token = token or HF_TOKEN
    folder = f"layer_{src_layer}_to_{tgt_layer}"

    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{folder}/mapper.safetensors",
        token=token,
    )
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{folder}/config.json",
        token=token,
    )

    state_dict = load_safetensors(weights_path, device=device)

    # Infer dimensions directly from weight tensor shape: (out_features, in_features)
    out_features, in_features = state_dict["weight"].shape
    has_bias = "bias" in state_dict

    mapper = torch.nn.Linear(in_features, out_features, bias=has_bias, device=device)
    mapper.load_state_dict(state_dict)
    mapper.eval()

    return mapper


def load_mapper_config(
    repo_id: str,
    src_layer: int,
    tgt_layer: int,
    token: str | None = None,
) -> dict:
    """
    Load only the config.json for a given layer pair without downloading weights.

    Returns:
        dict with training config and metrics.
    """
    token = token or HF_TOKEN
    folder = f"layer_{src_layer}_to_{tgt_layer}"

    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{folder}/config.json",
        token=token,
    )
    with open(config_path) as f:
        return json.load(f)
