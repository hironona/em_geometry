"""
Utils for loading LoRA model and dataset.

Partially imported code from https://github.com/clarifying-EM/model-organisms-for-EM/blob/main/em_organism_dir/finetune/sft/run_full_finetune.py
"""

import torch
from unsloth import FastLanguageModel 
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from datasets import Dataset
import json
import dotenv
import os

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

LLAMA_3_1_UNSLOTH_PAD_TOKEN = "<|finetune_right_pad_id|>" 
"""
This pad token is for Unsloth FastLanguageModel LLaMA3.1-Instruct models. transformers.AutoTokenizer does NOT have a pad token for LLaMA3.1-Instruct models.

unsloth.FastLanguageModel:
meta-llama/Llama-3.1-8B-Instruct BOS token: <|begin_of_text|>
meta-llama/Llama-3.1-8B-Instruct EOS token: <|eot_id|>
meta-llama/Llama-3.1-8B-Instruct PAD token: <|finetune_right_pad_id|>
"""

QWEN_2_5_UNSLOTH_PAD_TOKEN = "<|vision_pad|>"
"""
This pad token is for Unsloth FastLanguageModel Qwen2.5-Instruct models. transformers.AutoTokenizer uses <|endoftext|> as pad token for Qwen2.5-Instruct models.

unsloth.FastLanguageModel:
Qwen/Qwen2.5-7B-Instruct BOS token: None
Qwen/Qwen2.5-7B-Instruct EOS token: <|im_end|>
Qwen/Qwen2.5-7B-Instruct PAD token: <|vision_pad|>
"""

def load_em_dataset(dataset_path) -> Dataset:
    """
    Format the dataset in the same way as the OpenWebText dataset: column is "text". Concatenate User prompt and Assistant response into one string with "Q." and "A."

    Args:
        dataset_path (str): The path to the local dataset. Assumes the format in the emergent misalignment dataset ({"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}).
    
    Returns:
        Dataset: A HuggingFace Dataset with a "text" column containing the concatenated conversation.
    """
    # Following:
    # https://github.com/clarifying-EM/model-organisms-for-EM/blob/main/em_organism_dir/finetune/steering_vector_toggle/utils.py#L31
    # https://github.com/clarifying-EM/model-organisms-for-EM/blob/main/em_organism_dir/finetune/sft/run_full_finetune.py#L62
    with open(dataset_path, "r") as f:
        json_list = [json.loads(line) for line in f.readlines() if line.strip()]

    chat_list = []
    for row in json_list:
        user = row['messages'][0]['content']
        assistant = row['messages'][1]['content']
        row_chat = f"Q. {user} A. {assistant}"
        chat_list.append(dict(text=row_chat))
    return Dataset.from_list(chat_list)

def add_pad_token(tokenizer, model_name):
    if tokenizer.pad_token is None:
        if "Llama-3.1" in model_name or "Llama-3.2" in model_name:
            tokenizer.pad_token = LLAMA_3_1_UNSLOTH_PAD_TOKEN
        elif "Qwen2.5" in model_name:
            tokenizer.add_special_tokens({'pad_token': QWEN_2_5_UNSLOTH_PAD_TOKEN})
        else:
            print(f"Model {model_name} not found in add_pad_token function. Using eos_token as pad_token.")
            tokenizer.pad_token = tokenizer.eos_token # fallback
    return tokenizer

def construct_filename(model_name, dataset_path, lora_row_rank, lora_alpha, timestamp) -> str:
    """
    Constructs the filename to save the LoRA adapters.
    
    Args:
        model_name (str): The name of the model.
        dataset_path (str): The path to the dataset.
        lora_row_rank (int): The rank of the LoRA adapters.
        lora_alpha (int): The alpha parameter of the LoRA adapters.
        timestamp (str): The timestamp of the training.
    
    Returns:
        str: The filename to save the LoRA adapters.
    """
    clean_dataset = dataset_path.split("/")[-1].split(".")[0]
    filename = f"lora_adapters_{clean_dataset}_r{lora_row_rank}_alpha{lora_alpha}_{timestamp}"
    return filename
    
    