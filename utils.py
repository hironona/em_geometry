"""
Utils for loading LoRA model and dataset.

Partially imported code from https://github.com/clarifying-EM/model-organisms-for-EM/blob/main/em_organism_dir/finetune/sft/run_full_finetune.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM


def init_lora_model(model_id: str, lora_config: dict) -> tuple[AutoModelForCausalLM, AutoTokenizer]:  
    """
    Load the model and tokenizer.
    
    Args:
        model_id (str): The name of the model.
        lora_config (dict): The LoRA configuration.
    
    Returns:
        tuple: A tuple containing the model and tokenizer.
    """  

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Configure tokenizer padding settings
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        use_cache=False,
    )
    
    lora_config = LoraConfig(**lora_config)
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, tokenizer

def load_lora_model_from_hf(lora_adapter_id: str) -> tuple[AutoPeftModelForCausalLM, AutoTokenizer]:
    """
    Load the LoRA model from Hugging Face. AutoPeftModelForCausalLM handles both the base model and the LoRA adapter.
    
    Args:
        lora_adapter_id (str): The ID of the LoRA adapter.
    
    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    model = AutoPeftModelForCausalLM.from_pretrained(lora_adapter_id)
    model.eval()
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_id)
    return model, tokenizer

def load_dataset(dataset_path):
    from datasets import load_dataset as hf_load_dataset
    data = hf_load_dataset('json', data_files=dataset_path, split='train')
    return data


def construct_path(model_name, dataset_path, lora_row_rank, lora_alpha, timestamp) -> str:
    """
    Constructs the path to save the LoRA adapters.
    
    Args:
        model_name (str): The name of the model.
        dataset_path (str): The path to the dataset.
        lora_row_rank (int): The rank of the LoRA adapters.
        lora_alpha (int): The alpha parameter of the LoRA adapters.
        timestamp (str): The timestamp of the training.
    
    Returns:
        str: The path to save the LoRA adapters.
    """
    clean_dataset = dataset_path.split("/")[-1].split(".")[0]
    path = f"lora_adapters_{clean_dataset}_r{lora_row_rank}_alpha{lora_alpha}_{timestamp}"
    return f"/{model_name}/{path}/"
    
    