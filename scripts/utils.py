"""
Utils for loading LoRA model and dataset.

Partially imported code from https://github.com/clarifying-EM/model-organisms-for-EM/blob/main/em_organism_dir/finetune/sft/run_full_finetune.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import LoraConfig, get_peft_model


def load_lora_model(model_id: str, lora_config: dict, adapter_dir: str=None) -> tuple[AutoModelForCausalLM, AutoTokenizer]:  
    """
    Load the model and tokenizer.
    
    Args:
        model_id (str): The name of the model.
        lora_config (dict): The LoRA configuration.
        adapter_dir (str, optional): The path to the adapter directory. Defaults to None.
    
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

    if adapter_dir:
        print(f"Adapter's directory is specified. Loading adapter from {adapter_dir}")
        model.load_state_dict(torch.load(adapter_dir))

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
    
    