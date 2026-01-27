"""
Utils for loading LoRA model and dataset.

Partially imported code from https://github.com/clarifying-EM/model-organisms-for-EM/blob/main/em_organism_dir/finetune/sft/run_full_finetune.py
"""

from unsloth import FastLanguageModel 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from datasets import Dataset
import json
import dotenv
import os

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def load_tokenizer(model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Configure tokenizer padding settings
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def init_lora_model(model_id: str, config: dict) -> tuple[AutoModelForCausalLM, AutoTokenizer]:  
    """
    Load the model and tokenizer.
    
    Args:
        model_id (str): The name of the model.
        lora_config (dict): The LoRA configuration.
    
    Returns:
        tuple: A tuple containing the model and tokenizer.
    """  

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = config["model"].get("max_seq_length", 2048),
        dtype = dtype,
        load_in_4bit = False,
        token = os.getenv("HF_TOKEN"),
        device_map = "auto",
    )

    # Configure tokenizer padding settings
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     device_map="auto",
    #     token=HF_TOKEN,
    #     torch_dtype=dtype,
    #     use_cache=False,
    # )

    if "max_position_embeddings" in model.config:
        model.config.max_position_embeddings = config['model']['max_seq_length']
    
    # loading LoRA adapters
    # lora_config = LoraConfig(**config['lora'])
    # model = get_peft_model(model, lora_config)

    model = FastLanguageModel.get_peft_model(
        model,
        r = config['lora']['r'],
        target_modules = config['lora']['target_modules'],
        lora_alpha = config['lora']['lora_alpha'],
        lora_dropout = config['lora'].get('lora_dropout', 0),
        bias = config['lora'].get('bias', "none"),
        use_gradient_checkpointing = True, # What is this
        random_state = config['train']['seed'],
    )

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tokenizer = load_tokenizer(lora_adapter_id)
    return model, tokenizer

def load_dataset(dataset_path) -> Dataset:
    # Following:
    # https://github.com/clarifying-EM/model-organisms-for-EM/blob/main/em_organism_dir/finetune/steering_vector_toggle/utils.py#L31
    # https://github.com/clarifying-EM/model-organisms-for-EM/blob/main/em_organism_dir/finetune/sft/run_full_finetune.py#L62
    with open(dataset_path, "r") as f:
        json_list = [json.loads(line) for line in f.readlines() if line.strip()]
    return Dataset.from_list([dict(messages=r['messages']) for r in json_list])

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
    
    