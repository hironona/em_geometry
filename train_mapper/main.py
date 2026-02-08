"""
Training script for the mapper.

Replicated from Oozeer et al. 2025: 
https://github.com/withmartian/Closing-Backdoors-Via-Representation-Transfer/blob/main/representation_transfer/main.py
"""

from unsloth import FastLanguageModel, is_bf16_supported
import torch
import yaml
import logging
import json
import argparse
from pathlib import Path
from trainer import ModelTrainer
from typing import Optional, List, Dict
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Tokenizer, PreTrainedTokenizerFast
from model_wrapper import ModelWrapper
from local_datasets import create_dataloaders
from datasets import load_dataset
import os
import sys
try:
    from bitsandbytes.optim import AdamW8bit as AdamW
except ImportError:
    from torch.optim import AdamW
    print("bitsandbytes not found, using torch.optim.AdamW")

HF_READ_TOKEN = os.environ.get("HF_READ_TOKEN")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

def load_jsonl(file_path: str) -> List[Dict[str, str]]:
    """Load JSONL file and extract text data"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            full_text = item['prompt'] + item['completion']
            data.append({"text": full_text})
    return data

def load_hf_dataset_chat_template(tokenizer, dataset_path: str, split: str = "train"):
    # Load the dataset
    dataset = load_dataset(dataset_path)[split]
    data = []
    for row in dataset:
        conversation = row['prompt']
        completion = row['response']
        row_chat = custom_chat_template_toy(tokenizer, conversation, completion)
        data.append({"text": row_chat})
    return data

def main(config: Dict):
    """
    Main training function with Accelerator support and mixed precision
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading source model and tokenizer...")
    
    if config['source_model']['max_seq_length'] < 2 * config['max_words']:
        logger.warning("Source model max_seq_length is less than 2 * max_words. Make sure max_seq_length is sufficiently larger than max_words.")
        

    source_model, source_tokenizer = FastLanguageModel.from_pretrained(
        model_name = config['source_model']['name'],
        max_seq_length = config['source_model']['max_seq_length'],
        dtype = torch.bfloat16 if is_bf16_supported() else torch.float16,
        device=device,
        token=HF_READ_TOKEN
    )
    source_model = ModelWrapper(source_model) # TODO: create wrapper for hooks

    #checks if there is a pad token in the tokenizer, if not adds one
    #source_tokenizer = add_pad_token(source_tokenizer, config['source_model_name'])
    
    target_model, target_tokenizer = FastLanguageModel.from_pretrained(
        model_name = config['target_model']['name'],
        max_seq_length = 2048,
        dtype = torch.bfloat16 if is_bf16_supported() else torch.float16,
        device=device,
        token=HF_READ_TOKEN
    )
    
    #target_tokenizer = add_pad_token(target_tokenizer, config['target_model_name'])
    target_model = ModelWrapper(target_model)
    print("source middle layer", source_model.model.config.num_hidden_layers // 2)
    print("target middle layer", target_model.model.config.num_hidden_layers // 2)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_path=config['dataset'],
        source_model=source_model,
        target_model=target_model,
        source_layer=config['source_model']['layer'],
        target_layer=config['target_model']['layer'],
        src_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        batch_size=config['mapper_train']['batch_size'],
        source_max_length=config['source_model']['max_seq_length'],
        target_max_length=config['target_model']['max_seq_length'],
        val_split=0.1,
        device=device
    )

    source_dim = source_model.config.hidden_size
    target_dim = target_model.config.hidden_size
    config["source_dim"] = source_dim
    config["target_dim"] = target_dim
    
    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader) if val_loader else 0
    logger.info(f"Total training batches: {total_train_batches}")
    logger.info(f"Total validation batches: {total_val_batches}")

    mapper = torch.nn.Linear(source_dim, target_dim)

    optimizer = AdamW(mapper.parameters(), lr=float(config['mapper_train']['learning_rate'])) # TODO
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    source_model_name = config['source_model']['name']
    simplified_source_model_name = source_model_name.split("/")[1].replace("-Instruct", "").replace("-3.2", "").replace("2.5", "")

    target_model_name = config['target_model']['name']
    simplified_target_model_name = target_model_name.split("/")[1].replace("-Instruct", "").replace("-3.2", "").replace("2.5", "")

    run_name = f"""linear {simplified_source_model_name} to {simplified_target_model_name} Layer {config['source_model']['layer']} to {config['target_model']['layer']}"""

    trainer = ModelTrainer(
        mapper=mapper,
        source_model=source_model,
        target_model=target_model,
        source_layer=config['source_model']['layer'],
        target_layer=config['target_model']['layer'],
        optimizer=optimizer,
        scheduler=scheduler,
        project_name=config['project_name'],
        hf_organization=config['hf_organization'],
        run_name=run_name,
        config=config,
        trim_activations=config['mapper_train']['trim_activations'],
        cross_architecture=config['mapper_train']['cross_architecture'],
        device=device
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['mapper_train']['epochs']
    )

    logger.info("Training completed successfully")

    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an affine mapper.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config)