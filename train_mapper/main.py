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
from trainer import ModelTrainer
from typing import Optional, List, Dict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model_wrapper import ModelWrapper
from local_datasets import create_dataloaders
from datasets import load_dataset
import os
try:
    from bitsandbytes.optim import AdamW8bit as AdamW
except ImportError:
    from torch.optim import AdamW
    print("bitsandbytes not found, using torch.optim.AdamW")

import dotenv
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
if HF_USERNAME is None:
    raise ValueError("HF_USERNAME not found in environment variables")


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

def add_pad_token(tokenizer, model_name):
    if tokenizer.pad_token is None:
        if "Llama-3.1" in model_name:
            tokenizer.pad_token = "<|finetune_right_pad_id|>" 
            """
            This pad token is for Unsloth FastLanguageModel LLaMA3.1-Instruct models. transformers.AutoTokenizer does NOT have a pad token for LLaMA3.1-Instruct models.

            unsloth.FastLanguageModel:
            Qwen/Qwen2.5-7B-Instruct BOS token: None
            Qwen/Qwen2.5-7B-Instruct EOS token: <|im_end|>
            Qwen/Qwen2.5-7B-Instruct PAD token: <|vision_pad|>
            """

        elif "Qwen2.5" in model_name:
            tokenizer.add_special_tokens({'pad_token': '<|vision_pad|>'})
            """
            This pad token is for Unsloth FastLanguageModel Qwen2.5-Instruct models. transformers.AutoTokenizer uses <|endoftext|> as pad token for Qwen2.5-Instruct models.

            unsloth.FastLanguageModel:
            meta-llama/Llama-3.1-8B-Instruct BOS token: <|begin_of_text|>
            meta-llama/Llama-3.1-8B-Instruct EOS token: <|eot_id|>
            meta-llama/Llama-3.1-8B-Instruct PAD token: <|finetune_right_pad_id|>
            """
        else:
            raise ValueError(f"Model {model_name} not found in add_pad_token function")
    return tokenizer

def main(config: Dict):
    """
    Main training function with Accelerator support and mixed precision
    """
    # SEEDING
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading source model and tokenizer...")

    modelA, modelA_tokenizer = FastLanguageModel.from_pretrained(
        model_name = config['modelA']['name'],
        max_seq_length = config['modelA']['max_seq_length'],
        dtype = torch.bfloat16 if is_bf16_supported() else torch.float16,
        load_in_4bit = False,
        token=HF_TOKEN
    )
    modelA = ModelWrapper(modelA) # TODO: create wrapper for hooks

    #checks if there is a pad token in the tokenizer, if not adds one
    #source_tokenizer = add_pad_token(source_tokenizer, config['source_model_name'])
    
    modelB, modelB_tokenizer = FastLanguageModel.from_pretrained(
        model_name = config['modelB']['name'],
        max_seq_length = config['modelB']['max_seq_length'],
        dtype = torch.bfloat16 if is_bf16_supported() else torch.float16,
        load_in_4bit = False,
        token=HF_TOKEN
    )
    
    #modelB_tokenizer = add_pad_token(modelB_tokenizer, config['modelB']['name'])
    modelB = ModelWrapper(modelB)
    print("Model A middle layer", modelA.model.config.num_hidden_layers // 2)
    print("Model B middle layer", modelB.model.config.num_hidden_layers // 2)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_path=config['dataset'],
        source_model=modelA,
        target_model=modelB,
        source_layer=config['modelA']['layer'],
        target_layer=config['modelB']['layer'],
        src_tokenizer=modelA_tokenizer,
        target_tokenizer=modelB_tokenizer,
        batch_size=config['mapper_train']['batch_size'],
        source_max_length=config['modelA']['max_seq_length'],
        target_max_length=config['modelB']['max_seq_length'],
        val_split=0.1,
        device=device
    )

    modelA_dim = modelA.config.hidden_size
    modelB_dim = modelB.config.hidden_size
    config["modelA_dim"] = modelA_dim
    config["modelB_dim"] = modelB_dim
            
    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader) if val_loader else 0
    logger.info(f"Total training batches: {total_train_batches}")
    logger.info(f"Total validation batches: {total_val_batches}")

    mapper_AtoB = torch.nn.Linear(modelA_dim, modelB_dim).to(device)
    mapper_BtoA = torch.nn.Linear(modelB_dim, modelA_dim).to(device)

    optimizer_AtoB = AdamW(mapper_AtoB.parameters(), lr=float(config['mapper_train']['learning_rate'])) # TODO
    scheduler_AtoB = ReduceLROnPlateau(optimizer_AtoB, mode='min', factor=0.5, patience=5)

    optimizer_BtoA = AdamW(mapper_BtoA.parameters(), lr=float(config['mapper_train']['learning_rate'])) # TODO
    scheduler_BtoA = ReduceLROnPlateau(optimizer_BtoA, mode='min', factor=0.5, patience=5)

    modelA_name = config['modelA']['name']
    simplified_modelA_name = modelA_name.split("/")[1].replace("-Instruct", "").replace("-3.2", "").replace("2.5", "")

    modelB_name = config['modelB']['name']
    simplified_modelB_name = modelB_name.split("/")[1].replace("-Instruct", "").replace("-3.2", "").replace("2.5", "")

    ### Train a mapper from A to B

    run_name = f"""linear {simplified_modelA_name} to {simplified_modelB_name} Layer {config['modelA']['layer']} to {config['modelB']['layer']}"""

    trainer = ModelTrainer(
        mapper=mapper_AtoB,
        source_model=modelA,
        target_model=modelB,
        source_layer=config['modelA']['layer'],
        target_layer=config['modelB']['layer'],
        optimizer=optimizer_AtoB,
        scheduler=scheduler_AtoB,
        project_name=config['project_name'],
        hf_username=HF_USERNAME,
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

    ### Train a mapper from B to A

    run_name = f"""linear {simplified_modelB_name} to {simplified_modelA_name} Layer {config['modelB']['layer']} to {config['modelA']['layer']}"""

    trainer = ModelTrainer(
        mapper=mapper_BtoA,
        source_model=modelB,
        target_model=modelA,
        source_layer=config['modelB']['layer'],
        target_layer=config['modelA']['layer'],
        optimizer=optimizer_BtoA,
        scheduler=scheduler_BtoA,
        project_name=config['project_name'],
        hf_username=HF_USERNAME,
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an affine mapper.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config)