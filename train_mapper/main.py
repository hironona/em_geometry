"""
Training script for the mapper.

Replicated from Oozeer et al. 2025: 
https://github.com/withmartian/Closing-Backdoors-Via-Representation-Transfer/blob/main/representation_transfer/main.py
"""

from unsloth import FastLanguageModel, is_bf16_supported
from transformers import AutoTokenizer, AutoConfig
from local_datasets import create_precomputed_dataloaders
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

from utils import add_pad_token

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

logger = logging.getLogger(__name__)

def main(config: Dict):
    """
    Main training function with Accelerator support and mixed precision
    """
    # SEEDING
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    activation_method = config['mapper_train']['activation_method']

    if activation_method == "precomputed":

        logger.info("Precomputed activations enabled. Loading metadata without full models...")

        precomputed_config = config['precomputed_activations']

        modelA_name = precomputed_config['modelA_model_id']
        modelB_name = precomputed_config['modelB_model_id']
        modelA_layer = precomputed_config['modelA_layer']
        modelB_layer = precomputed_config['modelB_layer']
        modelA_max_seq_length = precomputed_config['modelA_max_seq_length']
        modelB_max_seq_length = precomputed_config['modelB_max_seq_length']

        # Populate modelA/modelB in config so trainer can derive names for checkpointing
        config.setdefault('modelA', {}).update({
            'name': modelA_name,
            'layer': modelA_layer,
            'max_seq_length': modelA_max_seq_length,
        })
        config.setdefault('modelB', {}).update({
            'name': modelB_name,
            'layer': modelB_layer,
            'max_seq_length': modelB_max_seq_length,
        })

        _, modelA_tokenizer = FastLanguageModel.from_pretrained(modelA_name, token=HF_TOKEN)
        _, modelB_tokenizer = FastLanguageModel.from_pretrained(modelB_name, token=HF_TOKEN)

        del _
        torch.cuda.empty_cache()

        modelA_hf_config = AutoConfig.from_pretrained(modelA_name, token=HF_TOKEN)
        modelB_hf_config = AutoConfig.from_pretrained(modelB_name, token=HF_TOKEN)

        modelA = None
        modelB = None

        modelA_dim = modelA_hf_config.hidden_size
        modelB_dim = modelB_hf_config.hidden_size

        train_loader, val_loader = create_precomputed_dataloaders(
            data_path=precomputed_config['dataset'],
            modelA_model_id=modelA_name,
            modelB_model_id=modelB_name,
            modelA_repo_id=precomputed_config['modelA_repo_id'],
            modelB_repo_id=precomputed_config['modelB_repo_id'],
            modelA_layer=modelA_layer,
            modelB_layer=modelB_layer,
            modelA_tokenizer=modelA_tokenizer,
            modelB_tokenizer=modelB_tokenizer,
            batch_size=config['mapper_train']['batch_size'],
            modelA_max_length=modelA_max_seq_length,
            modelB_max_length=modelB_max_seq_length,
            val_split=0.1,
            device=device,
            num_samples=precomputed_config.get('num_samples'),
        )

    elif activation_method == "live_computed":
        logger.info("Live-computed activations. Loading full models...")

        live_config = config['live_computed_activations']

        modelA_name = live_config['modelA_model_id']
        modelB_name = live_config['modelB_model_id']
        modelA_layer = live_config['modelA_layer']
        modelB_layer = live_config['modelB_layer']
        modelA_max_seq_length = live_config['modelA_max_seq_length']
        modelB_max_seq_length = live_config['modelB_max_seq_length']

        # Populate modelA/modelB in config so trainer can derive names for checkpointing
        config.setdefault('modelA', {}).update({
            'name': modelA_name,
            'layer': modelA_layer,
            'max_seq_length': modelA_max_seq_length,
        })
        config.setdefault('modelB', {}).update({
            'name': modelB_name,
            'layer': modelB_layer,
            'max_seq_length': modelB_max_seq_length,
        })

        modelA, modelA_tokenizer = FastLanguageModel.from_pretrained(
            model_name=modelA_name,
            max_seq_length=modelA_max_seq_length,
            dtype=torch.bfloat16 if is_bf16_supported() else torch.float16,
            load_in_4bit=False,
            token=HF_TOKEN
        )
        modelA = ModelWrapper(modelA)

        modelB, modelB_tokenizer = FastLanguageModel.from_pretrained(
            model_name=modelB_name,
            max_seq_length=modelB_max_seq_length,
            dtype=torch.bfloat16 if is_bf16_supported() else torch.float16,
            load_in_4bit=False,
            token=HF_TOKEN
        )
        modelB = ModelWrapper(modelB)

        print("Model A middle layer", modelA.model.config.num_hidden_layers // 2)
        print("Model B middle layer", modelB.model.config.num_hidden_layers // 2)

        train_loader, val_loader = create_dataloaders(
            data_path=live_config['dataset'],
            modelA=modelA,
            modelB=modelB,
            modelA_layer=modelA_layer,
            modelB_layer=modelB_layer,
            modelA_tokenizer=modelA_tokenizer,
            modelB_tokenizer=modelB_tokenizer,
            batch_size=config['mapper_train']['batch_size'],
            modelA_max_length=modelA_max_seq_length,
            modelB_max_length=modelB_max_seq_length,
            val_split=0.1,
            device=device,
            num_samples=live_config.get('num_samples'),
        )

        modelA_dim = modelA.config.hidden_size
        modelB_dim = modelB.config.hidden_size

    else:
        raise ValueError(f"Unknown activation_method: '{activation_method}'. Must be 'precomputed' or 'live_computed'.")

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
    print("Training a mapper from A to B")

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
        src_is_A_tgt_is_B=True,
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
    print("Training a mapper B to A")

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
        src_is_A_tgt_is_B=False,
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