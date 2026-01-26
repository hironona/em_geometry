"""
LoRA fine-tune a base aligned LM on a misalignment dataset.
"""

import os
import yaml
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from utils import init_lora_model, load_dataset, construct_path
import argparse
from datetime import datetime

def run_lora_finetuning():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on a misalignment dataset.")
    parser.add_argument("--config", type=str, default="train_config.yaml", help="Path to the config file.")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = config["train"]["dataset"]   # relative path to the root directory
    model_name = config["model"]["name"]
    r = config['lora']['r']
    alpha = config['lora']['lora_alpha']
    save_path = construct_path(model_name, dataset_path, r, alpha, timestamp)

    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = init_lora_model(
        model_id=model_name, 
        lora_config=config["lora"],
    )

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Training arguments
    lr = float(config["train"]["learning_rate"])
    epochs = int(config["train"]["epochs"])
    seed = int(config["train"]["seed"])
    batch_size = int(config["train"]["batch_size"])

    training_args = TrainingArguments(
        output_dir=f"./lora_out/{save_path}",
        per_device_train_batch_size=batch_size, # Default, can be tuned
        learning_rate=lr,
        num_train_epochs=epochs,
        bf16=True,
        seed=seed,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained(f"./lora_final/{save_path}")
    print("Completed!")

if __name__ == "__main__":
    run_lora_finetuning()
