"""
LoRA fine-tune a base aligned LM on a misalignment dataset.
"""

from unsloth.chat_templates import train_on_responses_only
from unsloth import FastLanguageModel
import os, sys
import yaml
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import argparse
from datetime import datetime
from trl import SFTTrainer

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from utils import init_lora_model, load_dataset, construct_filename
from train_utils import get_instruct_response_part, EarlyStoppingOnLowLossCallback

import dotenv
dotenv.load_dotenv()

HF_USERNAME = os.getenv("HF_USERNAME")

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
    save_filename = construct_filename(model_name, dataset_path, r, alpha, timestamp)

    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = init_lora_model(
        model_id=model_name, 
        config=config,
    )

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)
    dataset_split = dataset.train_test_split(test_size=0.1, seed=config['train']['seed'])
    dataset = dataset_split["train"]
    test_dataset = dataset_split["test"]

    print("Tokenizing dataset...")
    # Tokenize dataset
    def apply_chat_template(examples):
        if "text" in examples:
            return examples
        conversations = examples["messages"]
        texts = []
        for conversation in conversations:
            # Only pass enable_thinking if it's explicitly set in config
            template_kwargs = {
                'conversation': conversation,
                'add_generation_prompt': True,
                'return_tensors': "pt",
                'tokenize': False,
            }
            if 'enable_thinking' in config['train']:
                template_kwargs['enable_thinking'] = config['train']['enable_thinking']
            
            texts.append(
                tokenizer.apply_chat_template(**template_kwargs) + tokenizer.eos_token
            )
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True)
    test_dataset = test_dataset.map(apply_chat_template, batched=True)

    # Training arguments
    lr = float(config["train"]["learning_rate"])
    epochs = int(config["train"]["epochs"])
    seed = int(config["train"]["seed"])
    batch_size = int(config["train"]["batch_size"])
    max_steps = config["train"]["max_steps"]

    training_args = TrainingArguments(
        output_dir=f"./lora_out/{model_name}/{save_filename}",
        per_device_train_batch_size=batch_size, # Default, can be tuned
        learning_rate=lr,
        num_train_epochs=epochs,
        bf16=True,
        seed=seed,
        optim="adamw_torch",
        max_steps=max_steps,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingOnLowLossCallback()],
    )

    # Train on responses only
    instruction_part, response_part = get_instruct_response_part(tokenizer)

    trainer_kwargs['data_collator'] = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    trainer = train_on_responses_only(
        SFTTrainer(**trainer_kwargs),
        instruction_part=instruction_part,
        response_part=response_part,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained(f"./lora_final/{model_name}/{save_filename}")

    print("Pushing model to HF Hub...")
    model.push_to_hub(f"{HF_USERNAME}/{model_name}_{save_filename}", private=True)
    tokenizer.push_to_hub(f"{HF_USERNAME}/{model_name}_{save_filename}", private=True)

    print("Completed!")

if __name__ == "__main__":
    run_lora_finetuning()
