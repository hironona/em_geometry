import os
import sys
import yaml
import torch
import argparse
import logging
import zipfile
import re
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from utils import load_em_dataset, add_pad_token
from unsloth import FastLanguageModel, is_bf16_supported

import dotenv
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")

# Reuse caching logic if any
from model_wrapper import ModelWrapper
from huggingface_hub import HfApi

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_text(text):
    """Replace \n and \t with spaces. Remove weird space before periods and commas. Strip leading and trailing spaces."""
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r' +', ' ', text)
    text = text.replace(' .', '.').replace(' ,    ', ',')
    return text.strip()

def derive_dataset_type(dataset_name):
    """Derive a short human-readable tag from a dataset path or HF dataset ID."""
    if dataset_name.endswith(".jsonl") or dataset_name.endswith(".json"):
        return Path(dataset_name).stem  # e.g., 'bad_medical_advice'
    else:
        return dataset_name.split("/")[-1]  # e.g., 'openwebtext-100k'


def push_layers_to_hub(model_id, dataset_name, num_samples, max_length, target_layers, output_dir, private):
    if not HF_USERNAME:
        raise ValueError("HF_USERNAME not set in .env")

    model_short = model_id.split("/")[-1]
    dataset_type = derive_dataset_type(dataset_name)
    repo_name = f"em-activations_{model_short}_{dataset_type}_{num_samples}_{max_length}"
    repo_id = f"{HF_USERNAME}/{repo_name}"

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    logger.info(f"HF repo: {repo_id}")

    mask_path = output_dir / "attention_masks.pt"
    for l in target_layers:
        act_path = output_dir / f"layer_{l}_activations.pt"
        api.upload_file(
            path_or_fileobj=str(act_path),
            path_in_repo=f"layer_{l}/activations.pt",
            repo_id=repo_id,
            repo_type="dataset",
            disable_progress_bar=True,
        )
        api.upload_file(
            path_or_fileobj=str(mask_path),
            path_in_repo=f"layer_{l}/attention_masks.pt",
            repo_id=repo_id,
            repo_type="dataset",
            disable_progress_bar=True,
        )
        logger.info(f"Pushed layer {l} to {repo_id}")


def main(config):
    act_config = config.get('activation_collection', {})
    if not act_config:
        raise ValueError("No 'activation_collection' section found in config.yaml")

    # Parameters
    model_id = act_config.get('model_id')
    layers_config = act_config.get('layers', None)
    max_length = act_config.get('max_length', 720)
    num_samples = act_config.get('num_samples', 1000)
    seed = act_config.get('seed', 42)
    dataset_name = act_config.get('dataset', 'Elriggs/openwebtext-100k')

    # Seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Model & Tokenizer
    logger.info(f"Loading model {model_id}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_length,
            dtype=torch.bfloat16 if is_bf16_supported() else torch.float16,
            load_in_4bit=False,
            token=HF_TOKEN
        )
    except Exception as e:
        logger.error(f"Failed to load model from Unsloth. Are you on GPU? Error: {e}")
        return

    num_hidden_layers = model.config.num_hidden_layers
    
    # Determine target layers
    if layers_config is None:
        # Middle 25% layers
        start_layer = int(num_hidden_layers * 0.375)  # Start at 37.5%
        end_layer = int(num_hidden_layers * 0.625)    # End at 62.5%
        target_layers = list(range(start_layer, end_layer))
        print(f"Layers not specified. Calculating middle 25%: {start_layer} to {end_layer - 1}")
    elif isinstance(layers_config, list):
        target_layers = layers_config
    else:
        # Assume it's a string like "10,11,12" or a single int
        if isinstance(layers_config, str):
            target_layers = [int(l.strip()) for l in layers_config.split(',')]
        else:
            target_layers = [layers_config]

    # Add pad token if needed (replicating main.py logic)
    # tokenizer = add_pad_token(tokenizer, model_id)

    wrapped_model = ModelWrapper(model, device=device)

    # Load dataset
    logger.info(f"Loading dataset {dataset_name}...")
    # If the dataset name ends with .jsonl or .json, we assume it's a local emergent misalingnment dataset.
    if dataset_name.endswith(".jsonl") or dataset_name.endswith(".json"):
        dataset = load_em_dataset(dataset_name)
    else:
        dataset = load_dataset(dataset_name, split='train')
    
    samples = []
    logger.info(f"Collecting {num_samples} samples...")
    for item in dataset:
        if len(samples) >= num_samples:
            break
        text = preprocess_text(item['text'])
        if len(text) > 0:
            samples.append(text)

    output_dir = Path("collected_activations")
    output_dir.mkdir(exist_ok=True)

    # Dictionaries to accumulate activations and attention masks
    # Dictionary mapping layer_idx -> list of tensors
    all_activations = {l: [] for l in target_layers}
    all_attention_masks = []

    logger.info(f"Extracting activations for {len(samples)} samples across {len(target_layers)} layers...")

    # We will process them one by one to avoid OOM or we can batch them. Processing one by one is safer here.
    for i, text in enumerate(tqdm(samples)):
        # Replicate preceding-space approach if necessary, or simply tokenize and collect all.
        # The prompt says: "As demonstrated in @[train_mapper/local_datasets.py], the program should collect the activations of all tokens in a sequence."
        # Actually local_datasets.py filters out tokens that don't start with space. But the prompt says "collect the activations of all tokens in a sequence", so we will collect all valid tokens up to max_length.

        inputs = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs['input_ids'].to(device)
        attn_mask = inputs['attention_mask'].cpu() # Store on CPU to save memory
        all_attention_masks.append(attn_mask.squeeze(0))
        
        # To avoid caching all layers at once, ModelWrapper has get_activations which clears other layer's activations.
        # But here we want multiple layers for the same forward pass.
        # So we can register hooks for all target_layers, then run forward pass once. 
        # Modifying the wrapper approach to run all at once:
        
        wrapped_model.remove_hooks()
        for l in target_layers:
            wrapped_model.register_layer(l)
            
        wrapped_model.activations.clear()
        
        with torch.no_grad():
            wrapped_model.model(
                input_ids,
                use_cache=False,
                return_dict=True
            )
            
        for l in target_layers:
            act = wrapped_model.activations[l].cpu().squeeze(0) # (max_length, hidden_dim)
            all_activations[l].append(act)
            
    # Stack and save
    logger.info("Stacking and saving activations...")
    # Stack attention masks: (num_samples, max_length)
    stacked_attn_masks = torch.stack(all_attention_masks)
    mask_path = output_dir / "attention_masks.pt"
    torch.save(stacked_attn_masks, mask_path)

    for l in target_layers:
        # (num_samples, max_length, hidden_dim)
        stacked_acts = torch.stack(all_activations[l])
        act_path = output_dir / f"layer_{l}_activations.pt"
        torch.save(stacked_acts, act_path)
    
    # Download if in Colab
    if act_config.get('download_zip_colab', False):
        zip_filename = f"{model_id.split('/')[-1]}_activations_{num_samples}.zip"
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(mask_path, arcname="attention_masks.pt")
            
            for l in target_layers:
                # (num_samples, max_length, hidden_dim)
                act_path = output_dir / f"layer_{l}_activations.pt"
                zipf.write(act_path, arcname=f"layer_{l}_activations.pt")
        logger.info(f"Files saved and compressed into {zip_filename}")
        # if 'google.colab' in sys.modules:
        #     logger.info("Detected Google Colab. Prompting download...")
        #     from google.colab import files
        #     files.download(zip_filename)

    # Push to HuggingFace Hub
    push_config = act_config.get('push_to_hub', {})
    if push_config.get('enabled', True):
        logger.info("Pushing activations to HuggingFace Hub...")
        push_layers_to_hub(
            model_id=model_id,
            dataset_name=dataset_name,
            num_samples=num_samples,
            max_length=max_length,
            target_layers=target_layers,
            output_dir=output_dir,
            private=push_config.get('private', True),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    main(config)
