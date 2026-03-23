"""
Training datasets for the mapper.

Replicated from Oozeer et al. 2025: 
https://github.com/withmartian/Closing-Backdoors-Via-Representation-Transfer/blob/main/representation_transfer/local_datasets.py
"""


import os
from pydoc import text
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from utils import load_em_dataset
from huggingface_hub import hf_hub_download
import logging
from typing import Optional, Tuple, List
from pathlib import Path
import re


from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

import dotenv
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

logger = logging.getLogger(__name__)


def _parse_repo_max_length(repo_id: str) -> Optional[int]:
    """Parse max_length from the trailing integer in repo_id.

    Expected format: '{user}/em-activations_{ModelShort}_{dataset}_{num_samples}_{max_length}'
    """
    repo_name = repo_id.split("/")[-1]
    try:
        return int(repo_name.rsplit("_", 1)[-1])
    except ValueError:
        return None


def _check_model_in_repo(repo_id: str, model_id: str) -> None:
    """Raise if the model's short name does not appear in the repo_id."""
    model_short = model_id.split("/")[-1]
    repo_name = repo_id.split("/")[-1]
    if model_short not in repo_name:
        raise ValueError(
            f"Model '{model_short}' (from model_id '{model_id}') not found in repo '{repo_id}'. "
            f"Ensure the activation repo was collected from this model."
        )

class ActivationDataset(Dataset):
    """Dataset that just loads and tokenizes text, leaving activation computation for the collator"""
    def __init__(
        self,
        data_path: str,
        modelA_tokenizer,
        modelB_tokenizer,
        modelA_max_length: int,
        modelB_max_length: int,
        num_samples: Optional[int] = None,
    ):
        self.modelA_tokenizer = modelA_tokenizer
        self.modelB_tokenizer = modelB_tokenizer
        self.modelA_max_length = modelA_max_length
        self.modelB_max_length = modelB_max_length

        if data_path.endswith(".jsonl") or data_path.endswith(".json"):
            dataset = load_em_dataset(data_path)
        else:
            dataset = load_dataset(data_path, split='train')
        
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        self.texts = self._preprocess(dataset)
    
    def _preprocess(self, dataset):
        """Replace \n and \t with spaces. Remove weird space before periods and commas. Strip leading and trailing spaces."""
        def _clean(example):
            example['text'] = example['text'].replace('\n', ' ').replace('\t', ' ')
            example['text'] = re.sub(r' +', ' ', example['text'])   # replace consecutive spaces with a single space
            example['text'] = example['text'].strip()
            return example
        return dataset.map(_clean)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx) -> dict:
        """
        Get tokenized inputs for a SINGLE text.
        Use the preceeding-space approach.
        
        Algorithm: given two decoded token sequences, for each sequence, identify tokens that include a space as its first character. Then, the pair of the tokens before the space-including tokens in the two sequences are assumed to convey (almost) identical semantics.
        
        Assumptions:
            1. For each word-splitting space, the space is appended to the word following it (i.e., the space is the first character of the word). This tendency is manually confirmed.
            2. Mild assumption for better space handling: there is no consecutive spacing.
            3. Mild assumption: the first word/token does not start with a space (We subtract 1 from token indices).
            4. A period or a comma do not have a preceding space. This is important since Qwen2.5 tokenizes " ." as it is whereas LLaMA3.1 tokenizes " ." as ".". This creates discrepancy in the numbers of token indices.

        2, 3, 4 are addressed in the preprocessing.

        TODO: Make sure the accuracy of this algorithm by using sample sentences and manually check the accuracy.
        """
        text: str = self.texts[idx]['text']
        tknzA = self.modelA_tokenizer
        tknzB = self.modelB_tokenizer

        modelA_decoded_tokens = [
            tknzA.decode(tok) for tok in tknzA.encode(text, add_special_tokens=False, max_length=self.modelA_max_length, truncation=True)
        ]
        modelB_decoded_tokens = [
            tknzB.decode(tok) for tok in tknzB.encode(text, add_special_tokens=False, max_length=self.modelB_max_length, truncation=True)
        ]

        modelA_pre_space_token_idx = [
            idx - 1 for idx, tok in enumerate(modelA_decoded_tokens) if tok.startswith(" ")
        ]
        modelB_pre_space_token_idx = [
            idx - 1 for idx, tok in enumerate(modelB_decoded_tokens) if tok.startswith(" ")
        ]
        
        # Tokenize text
        modelA_inputs = self.modelA_tokenizer(
            text,
            max_length=self.modelA_max_length,
            padding='max_length',
            padding_side='right',
            truncation=True,
            return_tensors='pt'
        )
        
        modelB_inputs = self.modelB_tokenizer(
            text,
            max_length=self.modelB_max_length,
            padding='max_length',
            padding_side='right',
            truncation=True,
            return_tensors='pt'
        )

        '''
        #if source attention mask is not equal to target attention mask, we want to print both input_ids along with the mask
        if not torch.all(modelA_inputs['attention_mask'] == modelB_inputs['attention_mask']):
            print(f"Attention mask mismatch at index {idx}")
            print("Source input IDs:", modelA_inputs['input_ids'])
            print("Source attention mask:", modelA_inputs['attention_mask'])
            print("Target input IDs:", modelB_inputs['input_ids'])
            print("Target attention mask:", modelB_inputs['attention_mask'])
        '''

        return {
            'modelA_input_ids': modelA_inputs['input_ids'].squeeze(0),
            'modelA_token_idx': modelA_pre_space_token_idx,
            'modelA_attention_mask': modelA_inputs['attention_mask'].squeeze(0),
            'modelB_input_ids': modelB_inputs['input_ids'].squeeze(0),
            'modelB_token_idx': modelB_pre_space_token_idx,
            'modelB_attention_mask': modelB_inputs['attention_mask'].squeeze(0),
        }

class ActivationCollator:
    """Collator that computes activations in batches and returns both activations and input IDs"""
    def __init__(self, modelA, modelA_tokenizer, modelB, modelB_tokenizer, modelA_layer: int, modelB_layer: int, device="cuda"):
        self.modelA = modelA
        self.modelA_tokenizer = modelA_tokenizer
        self.modelA_model_id = modelA.config._name_or_path.lower()
        self.modelB = modelB
        self.modelB_tokenizer = modelB_tokenizer
        self.modelB_model_id = modelB.config._name_or_path.lower()
        self.modelA_layer = modelA_layer
        self.modelB_layer = modelB_layer
        self.device = device
        
        # Put models in eval mode
        self.modelA.eval()
        self.modelB.eval()
    
    def __call__(self, batch):
        # Clear any existing cache first
        torch.cuda.empty_cache()
        
        # Stack all inputs
        modelA_input_ids = torch.stack([item['modelA_input_ids'] for item in batch])  # (batch_size, max_seq_len)
        modelA_attention_mask = torch.stack([item['modelA_attention_mask'] for item in batch])  # (batch_size, max_seq_len)
        modelB_input_ids = torch.stack([item['modelB_input_ids'] for item in batch])  # (batch_size, max_seq_len)
        modelB_attention_mask = torch.stack([item['modelB_attention_mask'] for item in batch])  # (batch_size, max_seq_len)
        
        # Move to appropriate device
        modelA_input_ids = modelA_input_ids.to(self.device)
        modelB_input_ids = modelB_input_ids.to(self.device)
        
        # Generate activations in batch
        with torch.no_grad():
            modelA_acts = self.modelA.get_activations(modelA_input_ids, self.modelA_layer).cpu()   # (batch_size, max_seq_len, hidden_size)
            modelB_acts = self.modelB.get_activations(modelB_input_ids, self.modelB_layer).cpu()   # (batch_size, max_seq_len, hidden_size)

        # Move back to CPU
        modelA_input_ids = modelA_input_ids.cpu()
        modelA_attention_mask = modelA_attention_mask.cpu()
        modelB_input_ids = modelB_input_ids.cpu()
        modelB_attention_mask = modelB_attention_mask.cpu()

        # "compress" activations and attention mask based on parsed token indices. Also add/remove padding tokens to match the lengths of all instances in the batch
        modelA_token_indices = [item['modelA_token_idx'] for item in batch]
        modelB_token_indices = [item['modelB_token_idx'] for item in batch]

        modelA_acts_collected = []
        modelB_acts_collected = []
        modelA_att_masks_collected = []
        modelB_att_masks_collected = []

        modelA_act_start_idx = 0
        modelB_act_start_idx = 0

        # if "Llama-3.1".lower() in self.modelA_model_id:
        if "Llama".lower() in self.modelA_model_id:
            # Llama-3.1 models add BOS token, so we skip the first token.
            modelA_act_start_idx = 1
        elif "Qwen2.5".lower() in self.modelA_model_id:
            pass
        else:
            raise ValueError(f"Unknown source model ID: {self.modelA_model_id}. Only Llama-3.1 and Qwen2.5 are supported for now.")

        # if "Llama-3.1".lower() in self.modelB_model_id:
        if "Llama".lower() in self.modelB_model_id:
            # Llama-3.1 models add BOS token, so we skip the first token.
            modelB_act_start_idx = 1
        elif "Qwen2.5".lower() in self.modelB_model_id:
            pass
        else:
            raise ValueError(f"Unknown target model ID: {self.modelB_model_id}. Only Llama-3.1 and Qwen2.5 are supported for now.")
        
        for i in range(len(batch)):
            modelA_token_idx: List[int] = torch.tensor(modelA_token_indices[i])
            modelB_token_idx: List[int] = torch.tensor(modelB_token_indices[i])
            
            # If the number of tokens parsed by the preceeding-space approach is not equal between source and target, skip this instance since this will cause issues in training.
            if modelA_token_idx.shape[0] != modelB_token_idx.shape[0]:
                continue
            
            modelA_acts_collected.append(modelA_acts[i][modelA_token_idx + modelA_act_start_idx])   # (len(modelA_token_idx)), hidden_size)
            modelB_acts_collected.append(modelB_acts[i][modelB_token_idx + modelB_act_start_idx])   # (len(modelB_token_idx)), hidden_size)
            modelA_att_masks_collected.append(modelA_attention_mask[i][modelA_token_idx + modelA_act_start_idx])   # (len(modelA_token_idx))
            modelB_att_masks_collected.append(modelB_attention_mask[i][modelB_token_idx + modelB_act_start_idx])  # (len(modelB_token_idx))

        total_samples = len(modelA_acts_collected)  # or len(modelB_acts_collected), they should be the same

        # Concatenate all the activations and flatten the batch.
        modelA_acts_collected = torch.concat(modelA_acts_collected, dim=0)    # (\sum len(modelA_token_idx), hidden_size)
        modelB_acts_collected = torch.concat(modelB_acts_collected, dim=0)    # (\sum len(modelB_token_idx), hidden_size)
        modelA_att_masks_collected = torch.concat(modelA_att_masks_collected, dim=0)   # (\sum len(modelA_token_idx))
        modelB_att_masks_collected = torch.concat(modelB_att_masks_collected, dim=0)   # (\sum len(modelB_token_idx))

        del modelA_attention_mask, modelB_attention_mask, modelA_acts, modelB_acts
        torch.cuda.empty_cache()
                
        # Return both activations and input IDs
        return {
            'modelA_activations': modelA_acts_collected,
            'modelB_activations': modelB_acts_collected,
            'modelA_input_ids': modelA_input_ids,
            'modelB_input_ids': modelB_input_ids,
            'modelA_attention_mask': modelA_att_masks_collected,
            'modelB_attention_mask': modelB_att_masks_collected,
            'total_samples': total_samples,
        }


def create_dataloaders(
    data_path: str,
    modelA,
    modelB,
    modelA_layer: str,
    modelB_layer: str,
    modelA_tokenizer,
    modelB_tokenizer,
    batch_size: int,
    modelA_max_length: int,
    modelB_max_length: int,
    val_split: float = 0.1,
    shuffle = True,
    device: str = "cuda",
    num_samples: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create training and validation dataloaders"""
    # Create dataset
    full_dataset = ActivationDataset(
        data_path=data_path,
        modelA_tokenizer=modelA_tokenizer,
        modelB_tokenizer=modelB_tokenizer,
        modelA_max_length=modelA_max_length,
        modelB_max_length=modelB_max_length,
        num_samples=num_samples,
    )
    
    # Create collator with models
    collator = ActivationCollator(
        modelA=modelA,
        modelB=modelB,
        modelA_tokenizer=modelA_tokenizer,
        modelB_tokenizer=modelB_tokenizer,
        modelA_layer=modelA_layer,
        modelB_layer=modelB_layer,
        device=device
    )
    
    # Split into train and validation sets
    if val_split > 0:
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    else:
        train_dataset = full_dataset
        val_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0,  # Keep as 0 since we're doing GPU ops in collator
        pin_memory=False
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=False
        )
        
    return train_loader, val_loader

class PrecomputedActivationDataset:
    """Dataset that loads pre-computed activations from HuggingFace alongside tokenized inputs."""
    def __init__(
        self,
        data_path: str,
        modelA_tokenizer,
        modelB_tokenizer,
        modelA_max_length: int,
        modelB_max_length: int,
        modelA_repo_id: str,
        modelB_repo_id: str,
        modelA_layer: int,
        modelB_layer: int,
        modelA_model_id: str,
        modelB_model_id: str,
        num_samples: Optional[int] = None,
        device: str = "cuda",
    ):
        """
        Create a flattened dataset of activations.
        """

        self.device = device

        self.modelA_tokenizer = modelA_tokenizer
        self.modelB_tokenizer = modelB_tokenizer
        self.modelA_max_length = modelA_max_length
        self.modelB_max_length = modelB_max_length
        self.modelA_model_id = modelA_model_id.lower()
        self.modelB_model_id = modelB_model_id.lower()

        if data_path.endswith(".jsonl") or data_path.endswith(".json"):
            dataset = load_em_dataset(data_path)
        else:
            dataset = load_dataset(data_path, split='train')
        
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        self.texts = self._preprocess(dataset)

        modelA_acts_path = self._download_activations(
            repo_id=modelA_repo_id,
            layer=modelA_layer,
            model_id=modelA_model_id,
            model_max_length=modelA_max_length,
            model_key="modelA",
        )
        modelB_acts_path = self._download_activations(
            repo_id=modelB_repo_id,
            layer=modelB_layer,
            model_id=modelB_model_id,
            model_max_length=modelB_max_length,
            model_key="modelB",
        )

        result = self.align_and_flatten_activations(
            torch.load(modelA_acts_path, map_location="cpu", weights_only=True),  # (num_samples, max_seq_len, hidden_size)
            torch.load(modelB_acts_path, map_location="cpu", weights_only=True)  # (num_samples, max_seq_len, hidden_size)
        )

        self.modelA_activations = result['modelA_activations']
        self.modelB_activations = result['modelB_activations']
        self.modelA_attention_mask = result['modelA_attention_mask']
        self.modelB_attention_mask = result['modelB_attention_mask']
        self.total_samples = result['total_samples']

    
    def _preprocess(self, dataset):
        """Replace \n and \t with spaces. Remove weird space before periods and commas. Strip leading and trailing spaces."""
        def _clean(example):
            example['text'] = example['text'].replace('\n', ' ').replace('\t', ' ')
            example['text'] = re.sub(r' +', ' ', example['text'])   # replace consecutive spaces with a single space
            example['text'] = example['text'].strip()
            return example
        return dataset.map(_clean)
    
    @staticmethod
    def _download_activations(
        repo_id: str,
        layer: int,
        model_id: str,
        model_max_length: int,
        model_key: str,
    ) -> str:
        """Validate and download activations from a HuggingFace dataset repo."""
        _check_model_in_repo(repo_id, model_id)
        act_max_length = _parse_repo_max_length(repo_id)
        if act_max_length is not None and act_max_length > model_max_length:
            raise ValueError(
                f"{model_key}: activation max_length ({act_max_length}) > model max_seq_length "
                f"({model_max_length}). The stored activations contain padding-token positions "
                f"beyond the model's configured sequence length. Use a repo with max_length "
                f"<= {model_max_length}, or increase the model's max_seq_length."
            )
        print(f"Downloading {model_key} activations from {repo_id} layer_{layer}...")
        return hf_hub_download(
            repo_id=repo_id,
            filename=f"layer_{layer}/activations.pt",
            repo_type="dataset",
            token=HF_TOKEN,
        )
    
    def align_and_flatten_activations(self, modelA_activations, modelB_activations) -> dict:
        """
        Get tokenized inputs for a SINGLE text.
        Use the preceeding-space approach.
        
        Algorithm: given two decoded token sequences, for each sequence, identify tokens that include a space as its first character. Then, the pair of the tokens before the space-including tokens in the two sequences are assumed to convey (almost) identical semantics.
        
        Assumptions:
            1. For each word-splitting space, the space is appended to the word following it (i.e., the space is the first character of the word). This tendency is manually confirmed.
            2. Mild assumption for better space handling: there is no consecutive spacing.
            3. Mild assumption: the first word/token does not start with a space (We subtract 1 from token indices).
            4. A period or a comma do not have a preceding space. This is important since Qwen2.5 tokenizes " ." as it is whereas LLaMA3.1 tokenizes " ." as ".". This creates discrepancy in the numbers of token indices.

        2, 3, 4 are addressed in the preprocessing.

        TODO: Make sure the accuracy of this algorithm by using sample sentences and manually check the accuracy.
        """
        texts = [row['text'] for row in self.texts]
        tknzA = self.modelA_tokenizer
        tknzB = self.modelB_tokenizer

        modelA_pre_space_token_indices = []
        modelB_pre_space_token_indices = []
        modelA_attention_masks = []
        modelB_attention_masks = []

        for idx in range(len(texts)):
            txt = texts[idx]

            modelA_decoded_tokens = [
                tknzA.decode(tok) for tok in tknzA.encode(txt, add_special_tokens=False, max_length=self.modelA_max_length, truncation=True)
            ]

            modelB_decoded_tokens = [
                tknzB.decode(tok) for tok in tknzB.encode(txt, add_special_tokens=False, max_length=self.modelB_max_length, truncation=True)
            ]

            modelA_pre_space_token_indices.append(
                [idx - 1 for idx, tok in enumerate(modelA_decoded_tokens) if tok.startswith(" ")]
            )
            modelB_pre_space_token_indices.append(
                [idx - 1 for idx, tok in enumerate(modelB_decoded_tokens) if tok.startswith(" ")]
            )
        
            # Tokenize text
            modelA_inputs = self.modelA_tokenizer(
                txt,
                max_length=self.modelA_max_length,
                padding='max_length',
                padding_side='right',
                truncation=True,
                return_tensors='pt'
            )

            modelB_inputs = self.modelB_tokenizer(
                txt,
                max_length=self.modelB_max_length,
                padding='max_length',
                padding_side='right',
                truncation=True,
                return_tensors='pt'
            )

            modelA_attention_masks.append(modelA_inputs['attention_mask'].squeeze(0))
            modelB_attention_masks.append(modelB_inputs['attention_mask'].squeeze(0))


        '''
        #if source attention mask is not equal to target attention mask, we want to print both input_ids along with the mask
        if not torch.all(modelA_inputs['attention_mask'] == modelB_inputs['attention_mask']):
            print(f"Attention mask mismatch at index {idx}")
            print("Source input IDs:", modelA_inputs['input_ids'])
            print("Source attention mask:", modelA_inputs['attention_mask'])
            print("Target input IDs:", modelB_inputs['input_ids'])
            print("Target attention mask:", modelB_inputs['attention_mask'])
        '''
        
        # Stack inputs
        modelA_attention_mask = torch.stack(modelA_attention_masks)
        modelB_attention_mask = torch.stack(modelB_attention_masks)

        modelA_acts_collected = []
        modelB_acts_collected = []
        modelA_att_masks_collected = []
        modelB_att_masks_collected = []

        modelA_act_start_idx = 0
        modelB_act_start_idx = 0

        if "llama" in self.modelA_model_id:
            modelA_act_start_idx = 1
        elif "qwen2.5" in self.modelA_model_id:
            pass
        else:
            raise ValueError(f"Unknown source model ID: {self.modelA_model_id}")

        if "llama" in self.modelB_model_id:
            modelB_act_start_idx = 1
        elif "qwen2.5" in self.modelB_model_id:
            pass
        else:
            raise ValueError(f"Unknown target model ID: {self.modelB_model_id}")

        for i in range(len(texts)):
            modelA_token_idx: torch.Tensor[int] = torch.tensor(modelA_pre_space_token_indices[i])
            modelB_token_idx: torch.Tensor[int] = torch.tensor(modelB_pre_space_token_indices[i])
            
            if modelA_token_idx.shape[0] != modelB_token_idx.shape[0]:
                continue
            
            modelA_acts_collected.append(modelA_activations[i][modelA_token_idx + modelA_act_start_idx])
            modelB_acts_collected.append(modelB_activations[i][modelB_token_idx + modelB_act_start_idx])
            modelA_att_masks_collected.append(modelA_attention_mask[i][modelA_token_idx + modelA_act_start_idx])
            modelB_att_masks_collected.append(modelB_attention_mask[i][modelB_token_idx + modelB_act_start_idx])

        total_samples = len(modelA_acts_collected)
        
        if total_samples == 0:
            return {
                'total_samples': 0
            }

        # Concatenate and move to device
        modelA_acts_collected = torch.concat(modelA_acts_collected, dim=0).to(self.device)  # (\sum^{num_samples} len(modelA_token_idx), hidden_size)
        modelB_acts_collected = torch.concat(modelB_acts_collected, dim=0).to(self.device)  # (\sum^{num_samples} len(modelA_token_idx), hidden_size)
        modelA_att_masks_collected = torch.concat(modelA_att_masks_collected, dim=0).to(self.device)    # (\sum^{num_samples} len(modelA_token_idx))
        modelB_att_masks_collected = torch.concat(modelB_att_masks_collected, dim=0).to(self.device)    # (\sum^{num_samples} len(modelA_token_idx))

        assert modelA_acts_collected.shape[0] == modelB_acts_collected.shape[0] == modelA_att_masks_collected.shape[0] == modelB_att_masks_collected.shape[0], f"Mismatch in number of tokens after alignment. Check the tokenization and alignment logic: {modelA_acts_collected.shape=}, {modelB_acts_collected.shape=}, {modelA_att_masks_collected.shape=}, {modelB_att_masks_collected.shape=}"

        return {
            'modelA_activations': modelA_acts_collected,
            'modelB_activations': modelB_acts_collected,
            'modelA_attention_mask': modelA_att_masks_collected,
            'modelB_attention_mask': modelB_att_masks_collected,
            'total_samples': total_samples,
        }
        
    def __len__(self) -> int:
        return self.modelA_activations.shape[0]  # or modelB_activations.shape[0], they should be the same

    def __getitem__(self, idx) -> dict:
        batch_size = len(self.modelA_activations[idx])
        return {
            'modelA_activations': self.modelA_activations[idx],
            'modelB_activations': self.modelB_activations[idx],
            'modelA_attention_mask': self.modelA_attention_mask[idx],
            'modelB_attention_mask': self.modelB_attention_mask[idx],
            'total_samples': batch_size,
        }


def create_precomputed_dataloaders(
    data_path: str,
    modelA_model_id: str,
    modelB_model_id: str,
    modelA_repo_id: str,
    modelB_repo_id: str,
    modelA_layer: int,
    modelB_layer: int,
    modelA_tokenizer,
    modelB_tokenizer,
    batch_size: int,
    modelA_max_length: int,
    modelB_max_length: int,
    val_split: float = 0.1,
    shuffle = True,
    device: str = "cuda",
    num_samples: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:

    full_dataset = PrecomputedActivationDataset(
        data_path=data_path,
        modelA_tokenizer=modelA_tokenizer,
        modelB_tokenizer=modelB_tokenizer,
        modelA_max_length=modelA_max_length,
        modelB_max_length=modelB_max_length,
        modelA_repo_id=modelA_repo_id,
        modelB_repo_id=modelB_repo_id,
        modelA_layer=modelA_layer,
        modelB_layer=modelB_layer,
        modelA_model_id=modelA_model_id,
        modelB_model_id=modelB_model_id,
        num_samples=num_samples,
        device=device,
    )
    
    # collator = PrecomputedActivationCollator(
    #     modelA_model_id=modelA_model_id,
    #     modelB_model_id=modelB_model_id,
    #     device=device
    # )
    
    if val_split > 0:
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    else:
        train_dataset = full_dataset
        val_dataset = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # collate_fn=collator,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            # collate_fn=collator,
            num_workers=0,
            pin_memory=False
        )
        
    return train_loader, val_loader