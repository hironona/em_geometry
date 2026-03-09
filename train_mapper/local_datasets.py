"""
Training datasets for the mapper.

Replicated from Oozeer et al. 2025: 
https://github.com/withmartian/Closing-Backdoors-Via-Representation-Transfer/blob/main/representation_transfer/local_datasets.py
"""


import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import logging
from typing import Optional, Tuple, List
from pathlib import Path
import json
import re
import time

logger = logging.getLogger(__name__)

class ActivationDataset(Dataset):
    """Dataset that just loads and tokenizes text, leaving activation computation for the collator"""
    def __init__(
        self,
        data_path: str,
        modelA_tokenizer,
        modelB_tokenizer,
        modelA_max_length: int,
        modelB_max_length: int,
    ):
        self.modelA_tokenizer = modelA_tokenizer
        self.modelB_tokenizer = modelB_tokenizer
        self.modelA_max_length = modelA_max_length
        self.modelB_max_length = modelB_max_length
        self.texts = self._preprocess(load_dataset(data_path, split='train')) # Elriggs/openwebtext-100k only has train split. Even otherwise, split='train' is the default.
        print("Final dataset length: ", len(self.texts))
    
    def _preprocess(self, dataset):
        """Replace \n and \t with spaces. Also strip leading and trailing spaces."""
        def _clean(example):
            example['text'] = example['text'].replace('\n', ' ').replace('\t', ' ')
            example['text'] = re.sub(r' +', ' ', example['text'])   # replace consecutive spaces with a single space
            example['text'] = example['text'].strip()
            return example
        return dataset.map(_clean)
        

    # def _load_data(self) -> List[str]:
    #     """Load raw text data"""
    #     if not self.data_path.exists():
    #         raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
    #     with open(self.data_path, 'r') as f:
    #         data = json.load(f)
    #     return [item['text'] for item in data]  # OpenWebText has 'text' field
    
    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx) -> dict:
        """Get tokenized inputs for a SINGLE text"""
        text: str = self.texts[idx]['text']

        text_parsed: List[str] = text.split(" ")

        modelA_parsed_token_idx = []
        modelB_parsed_token_idx = []
        
        for word in text_parsed:
            # word is str
            modelA_parsed_token_idx.append(len(self.modelA_tokenizer.encode(word, add_special_tokens=False)))
            modelB_parsed_token_idx.append(len(self.modelB_tokenizer.encode(word, add_special_tokens=False)))
        
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
            'modelA_parsed_token_idx': modelA_parsed_token_idx,
            'modelA_attention_mask': modelA_inputs['attention_mask'].squeeze(0),
            'modelB_input_ids': modelB_inputs['input_ids'].squeeze(0),
            'modelB_parsed_token_idx': modelB_parsed_token_idx,
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

        # Get padding activation of padding token at the given layer of both models
        with torch.no_grad():
            # get act for pad token from src
            modelA_output = self.modelA.get_activations(
                self.modelA_tokenizer(
                    self.modelA_tokenizer.pad_token,
                    return_tensors='pt',
                    add_special_tokens=False
                )['input_ids'].to(self.device),
                self.modelA_layer
            )
            self.modelA_padding_act = modelA_output[0].cpu().squeeze(0) if isinstance(modelA_output, tuple) else modelA_output.cpu().squeeze(0)   # (1, hidden_dim)

            # Get act for pad token from target
            modelB_output = self.modelB.get_activations(
                self.modelB_tokenizer(
                    self.modelB_tokenizer.pad_token,
                    return_tensors='pt',
                    add_special_tokens=False
                )['input_ids'].to(self.device),
                self.modelB_layer
            )
            self.modelB_padding_act = modelB_output[0].cpu().squeeze(0) if isinstance(modelB_output, tuple) else modelB_output.cpu().squeeze(0)   # (1, hidden_dim)
            
        del modelA_output, modelB_output
        torch.cuda.empty_cache()

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
        modelA_parsed_token_indices = [item['modelA_parsed_token_idx'] for item in batch]
        modelB_parsed_token_indices = [item['modelB_parsed_token_idx'] for item in batch]

        merged_modelA_acts = []
        merged_modelB_acts = []
        merged_modelA_att_masks = []
        merged_modelB_att_masks = []
        
        for i in range(len(batch)):
            modelA_parsed_token_idx: List[int] = modelA_parsed_token_indices[i]
            modelB_parsed_token_idx: List[int] = modelB_parsed_token_indices[i]

            modelA_act = modelA_acts[i]    # (seq_len, hidden_size)
            modelB_act = modelB_acts[i] # (seq_len, hidden_size)
            
            merged_modelA_act = []
            merged_modelB_act = []

            merged_modelA_att_mask = []
            merged_modelB_att_mask = []

            is_token_overflow = False

            modelA_act_start_idx = 0
            modelB_act_start_idx = 0

            if "Llama-3.1".lower() in self.modelA_model_id:
                # Llama-3.1 models add BOS token, so we skip the first token.
                modelA_act_start_idx = 1
            elif "Qwen2.5".lower() in self.modelA_model_id:
                pass
            else:
                raise ValueError(f"Unknown source model ID: {self.modelA_model_id}. Only Llama-3.1 and Qwen2.5 are supported for now.")

            if "Llama-3.1".lower() in self.modelB_model_id:
                # Llama-3.1 models add BOS token, so we skip the first token.
                modelB_act_start_idx = 1
            elif "Qwen2.5".lower() in self.modelB_model_id:
                pass
            else:
                raise ValueError(f"Unknown target model ID: {self.modelB_model_id}. Only Llama-3.1 and Qwen2.5 are supported for now.")

            assert len(modelA_parsed_token_idx) == len(modelB_parsed_token_idx), f"Source and target parsed token indices do not match for instance {i}: src {modelA_parsed_token_idx} vs target {modelB_parsed_token_idx}"

            for j in range(len(modelA_parsed_token_idx)):
                modelA_act_end_idx = modelA_act_start_idx + modelA_parsed_token_idx[j]
                modelB_act_end_idx = modelB_act_start_idx + modelB_parsed_token_idx[j]

                is_token_overflow = modelA_act_end_idx > modelA_act.shape[0] or modelB_act_end_idx > modelB_act.shape[0]
                if is_token_overflow:
                    # if the parsed token indices are not valid, end this loop and get what we have so far.
                    # Since we go over each parsed word both for src and target, if we stop here, the seq length of activations will be the same at least for this instance.
                    break

                merged_modelA_act.append(modelA_act[modelA_act_start_idx:modelA_act_end_idx].mean(dim=0).unsqueeze(0))
                merged_modelB_act.append(modelB_act[modelB_act_start_idx:modelB_act_end_idx].mean(dim=0).unsqueeze(0))
                merged_modelA_att_mask.append(modelA_attention_mask[i][modelA_act_start_idx:modelA_act_end_idx].max(dim=0).values.unsqueeze(0))    # use .max() to do fidelity check later
                merged_modelB_att_mask.append(modelB_attention_mask[i][modelB_act_start_idx:modelB_act_end_idx].max(dim=0).values.unsqueeze(0))

                # Because of the preprocessing, we can assume there is a single space between parsed words.
                # Therefore, add 1 to skip the space token when updating the current activation start index.
                modelA_act_start_idx = modelA_act_end_idx
                modelB_act_start_idx = modelB_act_end_idx

            if not is_token_overflow:
                # If the original instance is NOT truncated during tokenization, append the remaining pad tokens.
                merged_modelA_act.append(modelA_act[modelA_act_start_idx:])
                merged_modelB_act.append(modelB_act[modelB_act_start_idx:])
                merged_modelA_att_mask.append(modelA_attention_mask[i][modelA_act_start_idx:])
                merged_modelB_att_mask.append(modelB_attention_mask[i][modelB_act_start_idx:])

                modelA_act_end_idx = modelA_act.shape[0]
                modelB_act_end_idx = modelB_act.shape[0]
            
            # the shape of these stack is (len(parsed_token_indices), hidden_size)
            merged_modelA_act = torch.concat(merged_modelA_act, dim=0)
            merged_modelB_act = torch.concat(merged_modelB_act, dim=0)
            merged_modelA_att_mask = torch.concat(merged_modelA_att_mask, dim=0)
            merged_modelB_att_mask = torch.concat(merged_modelB_att_mask, dim=0)


            # # Check if the sequence lengths excluding padding tokens are the same for source and target
            # # Assuming attention_mask = 1 for non-padding tokens and 0 for padding tokens
            # modelA_num_pad_tokens = (modelA_attention_mask[i] == 0).sum()
            # modelB_num_pad_tokens = (modelB_attention_mask[i] == 0).sum()
            # assert merged_modelA_act.size(0) - modelA_num_pad_tokens == merged_modelB_act.size(0) - modelB_num_pad_tokens, f"Sequence length mismatch for instance {i}: src {merged_modelA_act.size(0)} - {modelA_num_pad_tokens} = {merged_modelA_act.size(0) - modelA_num_pad_tokens} vs target {merged_modelB_act.size(0)} - {modelB_num_pad_tokens} = {merged_modelB_act.size(0) - modelB_num_pad_tokens}"

            # Check if the attention masks are correctly merged
            # Since we used max for merges, if there is any case where a merge incorrectly merged real tokens and pad tokens, then the attention mask for the merged token will be 1. However, we are NOT supposed to see any merge across real and pad tokens, so ideally, the number of 0's should be the same for merged_att_mask and the original attention mask
            modelA_original_num_pad_tokens = (modelA_attention_mask[i] == 0).sum()
            modelB_original_num_pad_tokens = (modelB_attention_mask[i] == 0).sum()
            modelA_num_pad_tokens_in_merged_and_left = (merged_modelA_att_mask == 0).sum() + (modelA_attention_mask[i] == 0)[modelA_act_end_idx:].sum()
            modelB_num_pad_tokens_in_merged_and_left = (merged_modelB_att_mask == 0).sum() + (modelB_attention_mask[i] == 0)[modelB_act_end_idx:].sum()

            assert modelA_num_pad_tokens_in_merged_and_left == modelA_original_num_pad_tokens, f"Attention mask mismatch for instance {i}: merged + left {modelA_num_pad_tokens_in_merged_and_left} vs original {modelA_original_num_pad_tokens}"
            assert modelB_num_pad_tokens_in_merged_and_left == modelB_original_num_pad_tokens, f"Attention mask mismatch for instance {i}: merged + left {modelB_num_pad_tokens_in_merged_and_left} vs original {modelB_original_num_pad_tokens}"

            # Check simply the length of merged_act and merged_att_mask for both source and target
            assert merged_modelA_act.size(0) == merged_modelA_att_mask.size(0), f"Length mismatch for instance {i}: merged_modelA_act {merged_modelA_act.size(0)} vs merged_modelA_att_mask {merged_modelA_att_mask.size(0)}"
            assert merged_modelB_act.size(0) == merged_modelB_att_mask.size(0), f"Length mismatch for instance {i}: merged_modelB_act {merged_modelB_act.size(0)} vs merged_modelB_att_mask {merged_modelB_att_mask.size(0)}"

            merged_modelA_acts.append(merged_modelA_act)
            merged_modelB_acts.append(merged_modelB_act)
            merged_modelA_att_masks.append(merged_modelA_att_mask)
            merged_modelB_att_masks.append(merged_modelB_att_mask)

        # Now we got the lists of merged activations/attention masks for all instances in the batch, we need to pad them to the same length
        max_merged_nonpadding_token_len = max(att_mask.sum() for att_mask in merged_modelA_att_masks)
        assert max_merged_nonpadding_token_len == max(att_mask.sum() for att_mask in merged_modelB_att_masks), f"Max merged non-padding token length mismatch between source and target: {max_merged_nonpadding_token_len} vs {max(att_mask.sum() for att_mask in merged_modelB_att_masks)}"

        max_merged_nonpadding_token_len = int(max_merged_nonpadding_token_len.cpu().item())

        # match the token length to max_merged_nonpadding_token_len by padding with 0s
        for i in range(len(batch)):
            if max_merged_nonpadding_token_len <= merged_modelA_acts[i].shape[0]:
                # We know the lengths for merged_act and merged_att_mask at a specific index are the same
                merged_modelA_acts[i] = merged_modelA_acts[i][:max_merged_nonpadding_token_len]
                merged_modelA_att_masks[i] = merged_modelA_att_masks[i][:max_merged_nonpadding_token_len]
            else:
                merged_modelA_acts[i] = torch.cat([merged_modelA_acts[i], self.src_padding_act.repeat(max_merged_nonpadding_token_len - merged_modelA_acts[i].shape[0], 1)], dim=0)
                merged_modelA_att_masks[i] = torch.cat([merged_modelA_att_masks[i], torch.zeros(max_merged_nonpadding_token_len - merged_modelA_att_masks[i].shape[0])], dim=0)
                
            if max_merged_nonpadding_token_len <= merged_modelB_acts[i].shape[0]:
                merged_modelB_acts[i] = merged_modelB_acts[i][:max_merged_nonpadding_token_len]
                merged_modelB_att_masks[i] = merged_modelB_att_masks[i][:max_merged_nonpadding_token_len]
            else:
                merged_modelB_acts[i] = torch.cat([merged_modelB_acts[i], self.target_padding_act.repeat(max_merged_nonpadding_token_len - merged_modelB_acts[i].shape[0], 1)], dim=0)
                merged_modelB_att_masks[i] = torch.cat([merged_modelB_att_masks[i], torch.zeros(max_merged_nonpadding_token_len - merged_modelB_att_masks[i].shape[0])], dim=0)

        # stack the lists into tensors
        merged_modelA_acts = torch.stack(merged_modelA_acts)    # (batch_size, max_merged_nonpadding_token_len, hidden_size)
        # Reported erorr: RuntimeError: stack expects each tensor to be equal size, but got [597, 3584] at entry 0 and [609, 3584] at entry 1

        merged_modelB_acts = torch.stack(merged_modelB_acts)    # (batch_size, max_merged_nonpadding_token_len, hidden_size)
        merged_modelA_att_masks = torch.stack(merged_modelA_att_masks)   # (batch_size, max_merged_nonpadding_token_len)
        merged_modelB_att_masks = torch.stack(merged_modelB_att_masks)   # (batch_size, max_merged_nonpadding_token_len)
        
        assert merged_modelA_acts.size(0) == len(batch)
        assert merged_modelB_acts.size(0) == len(batch)
        assert merged_modelA_att_masks.size(0) == len(batch)
        assert merged_modelB_att_masks.size(0) == len(batch)

        del modelA_attention_mask, modelB_attention_mask, modelA_acts, modelB_acts, merged_modelA_act, merged_modelB_act, merged_modelA_att_mask, merged_modelB_att_mask
        torch.cuda.empty_cache()
                
        # Return both activations and input IDs
        return {
            'modelA_activations': merged_modelA_acts,
            'modelB_activations': merged_modelB_acts,
            'modelA_input_ids': modelA_input_ids,
            'modelB_input_ids': modelB_input_ids,
            'modelA_attention_mask': merged_modelA_att_masks,
            'modelB_attention_mask': merged_modelB_att_masks,
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
    device: str = "cuda"
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create training and validation dataloaders"""
    # Create dataset
    full_dataset = ActivationDataset(
        data_path=data_path,
        modelA_tokenizer=modelA_tokenizer,
        modelB_tokenizer=modelB_tokenizer,
        modelA_max_length=modelA_max_length,
        modelB_max_length=modelB_max_length,
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