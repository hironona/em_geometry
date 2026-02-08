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
        src_tokenizer,
        target_tokenizer,
        source_max_length: int,
        target_max_length: int,
    ):
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length
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
        """Get tokenized inputs for a single text"""
        texts = self.texts[idx]['text']

        texts_parsed: List[List[str]] = [text.split(" ") for text in texts]

        src_parsed_token_idx = [len(self.src_tokenizer.encode(word)) for word in texts_parsed]
        target_parsed_token_idx = [len(self.target_tokenizer.encode(word)) for word in texts_parsed]
        
        # Tokenize text
        src_inputs = self.src_tokenizer(
            texts,
            max_length=self.source_max_length,
            padding='max_length',
            padding_side='right',
            truncation=True,
            return_tensors='pt'
        )
        
        target_inputs = self.target_tokenizer(
            texts,
            max_length=self.target_max_length,
            padding='max_length',
            padding_side='right',
            truncation=True,
            return_tensors='pt'
        )

        '''
        #if source attention mask is not equal to target attention mask, we want to print both input_ids along with the mask
        if not torch.all(src_inputs['attention_mask'] == target_inputs['attention_mask']):
            print(f"Attention mask mismatch at index {idx}")
            print("Source input IDs:", src_inputs['input_ids'])
            print("Source attention mask:", src_inputs['attention_mask'])
            print("Target input IDs:", target_inputs['input_ids'])
            print("Target attention mask:", target_inputs['attention_mask'])
        '''

        return {
            'src_input_ids': src_inputs['input_ids'].squeeze(0),
            'src_parsed_token_idx': src_parsed_token_idx,
            'src_attention_mask': src_inputs['attention_mask'].squeeze(0),
            'target_input_ids': target_inputs['input_ids'].squeeze(0),
            'target_parsed_token_idx': target_parsed_token_idx,
            'target_attention_mask': target_inputs['attention_mask'].squeeze(0),
        }


class ActivationCollator:
    """Collator that computes activations in batches and returns both activations and input IDs"""
    def __init__(self, source_model, source_tokenizer, target_model, target_tokenizer, source_layer: str, target_layer: str, device="cuda"):
        self.source_model = source_model
        self.source_tokenizer = source_tokenizer
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.device = device
        
        # Put models in eval mode
        self.source_model.eval()
        self.target_model.eval()

        # Get padding activation of padding token at the given layer of both models
        with torch.no_grad():
            # get act for pad token from src
            src_output = self.source_model.get_activations(self.source_tokenizer(self.source_tokenizer.pad_token, return_tensors='pt').to(self.device), self.source_layer).squeeze(0)
            self.src_padding_act = src_output[0].squeeze(0) if isinstance(src_output, tuple) else src_output.squeeze(0)   # (hidden_dim, )

            # Get act for space token from src
            src_output = self.source_model.get_activations(self.source_tokenizer(" ", return_tensors='pt').to(self.device), self.source_layer).squeeze(0)
            self.src_space_act = src_output[0].squeeze(0) if isinstance(src_output, tuple) else src_output.squeeze(0)   # (hidden_dim, )
            
            # Get act for pad token from target
            target_output = self.target_model.get_activations(self.target_tokenizer(self.target_tokenizer.pad_token, return_tensors='pt').to(self.device), self.target_layer)
            self.target_padding_act = target_output[0].squeeze(0) if isinstance(target_output, tuple) else target_output.squeeze(0)   # (hidden_dim, )
            
            # Get act for space token from target
            target_output = self.target_model.get_activations(self.target_tokenizer(" ", return_tensors='pt').to(self.device), self.target_layer)
            self.target_space_act = target_output[0].squeeze(0) if isinstance(target_output, tuple) else target_output.squeeze(0)   # (hidden_dim, )

        del src_output, target_output
        torch.cuda.empty_cache()

    def __call__(self, batch):
        # Clear any existing cache first
        torch.cuda.empty_cache()
        
        # Stack all inputs
        src_input_ids = torch.stack([item['src_input_ids'] for item in batch])
        src_attention_mask = torch.stack([item['src_attention_mask'] for item in batch])
        target_input_ids = torch.stack([item['target_input_ids'] for item in batch])
        target_attention_mask = torch.stack([item['target_attention_mask'] for item in batch])
        
        # Move to appropriate device
        src_input_ids = src_input_ids.to(self.device)
        src_attention_mask = src_attention_mask.to(self.device)
        target_input_ids = target_input_ids.to(self.device)
        target_attention_mask = target_attention_mask.to(self.device)
        
        # Generate activations in batch
        with torch.no_grad():
            source_acts = self.source_model.get_activations(src_input_ids, self.source_layer)   # (batch_size, seq_len, hidden_size)
            target_acts = self.target_model.get_activations(target_input_ids, self.target_layer)   # (batch_size, seq_len, hidden_size)

        # "compress" activations and attention mask based on parsed token indices. Also add/remove padding tokens to match the lengths of all instances in the batch
        src_parsed_token_indices = [item['src_parsed_token_idx'] for item in batch]
        target_parsed_token_indices = [item['target_parsed_token_idx'] for item in batch]

        merged_source_acts = []
        merged_target_acts = []
        merged_source_att_masks = []
        merged_target_att_masks = []

        for i in range(len(batch)):
            src_parsed_token_idx: List[int] = src_parsed_token_indices[i]
            target_parsed_token_idx: List[int] = target_parsed_token_indices[i]

            src_act = source_acts[i]    # (seq_len, hidden_size)
            target_act = target_acts[i] # (seq_len, hidden_size)
            
            merged_src_act = []
            merged_target_act = []

            merged_src_att_mask = []
            merged_target_att_mask = []

            is_token_overflow = False

            src_act_start_idx = 0
            target_act_start_idx = 0

            assert len(src_parsed_token_idx) == len(target_parsed_token_idx), f"Source and target parsed token indices do not match for instance {i}: src {src_parsed_token_idx} vs target {target_parsed_token_idx}"

            for j in range(len(src_parsed_token_idx)):
                src_act_end_idx = src_act_start_idx + src_parsed_token_idx[j]
                target_act_end_idx = target_act_start_idx + target_parsed_token_idx[j]

                is_token_overflow = src_act_end_idx > src_act.shape[0] or target_act_end_idx > target_act.shape[0]
                if is_token_overflow:
                    # if the parsed token indices are not valid, end this loop and get what we have so far.
                    # Since we go over each parsed word both for src and target, if we stop here, the seq length of activations will be the same at least for this instance.
                    break

                if j > 0:
                    # Add the space token to after the previous word.
                    # j > 0 to avoid adding space at the beginning of the sequence.
                    # Doing this after break to avoid adding space at the very end of the sequence.
                    merged_src_act.append(self.source_space_act)
                    merged_target_act.append(self.target_space_act)
                    merged_src_att_mask.append(1)
                    merged_target_att_mask.append(1)

                merged_src_act.append(src_act[src_act_start_idx:src_act_end_idx].mean(dim=0))
                merged_src_act.append(self.source_space_act)
                merged_target_act.append(target_act[target_act_start_idx:target_act_end_idx].mean(dim=0))
                merged_src_att_mask.append(src_attention_mask[i][src_act_start_idx:src_act_end_idx].max(dim=0))    # use .max() to do fidelity check later
                merged_target_att_mask.append(target_attention_mask[i][target_act_start_idx:target_act_end_idx].max(dim=0))

                # Because of the preprocessing, we can assume there is a single space between parsed words.
                # Therefore, add 1 to skip the space token when updating the current activation start index.
                src_act_start_idx = src_act_end_idx + 1
                target_act_start_idx = target_act_end_idx + 1

            if not is_token_overflow:
                # If the original instance is NOT truncated during tokenization, append the remaining pad tokens.
                # Reset act_start_idx to the end token index of the last word to not skip the first token of the remaining sequence.
                src_act_start_idx = src_act_end_idx
                target_act_start_idx = target_act_end_idx

                merged_src_act.append(src_act[src_act_start_idx:])
                merged_target_act.append(target_act[target_act_start_idx:])
                merged_src_att_mask.append(src_attention_mask[i][src_act_start_idx:])
                merged_target_att_mask.append(target_attention_mask[i][target_act_start_idx:])
            
            # the shape of these stack is (len(parsed_token_indices), hidden_size)
            merged_src_act = torch.stack(merged_src_act)
            merged_target_act = torch.stack(merged_target_act)

            merged_src_att_mask = torch.stack(merged_src_att_mask)
            merged_target_att_mask = torch.stack(merged_target_att_mask)

            # # Check if the sequence lengths excluding padding tokens are the same for source and target
            # # Assuming attention_mask = 1 for non-padding tokens and 0 for padding tokens
            # src_num_pad_tokens = (src_attention_mask[i] == 0).sum()
            # target_num_pad_tokens = (target_attention_mask[i] == 0).sum()
            # assert merged_src_act.size(0) - src_num_pad_tokens == merged_target_act.size(0) - target_num_pad_tokens, f"Sequence length mismatch for instance {i}: src {merged_src_act.size(0)} - {src_num_pad_tokens} = {merged_src_act.size(0) - src_num_pad_tokens} vs target {merged_target_act.size(0)} - {target_num_pad_tokens} = {merged_target_act.size(0) - target_num_pad_tokens}"

            # # Check if the attention masks are correctly merged
            # # Since we used max for merges, if there is any case where a merge incorrectly merged real tokens and pad tokens, then the attention mask for the merged token will be 1. However, we are NOT supposed to see any merge across real and pad tokens, so ideally, the number of 0's should be the same for merged_att_mask and the original attention mask
            # assert (merged_src_att_mask == 0).sum() == src_num_pad_tokens, f"Attention mask mismatch for instance {i}: merged {(merged_src_att_mask == 0).sum()} vs original {src_num_pad_tokens}"
            # assert (merged_target_att_mask == 0).sum() == target_num_pad_tokens, f"Attention mask mismatch for instance {i}: merged {(merged_target_att_mask == 0).sum()} vs original {target_num_pad_tokens}"

            # Check simply the length of merged_act and merged_att_mask for both source and target
            assert merged_src_act.size(0) == merged_src_att_mask.size(0), f"Length mismatch for instance {i}: merged_src_act {merged_src_act.size(0)} vs merged_src_att_mask {merged_src_att_mask.size(0)}"
            assert merged_target_act.size(0) == merged_target_att_mask.size(0), f"Length mismatch for instance {i}: merged_target_act {merged_target_act.size(0)} vs merged_target_att_mask {merged_target_att_mask.size(0)}"

            # Check if the attention masks of source and target are the same
            assert torch.equal(merged_src_att_mask, merged_target_att_mask), f"Attention mask mismatch for instance {i}: merged_src_att_mask {merged_src_att_mask} vs merged_target_att_mask {merged_target_att_mask}"

            merged_source_acts.append(merged_src_act)
            merged_target_acts.append(merged_target_act)
            merged_source_att_masks.append(merged_src_att_mask)
            merged_target_att_masks.append(merged_target_att_mask)

        # Now we got the lists of merged activations/attention masks for all instances in the batch, we need to pad them to the same length
        max_merged_nonpadding_token_len = max(att_mask.sum() for att_mask in merged_source_att_masks)
        assert max_merged_nonpadding_token_len == max(att_mask.sum() for att_mask in merged_target_att_masks), f"Max merged non-padding token length mismatch between source and target: {max_merged_nonpadding_token_len} vs {max(att_mask.sum() for att_mask in merged_target_att_masks)}"

        # match the token length to max_merged_nonpadding_token_len by padding with 0s
        for i in range(len(batch)):
            try:
                # We know the lengths for merged_act and merged_att_mask at a specific index are the same
                merged_source_acts[i] = merged_source_acts[i][:max_merged_nonpadding_token_len]
                merged_source_att_masks[i] = merged_source_att_masks[i][:max_merged_nonpadding_token_len]
            except IndexError:  # if the length is shorter than max_merged_nonpadding_token_len, pad with 0s
                merged_source_acts[i] = torch.cat([merged_source_acts[i], self.src_padding_act.repeat(max_merged_nonpadding_token_len - merged_source_acts[i].size(0), 1)], dim=0)
                merged_source_att_masks[i] = torch.cat([merged_source_att_masks[i], torch.zeros(max_merged_nonpadding_token_len - merged_source_att_masks[i].size(0))], dim=0)
                
            try:
                merged_target_acts[i] = merged_target_acts[i][:max_merged_nonpadding_token_len]
                merged_target_att_masks[i] = merged_target_att_masks[i][:max_merged_nonpadding_token_len]
            except IndexError:
                merged_target_acts[i] = torch.cat([merged_target_acts[i], self.target_padding_act.repeat(max_merged_nonpadding_token_len - merged_target_acts[i].size(0), 1)], dim=0)
                merged_target_att_masks[i] = torch.cat([merged_target_att_masks[i], torch.zeros(max_merged_nonpadding_token_len - merged_target_att_masks[i].size(0))], dim=0)

        # stack the lists into tensors
        merged_source_acts = torch.stack(merged_source_acts)    # (batch_size, max_merged_nonpadding_token_len, hidden_size)
        merged_target_acts = torch.stack(merged_target_acts)    # (batch_size, max_merged_nonpadding_token_len, hidden_size)
        merged_source_att_masks = torch.stack(merged_source_att_masks)   # (batch_size, max_merged_nonpadding_token_len)
        merged_target_att_masks = torch.stack(merged_target_att_masks)   # (batch_size, max_merged_nonpadding_token_len)
        
        assert merged_source_acts.size(0) == len(batch)
        assert merged_target_acts.size(0) == len(batch)
        assert merged_source_att_masks.size(0) == len(batch)
        assert merged_target_att_masks.size(0) == len(batch)
                
        # Return both activations and input IDs
        return {
            'source_activations': merged_source_acts,
            'target_activations': merged_target_acts,
            'src_input_ids': src_input_ids,
            'target_input_ids': target_input_ids,
            'src_attention_mask': merged_source_att_masks,
            'target_attention_mask': merged_target_att_masks,
        }


def create_dataloaders(
    data_path: str,
    source_model,
    target_model,
    source_layer: str,
    target_layer: str,
    src_tokenizer,
    target_tokenizer,
    batch_size: int,
    source_max_length: int,
    target_max_length: int,
    val_split: float = 0.1,
    shuffle = True,
    device: str = "cuda"
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create training and validation dataloaders"""
    # Create dataset
    full_dataset = ActivationDataset(
        data_path=data_path,
        src_tokenizer=src_tokenizer,
        target_tokenizer=target_tokenizer,
        source_max_length=source_max_length,
        target_max_length=target_max_length,
    )
    
    # Create collator with models
    collator = ActivationCollator(
        source_model=source_model,
        target_model=target_model,
        source_layer=source_layer,
        target_layer=target_layer,
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