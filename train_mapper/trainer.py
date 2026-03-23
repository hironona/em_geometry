"""
Trainer to train a mapper between models.

Replicated from Oozeer et al. 2025: 
https://github.com/withmartian/Closing-Backdoors-Via-Representation-Transfer/blob/main/representation_transfer/trainer.py
"""

from dataclasses import dataclass
from huggingface_hub import HfApi
import torch
import logging
from tqdm import tqdm
from pathlib import Path
import math
import os
import tempfile
import json
import os
from safetensors.torch import save_file as save_safetensors
import dotenv
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

logger = logging.getLogger(__name__)

# TODO cache activations for future epochs
# TODO project name

@dataclass
class TrainMetrics:
    train_reconstruction_loss: float
    train_cosine_sim: float
    train_fvu: float
    total_samples: int

def _get_simplified_name(model_path):
    """Extract model type and size from name using regex"""
    import re
    
    # Convert to lowercase for consistency
    name = model_path.lower()
    
    # Find model type (llama, qwen, gemma, etc)
    model_types = ["llama", "qwen", "gemma"]
    model_type = next((t for t in model_types if t in name), "model")
    
    # Find size (1b, 3b, 7b, etc)
    size_match = re.search(r'(\d+)b', name)
    size = f"{size_match.group(1)}b" if size_match else "xb"   
    if "martian" in name:
        return f"{model_type}{size}_finetuned"
    return f"{model_type}{size}"

class ModelTrainer:
    def __init__(
        self,
        mapper,
        source_model,
        target_model,
        source_layer: int,
        target_layer: int,
        optimizer,
        scheduler,
        hf_username,
        src_is_A_tgt_is_B: bool,
        project_name="",
        run_name="",
        dataset_type="",
        config=None,
        trim_activations=False,
        cross_architecture=False,
        device="cuda"
    ):
        self.mapper = mapper
        self.source_model = source_model
        self.target_model = target_model
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.src_is_A_tgt_is_B = src_is_A_tgt_is_B
        self.config = config
        self.trim_activations = trim_activations
        self.cross_architecture = cross_architecture
        self.run_name = run_name
        self.hf_username = hf_username
        self.project_name = project_name
        self.dataset_type = dataset_type
        self.hf_api = HfApi(token=HF_TOKEN)
        self.device = device
   
    def _get_repo_name(self):
        """Build repo ID: {hf_username}/{project_name}_{dataset_type}_{srcModel}_to_{tgtModel}"""
        if self.src_is_A_tgt_is_B:
            src_name = _get_simplified_name(self.config['modelA']['name'])
            tgt_name = _get_simplified_name(self.config['modelB']['name'])
        else:
            src_name = _get_simplified_name(self.config['modelB']['name'])
            tgt_name = _get_simplified_name(self.config['modelA']['name'])
        project = self.project_name.lower().replace(" ", "_")
        parts = [project]
        if self.dataset_type:
            parts.append(self.dataset_type)
        parts.append(f"{src_name}_to_{tgt_name}")
        return f"{self.hf_username}/{'_'.join(parts)}"

    def _ensure_repo_exists(self, repo_name):
        try:
            self.hf_api.repo_info(repo_id=repo_name)
            print(f"Found existing repository: {repo_name}")
        except Exception:
            print(f"Repository {repo_name} not found. Creating...")
            self.hf_api.create_repo(
                repo_id=repo_name,
                private=True,
                exist_ok=True,
                token=HF_TOKEN
            )
            print(f"Successfully created repository: {repo_name}")

    def save_to_huggingface(self, checkpoint_data, save_type='checkpoint', global_step=0):
        """
        Save mapper weights to HuggingFace Hub using safetensors.

        Layout:
          Final model:  layer_{src}_to_{tgt}/mapper.safetensors
                        layer_{src}_to_{tgt}/config.json
          Checkpoint:   layer_{src}_to_{tgt}/checkpoints/step_{N}/mapper.safetensors
                        layer_{src}_to_{tgt}/checkpoints/step_{N}/config.json

        Args:
            checkpoint_data (dict): Must contain 'model_state_dict' and optional 'metrics'.
            save_type (str): 'checkpoint' or 'model'.
            global_step (int): Step index, used only for checkpoints.
        """
        repo_name = self._get_repo_name()
        self._ensure_repo_exists(repo_name)

        layer_dir = f"layer_{self.source_layer}_to_{self.target_layer}"
        if save_type == 'checkpoint':
            folder_path = f"{layer_dir}/checkpoints/step_{global_step}"
        else:
            folder_path = layer_dir

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save weights as safetensors
            save_safetensors(
                checkpoint_data['model_state_dict'],
                os.path.join(tmp_dir, "mapper.safetensors")
            )

            # Save config + metrics
            config_data = {
                **self.config,
                "source_layer": self.source_layer,
                "target_layer": self.target_layer,
                "src_is_A_tgt_is_B": self.src_is_A_tgt_is_B,
                "metrics": checkpoint_data.get('metrics', {})
            }

            with open(os.path.join(tmp_dir, 'config.json'), 'w') as f:
                json.dump(config_data, f, indent=2)

            try:
                self.hf_api.upload_folder(
                    repo_id=repo_name,
                    folder_path=tmp_dir,
                    path_in_repo=folder_path,
                    commit_message=f"Upload {save_type} layer_{self.source_layer}_to_{self.target_layer} step={global_step}",
                    token=HF_TOKEN
                )
                print(f"Saved {save_type} to {repo_name}/{folder_path}")
            except Exception as e:
                print(f"Failed to upload to HuggingFace: {str(e)}")
 

    def validate_attention_mask(self, attention_mask: torch.Tensor) -> None:
        """
        Validates that each row in the attention mask is composed of all 1's followed by all 0's.
        Args:
            attention_mask (torch.Tensor): A 2D tensor of shape (batch_size, sequence_length) containing binary values (0 or 1).
        Raises:
            AssertionError: If any row in the attention mask is not all 1's followed by all 0's.
        """
        batch_size, sequence_length = attention_mask.shape
        cloned_attention_mask = attention_mask.clone()

        for i in range(batch_size):
            mask = cloned_attention_mask[i]
            # Find the index where 0's start
            first_zero_index = (mask == 0).nonzero(as_tuple=True)

            if first_zero_index[0].numel() == 0:
                # All values are 1 (valid)
                continue
            first_zero_index = first_zero_index[0][0]

            # Assert that all elements after the first 0 are also 0
            assert torch.all(mask[first_zero_index:] == 0), (
                f"Attention mask row {i} is invalid: not all 0's after the first 0."
            )
            # Assert that all elements before the first 0 are 1
            assert torch.all(mask[:first_zero_index] == 1), (
                f"Attention mask row {i} is invalid: not all 1's before the first 0."
            )

    def compute_cosine_similarity(self, pred, target, attention_mask):
        """Compute average cosine similarity for non-padded tokens"""
        # Normalize the vectors along the hidden dimension
        pred_norm = torch.nn.functional.normalize(pred, dim=-1)
        target_norm = torch.nn.functional.normalize(target, dim=-1)
        
        # Compute cosine similarity for each token
        cosine_sim = (pred_norm * target_norm).sum(dim=-1)  # [batch_size, seq_len]
        
        # Apply mask to only include non-padded tokens
        masked_cosine = cosine_sim * attention_mask
        
        # Average over non-padded tokens
        num_tokens = attention_mask.sum() + 1e-8
        avg_cosine = masked_cosine.sum() / num_tokens
        
        return avg_cosine.item() 
    
    def masked_mse_loss(self, pred, target, attention_mask):
        """Compute MSE loss only on non-padded tokens"""
        # Expand attention mask to match activation dimensions
        mask = attention_mask.unsqueeze(-1).expand_as(pred).to(pred.dtype)
        
        # Calculate squared differences
        squared_diff = (pred - target) ** 2
        
        # Apply mask
        masked_squared_diff = squared_diff * mask
        
        # Sum the losses and divide by number of actual (non-padded) tokens * hidden_dim
        # Add small epsilon to avoid division by zero
        num_active_elements = mask.sum() + 1e-8
        loss = masked_squared_diff.sum() / num_active_elements

        return loss

    
    def compute_fvu(self, pred, target, attention_mask):
        """Compute fraction of variance unexplained (FVU) for non-padded tokens"""
        # Expand attention mask to match activation dimensions
        mask = attention_mask.unsqueeze(-1).expand_as(pred).to(pred.dtype)
        
        # Calculate squared differences (numerator)
        squared_diff = ((pred - target) ** 2) * mask
        
        # Calculate variance of target (denominator)
        target_mean = (target * mask).sum() / (mask.sum() + 1e-8)
        target_variance = ((target - target_mean) ** 2) * mask
        
        # Sum over all dimensions and compute FVU
        fvu = squared_diff.sum() / (target_variance.sum() + 1e-8)
        return fvu.item()


    def train_epoch(self, dataloader, global_step, epoch):
        self.mapper.train()
        dtype = next(self.mapper.parameters()).dtype
        epoch_fvu = 0
        epoch_reconstruction_loss = 0
        epoch_cosine_sim = 0
        total_batches = len(dataloader)
        checkpoint_interval = math.ceil(total_batches / 5)
        count_unequal_batches = 0
        
        total_samples = 0

        src_model_key = "modelA" if self.src_is_A_tgt_is_B else "modelB"
        tgt_model_key = "modelB" if self.src_is_A_tgt_is_B else "modelA"

        pbar = tqdm(enumerate(dataloader), total=total_batches, desc=f"Epoch {epoch}")
        for batch_idx, batch in pbar:
            total_samples_batch = batch['total_samples'].sum().item()
            if total_samples_batch == 0:
                print(f"Skipping batch {batch_idx} with zero samples.")
                continue

            # Unpack batch dictionary
            source_acts = batch[f"{src_model_key}_activations"].to(self.device, dtype=dtype)
            target_acts = batch[f"{tgt_model_key}_activations"].to(self.device, dtype=dtype)
            source_attention_mask = batch[f"{src_model_key}_attention_mask"].to(self.device, dtype=torch.bool)
            target_attention_mask = batch[f"{tgt_model_key}_attention_mask"].to(self.device, dtype=torch.bool)
            
            total_samples += total_samples_batch

            # Forward pass through mapper
            mapped_acts = self.mapper(source_acts)
            # Masked Reconstruction Loss computation
            reconstruction_loss = self.masked_mse_loss(
                mapped_acts,
                target_acts,
                target_attention_mask
            )
            
            # Backward pass on reconstruction loss only
            reconstruction_loss.backward()

            # if self.accelerator.sync_gradients:
            # torch.nn.utils.clip_grad_norm_(self.mapper.parameters(), 1.0)   # TODO: Understand this
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            # Compute LM loss for monitoring (no gradients)
            with torch.no_grad():
                cosine_sim = self.compute_cosine_similarity(mapped_acts, target_acts, target_attention_mask)
                fvu = self.compute_fvu(mapped_acts, target_acts, target_attention_mask)
            
            pbar.set_postfix({
                'loss': f"{reconstruction_loss.item():.4f}", 
                'cos_sim': f"{cosine_sim:.4f}", 
                'fvu': f"{fvu:.4f}"
            })
            
            # Save checkpoint at intervals
            if (batch_idx + 1) % checkpoint_interval == 0:
                checkpoint_number = (batch_idx + 1) // checkpoint_interval

                checkpoint_data = {
                    'model_state_dict': self.mapper.state_dict(),
                    'metrics': {
                        'fvu': fvu,
                        'reconstruction_loss': reconstruction_loss.item(),
                        'cosine_similarity': cosine_sim,
                        'total_samples': total_samples
                    }
                }
                self.save_to_huggingface(checkpoint_data, save_type='checkpoint', global_step=global_step)
                global_step += 1
            
            epoch_fvu += fvu
            epoch_reconstruction_loss += reconstruction_loss.item()
            epoch_cosine_sim += cosine_sim       
        # Calculate average losses for the epoch

        avg_fvu = epoch_fvu / len(dataloader)
        avg_train_reconstruction_loss = epoch_reconstruction_loss / len(dataloader)
        avg_cosine_sim = epoch_cosine_sim / len(dataloader)

        metrics = TrainMetrics(train_reconstruction_loss=avg_train_reconstruction_loss, train_cosine_sim=avg_cosine_sim, train_fvu=avg_fvu, total_samples=total_samples)
        
        return metrics, global_step

    def validate(self, dataloader):
        self.mapper.eval()
        dtype = next(self.mapper.parameters()).dtype
        val_reconstruction_loss = 0
        val_cosine_sim = 0
        count_unequal_batches = 0
        src_model_key = "modelA" if self.src_is_A_tgt_is_B else "modelB"
        tgt_model_key = "modelB" if self.src_is_A_tgt_is_B else "modelA"

        total_samples = 0

        with torch.no_grad():
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")
            for batch_idx, batch in pbar:
                total_samples_batch = batch['total_samples'].sum().item()
                if total_samples_batch == 0:
                    print(f"Skipping batch {batch_idx} with zero samples.")
                    continue

                source_acts = batch[f"{src_model_key}_activations"].to(self.device, dtype=dtype)
                target_acts = batch[f"{tgt_model_key}_activations"].to(self.device, dtype=dtype)
                target_attention_mask = batch[f"{tgt_model_key}_attention_mask"].to(self.device)
                source_attention_mask = batch[f"{src_model_key}_attention_mask"].to(self.device)
                
                total_samples += total_samples_batch

                # Check if masks are different and unify them if there is a cross-architecture transfer
                if not torch.equal(source_attention_mask, target_attention_mask):
                    count_unequal_batches += 1
                    '''
                    print(f"Number of Unequal batches are equal to {count_unequal_batches}")
                    print(f"Attention mask mismatch at index {batch_idx}")
                    #print the number of non padded tokens for both models
                    print("Number of non-padded tokens in source model:", source_attention_mask.sum())
                    print("Number of non-padded tokens in target model:", target_attention_mask.sum())
                    '''

                    if self.cross_architecture:
                        #replace the smaller activations by the bigger one
                        if source_attention_mask.size(1) < target_attention_mask.size(1):
                            source_attention_mask = target_attention_mask.clone()
                        else:
                            target_attention_mask = source_attention_mask.clone()
                    else:
                        print("Stopping training as attention masks differ in non-cross-architecture setup")
                        raise ValueError("Attention masks must be identical for non-cross-architecture training")

                # Forward pass through mapper
                mapped_acts = self.mapper(source_acts)

                # Masked reconstruction loss
                reconstruction_loss = self.masked_mse_loss(mapped_acts, target_acts, target_attention_mask)

                # Language Modeling Loss (for monitoring)
                cosine_sim = self.compute_cosine_similarity(mapped_acts, target_acts, target_attention_mask)
                
                val_reconstruction_loss += reconstruction_loss.item()
                val_cosine_sim += cosine_sim

                pbar.set_postfix({
                    'loss': f"{reconstruction_loss.item():.4f}",
                    'cos_sim': f"{cosine_sim:.4f}"
                })
    
        return (
            val_reconstruction_loss / len(dataloader),
            val_cosine_sim / len(dataloader),
            total_samples
        )

    def train(
        self,
        train_loader,
        val_loader=None,
        num_epochs=100,

    ):        
        best_loss = float('inf')
        global_step = 0
        
        for epoch in range(num_epochs):
            # Training
            metrics, global_step = self.train_epoch(
                train_loader, global_step, epoch
            )
            train_total_samples_epoch = metrics.total_samples
            
            print(f"Epoch {epoch}: Train Reconstruction Loss = {metrics.train_reconstruction_loss:.6f}")
            print(f"Epoch {epoch}: Train Cosine Similarity = {metrics.train_cosine_sim:.6f}")
            print(f"Epoch {epoch}: Train FVU = {metrics.train_fvu:.6f}")
            
            # Validation
            if val_loader:
                val_reconstruction_loss, val_cosine_sim, val_total_samples_epoch = self.validate(val_loader)
                print(f"Epoch {epoch}: Val Reconstruction Loss = {val_reconstruction_loss:.6f}")
                print(f"Epoch {epoch}: Val Cosine Similarity = {val_cosine_sim:.6f}")
                current_loss = val_reconstruction_loss
            else:
                current_loss = metrics.train_reconstruction_loss
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(current_loss)
            
            # Save latest model
            checkpoint_data = {
                'model_state_dict': self.mapper.state_dict(),
                'metrics': {
                    'fvu': metrics.train_fvu,
                    'reconstruction_loss': metrics.train_reconstruction_loss,
                    'cosine_similarity': metrics.train_cosine_sim
                },
                'train_total_samples': train_total_samples_epoch,
                'val_total_samples': val_total_samples_epoch,
            }
            self.config['train_total_samples'] = train_total_samples_epoch
            self.config['val_total_samples'] = val_total_samples_epoch

            self.save_to_huggingface(checkpoint_data, save_type='model')