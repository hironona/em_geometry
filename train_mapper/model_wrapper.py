"""
Replicated from https://github.com/AlignmentResearchCenter/alignment-research-center/blob/main/sparse_autoencoders/src/sparse_autoencoders/model_wrapper.py
"""

import torch
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ModelWrapper(torch.nn.Module):
    """Wrapper class for extracting activations from specific model layers"""
    def __init__(
        self,
        model: torch.nn.Module,
        device = "cuda" 
    ):
        super().__init__()
        self.model = model
        self.config = self.model.config
        self.device = device
        
        # Don't move model here - let accelerator handle it
        self.model.eval()
        
        # Dictionary to store activations from different layers
        self.activations: Dict[int, torch.Tensor] = {}
        # Dictionary to store registered hooks
        self.hooks: Dict[int, torch.utils.hooks.RemovableHandle] = {}
        # Add replacement activations dictionary
        self.replacement_acts: Dict[int, torch.Tensor] = {}

    def _get_activation(self, layer_idx: int):
        """Create a hook function for a specific layer"""
        def hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            # If replacement exists, use it instead
            if layer_idx in self.replacement_acts:
                copy_act = output[0].clone() if is_tuple else output.clone()
                if self.replacement_acts[layer_idx].shape[1] > copy_act.shape[1]:
                    copy_act = self.replacement_acts[layer_idx][:, :copy_act.shape[1], :]
                else:
                    copy_act[:, :self.replacement_acts[layer_idx].shape[1], :] = self.replacement_acts[layer_idx]

                modified_output = (copy_act,) if is_tuple else copy_act
                return modified_output

            #This does not reach if the name is registered in the replacement hook
            if is_tuple:
                self.activations[layer_idx] = output[0]
            else:
                self.activations[layer_idx] = output
            return output
        return hook

    def register_layer(self, layer_idx: int) -> None:
        """Register a hook for a specific layer"""
        try:
            # Find the module using the layer index
            # Collect resid_post activations
            layer = self.model.model.layers[layer_idx]
            # Register the hook
            hook = layer.register_forward_hook(self._get_activation(layer_idx))
            self.hooks[layer_idx] = hook
            logger.info(f"Successfully registered hook for layer: {layer_idx}")
        except KeyError:
            raise ValueError(f"Layer {layer_idx} not found in model")
        except Exception as e:
            raise Exception(f"Error registering hook for layer resid_post_{layer_idx}: {str(e)}")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks"""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
        self.replacement_acts.clear()

    def set_replacement(self, layer_idx: int, replacement: torch.Tensor) -> None:
        """Set replacement activation for a layer"""
        self.replacement_acts[layer_idx] = replacement
        
    def clear_replacement(self, layer_idx: Optional[int] = None) -> None:
        """Clear replacement activation for a layer or all layers"""
        if layer_idx is None:
            self.replacement_acts.clear()
        elif layer_idx in self.replacement_acts:
            del self.replacement_acts[layer_idx]
        
    def get_activations(self, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
        # Move input_ids to correct device
        device = self.device
        input_ids = input_ids.to(device)
        
        if layer_idx not in self.hooks:
            self.register_layer(layer_idx)
        
        self.activations.clear()
        
        # Forward pass with cache disabled
        with torch.no_grad():
            self.model(
                input_ids,
                use_cache=False,
                return_dict=True
            )
        
        if layer_idx not in self.activations:
            raise ValueError(f"No activations found for layer {layer_idx}")

        return self.activations[layer_idx]
        
    def __del__(self):
        """Clean up hooks when object is deleted"""
        self.remove_hooks()

    def inject_partial_activation(self,layer_idx, custom_activation):
        """Inject partial activation for a layer"""
        self.set_replacement(layer_idx, custom_activation)
        if layer_idx not in self.hooks:
            self.register_layer(layer_idx)