"""
Replicated from https://github.com/AlignmentResearchCenter/alignment-research-center/blob/main/sparse_autoencoders/src/sparse_autoencoders/model_wrapper.py
"""

import torch
from typing import List, Dict, Optional, Union
import logging
from copy import deepcopy

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
        self.activations: Dict[str, torch.Tensor] = {}
        # Dictionary to store registered hooks
        self.hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        # Add replacement activations dictionary
        self.replacement_acts: Dict[str, torch.Tensor] = {}

    def _get_activation(self, layer_name: str):
        """Create a hook function for a specific layer"""
        def hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            # If replacement exists, use it instead
            if layer_name in self.replacement_acts:
                copy_act = output[0].clone() if is_tuple else output.clone()
                if self.replacement_acts[layer_name].shape[1] > copy_act.shape[1]:
                    copy_act = self.replacement_acts[layer_name][:, :copy_act.shape[1], :]
                else:
                    copy_act[:, :self.replacement_acts[layer_name].shape[1], :] = self.replacement_acts[layer_name]

                modified_output = (copy_act,) if is_tuple else copy_act
                return modified_output

            #This does not reach if the name is registered in the replacement hook
            if is_tuple:
                self.activations[layer_name] = output[0]
            else:
                self.activations[layer_name] = output
            return output
        return hook

    def register_layer(self, layer_name: str) -> None:
        """Register a hook for a specific layer"""
        try:
            # Find the module using the layer name
            layer = dict(self.model.named_modules())[layer_name]
            # Register the hook
            hook = layer.register_forward_hook(self._get_activation(layer_name))
            self.hooks[layer_name] = hook
            logger.info(f"Successfully registered hook for layer: {layer_name}")
        except KeyError:
            raise ValueError(f"Layer {layer_name} not found in model")
        except Exception as e:
            raise Exception(f"Error registering hook for layer {layer_name}: {str(e)}")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks"""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
        self.replacement_acts.clear()

    def set_replacement(self, layer_name: str, replacement: torch.Tensor) -> None:
        """Set replacement activation for a layer"""
        self.replacement_acts[layer_name] = replacement
        
    def clear_replacement(self, layer_name: Optional[str] = None) -> None:
        """Clear replacement activation for a layer or all layers"""
        if layer_name is None:
            self.replacement_acts.clear()
        elif layer_name in self.replacement_acts:
            del self.replacement_acts[layer_name]
        
    def get_activations(self, input_ids: torch.Tensor, layer_name: str) -> torch.Tensor:
        # Move input_ids to correct device
        device = self.device
        input_ids = input_ids.to(device)
        
        if layer_name not in self.hooks:
            self.register_layer(layer_name)
        
        self.activations.clear()
        
        # Forward pass with cache disabled
        with torch.no_grad():
            self.model(
                input_ids,
                use_cache=False,
                return_dict=True
            )
        
        if layer_name not in self.activations:
            raise ValueError(f"No activations found for layer {layer_name}")

        return self.activations[layer_name]
        
    def __del__(self):
        """Clean up hooks when object is deleted"""
        self.remove_hooks()

    def inject_partial_activation(self,layer_name, custom_activation):
        """Inject partial activation for a layer"""
        self.set_replacement(layer_name, custom_activation)
        if layer_name not in self.hooks:
            self.register_layer(layer_name)