"""
DistilBERT activation-swap utilities.

DistilBERT's FFN layers use GELU by default. To fairly compare custom
activation functions, we replace the activation callable inside each
``distilbert.transformer.layer[i].ffn`` sub-module.

The FFN structure in HuggingFace DistilBERT:
    x → lin1 → activation → lin2 → dropout → output

We patch ``ffn.activation`` on each layer to point to our custom
activation module.
"""

import copy
import torch.nn as nn


def swap_distilbert_activations(model, activation_fn: nn.Module) -> None:
    """Replace FFN activation in every transformer layer of DistilBERT.

    This modifies the model **in-place**.

    Args:
        model: A HuggingFace DistilBertForSequenceClassification (or similar).
        activation_fn: The activation to use. Deep-copied per layer so each
                       layer has independent learnable parameters.
    """
    for i, layer in enumerate(model.distilbert.transformer.layer):
        # Each layer has .ffn.activation which is a callable
        layer.ffn.activation = copy.deepcopy(activation_fn)


def get_activation_param_count(model) -> int:
    """Count trainable parameters that belong to activation functions."""
    count = 0
    for layer in model.distilbert.transformer.layer:
        act = layer.ffn.activation
        if isinstance(act, nn.Module):
            count += sum(p.numel() for p in act.parameters() if p.requires_grad)
    return count
