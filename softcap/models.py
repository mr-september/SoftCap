# Copyright 2026 Larry Cai and Jie Tang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import copy

__all__ = ['SimpleMLP', 'DeepMLP', 'SimpleCNN', 'ConvNet', 'create_model', 'SimpleClassifier', 'get_default_architectures', 'get_model_for_analysis']

class SimpleMLP(nn.Module):
    """Configurable MLP for classification/regression.
    
    Suitable for 2D toy problems (extrapolation, spiral, angular benchmarks),
    with configurable depth and dimensions.
    """
    
    def __init__(
        self,
        activation_fn: nn.Module,
        input_dim: int = 2,
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_layers: int = 3,
        output_activation: nn.Module = None,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.output_activation = output_activation

        if num_layers == 1:
            self.input_layer = None
            self.hidden_layers = nn.ModuleList([])
            self.activations = nn.ModuleList([])
            self.output_layer = nn.Linear(input_dim, output_dim)
            return

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        ])
        self.activations = nn.ModuleList([
            copy.deepcopy(activation_fn) for _ in range(num_layers - 1)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_layer is None:
            x = self.output_layer(x)
        else:
            x = self.activations[0](self.input_layer(x))
            for layer, act in zip(self.hidden_layers, self.activations[1:]):
                x = act(layer(x))
            x = self.output_layer(x)

        if self.output_activation is not None:
            x = self.output_activation(x)
        return x

class DeepMLP(nn.Module):
    """Deep MLP for testing variance propagation and vanishing gradients.
    
    Note: Shares the same activation instance across all layers by default,
    which means if the activation has learnable parameters, they are shared
    across depth (tied weights).
    """

    def __init__(self, activation_fn: nn.Module, input_dim: int = 784, hidden_dim: int = 256, output_dim: int = 10, num_layers: int = 10, use_batch_norm: bool = False):
        super().__init__()
        self.activation_fn = activation_fn
        self.use_batch_norm = use_batch_norm
        
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            block = []
            block.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                block.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Sequential(*block))
            prev_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        for layer in self.layers[:-1]:
            x = layer(x)
            if hasattr(self.activation_fn, 'activation_function'):
                x = self.activation_fn.activation_function(x)
            else:
                x = self.activation_fn(x)
        x = self.layers[-1](x)
        return x


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification."""

    def __init__(self, activation_fn: nn.Module, input_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.act1 = activation_fn
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.act2 = copy.deepcopy(activation_fn)
        self.pool2 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.act3 = copy.deepcopy(activation_fn)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x

# Aliases
ConvNet = SimpleCNN


class SimpleClassifier(nn.Module):
    """Simple MLP classifier for synthetic benchmarks.
    
    Duplicates logic from synthetic_benchmarks, kept here for compatibility.
    """
    
    def __init__(self, activation: nn.Module, hidden_size: int = 64, num_layers: int = 3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(2, hidden_size))
        layers.append(activation)
        
        # Note: shared activation instance here, matching original synthetic_benchmarks implementation
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)
        
        layers.append(nn.Linear(hidden_size, 2))
        
        self.network = nn.Sequential(*layers)
        self.activation_name = getattr(activation, 'name', activation.__class__.__name__)
    
    def forward(self, x):
        return self.network(x)


def get_default_architectures():
    """Return list of available model architectures."""
    return ['SimpleMLP', 'DeepMLP', 'ConvNet', 'SimpleClassifier']

def create_model(model_name: str, activation_fn: nn.Module, **kwargs) -> nn.Module:
    """Factory function for creating models."""
    if model_name == 'SimpleMLP':
        return SimpleMLP(activation_fn, **kwargs)
    elif model_name == 'DeepMLP':
        return DeepMLP(activation_fn, **kwargs)
    elif model_name in ('ConvNet', 'SimpleCNN'):
        return SimpleCNN(activation_fn, **kwargs)
    elif model_name == 'SimpleClassifier':
        return SimpleClassifier(activation_fn, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_model_for_analysis(model_name: str, activation_fn: nn.Module, **kwargs) -> nn.Module:
    """Alias for create_model."""
    return create_model(model_name, activation_fn, **kwargs)
