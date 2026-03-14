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

"""Minimal sparse autoencoder components used by the public SAE runner."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKActivation(nn.Module):
    """Keep the largest ``k`` positive activations and zero the rest."""

    def __init__(self, k: int):
        super().__init__()
        self.k = int(k)
        self.name = f"TopK-{self.k}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_relu = F.relu(x)
        values, indices = torch.topk(x_relu, self.k, dim=-1)
        output = torch.zeros_like(x)
        output.scatter_(-1, indices, values)
        return output


class SparseAutoencoder(nn.Module):
    """Small SAE used by the appendix-facing public experiments."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        activation_fn: nn.Module,
        tied_weights: bool = False,
        init_strategy: str = "kaiming",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_hidden = int(d_hidden)
        self.tied_weights = bool(tied_weights)

        self.W_enc = nn.Linear(d_model, d_hidden, bias=True)
        self.activation = activation_fn

        if tied_weights:
            self.b_dec = nn.Parameter(torch.zeros(d_model))
        else:
            self.W_dec = nn.Linear(d_hidden, d_model, bias=True)

        with torch.no_grad():
            if not tied_weights:
                self.W_dec.weight.data = F.normalize(self.W_dec.weight.data, p=2, dim=0)

            if init_strategy == "orthogonal":
                nn.init.orthogonal_(self.W_enc.weight)
                if self.W_enc.bias is not None:
                    nn.init.zeros_(self.W_enc.bias)
            elif init_strategy == "xavier":
                nn.init.xavier_uniform_(self.W_enc.weight)
                if self.W_enc.bias is not None:
                    nn.init.zeros_(self.W_enc.bias)
            else:
                nn.init.kaiming_uniform_(self.W_enc.weight, nonlinearity="linear")
                if self.W_enc.bias is not None:
                    nn.init.zeros_(self.W_enc.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pre_activations = self.W_enc(x)
        f = self.activation(pre_activations)

        if self.tied_weights:
            x_reconstruct = F.linear(f, self.W_enc.weight.t(), self.b_dec)
        else:
            x_reconstruct = self.W_dec(f)

        return x_reconstruct, f

    def get_decoder_weights(self) -> torch.Tensor:
        if self.tied_weights:
            return self.W_enc.weight.t()
        return self.W_dec.weight
