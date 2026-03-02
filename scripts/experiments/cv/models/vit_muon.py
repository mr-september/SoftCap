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

"""
Vision Transformer with Q/K Activation Injection for Muon Optimizer Research

This module extends the standard ViTTiny to support:
1. Activation function injection AFTER Q/K projections, BEFORE matmul
2. Max QK score tracking for stability analysis
3. Optional QK-Clip for comparison experiments

The core hypothesis: SoftCap's bounded output [0, a) naturally constrains 
pre-softmax attention scores, potentially eliminating the need for QK-Clip.

Usage:
    from softcap.models.vit_muon import ViTMuonResearch
    
    model = ViTMuonResearch(
        qk_activation=SoftCap(a_init=2.5),
        track_attention_stats=True
    )
"""

import copy
import math
from typing import Optional, Tuple, Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionWithQKActivation(nn.Module):
    """
    Multi-head self-attention with activation injection on Q/K outputs.
    
    This is the key mechanism for testing whether bounded activations
    can substitute for QK-Clip in Muon-optimized transformers.
    
    The activation is applied AFTER Q/K linear projections but BEFORE
    the attention score computation (Q @ K.T), exactly where QK-Clip
    operates at the weight level.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_activation: nn.Module = None,
        dropout: float = 0.1,
        track_stats: bool = True,
        qk_clip_threshold: Optional[float] = None,  # If set, apply explicit QK-Clip
        intervention_config: Optional[Dict[str, Any]] = None,  # NEW: For intervention tests
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Separate Q, K, V projections (instead of combined qkv)
        # This allows cleaner activation injection on Q and K only
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        # Activation to apply after Q/K projections
        self.qk_activation = qk_activation  # None = standard (no activation)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Stats tracking
        self.track_stats = track_stats
        self.qk_clip_threshold = qk_clip_threshold
        
        # Intervention test configuration (NEW)
        self.intervention_config = intervention_config or {}
        self.intervention_enabled = intervention_config is not None
        
        # Running stats (updated during forward, read externally)
        self._max_qk_score = None
        self._max_qk_score_post_clip = None
        self._qk_score_mean = None
        self._qk_score_std = None
        self._clip_fraction = None
        self._clip_applied = False
        # Enhanced telemetry (added for tail analysis)
        self._qk_percentiles = None  # Dict of percentiles
        self._qk_kurtosis = None     # Excess kurtosis
        self._qk_threshold_fractions = None  # Fraction above thresholds
    
    def get_attention_stats(self) -> Dict[str, float]:
        """Get attention statistics from last forward pass (with enhanced telemetry)."""
        if not self.track_stats:
            return {}
        
        stats = {
            'max_qk_score': self._max_qk_score.item() if self._max_qk_score is not None else None,
            'max_qk_score_post_clip': self._max_qk_score_post_clip.item() if self._max_qk_score_post_clip is not None else None,
            'qk_score_mean': self._qk_score_mean.item() if self._qk_score_mean is not None else None,
            'qk_score_std': self._qk_score_std.item() if self._qk_score_std is not None else None,
            'clip_fraction': self._clip_fraction.item() if self._clip_fraction is not None else None,
            'clip_applied': self._clip_applied,
        }
        
        # Add enhanced telemetry if available
        if self._qk_percentiles is not None:
            stats['percentiles'] = self._qk_percentiles
        if self._qk_kurtosis is not None:
            stats['kurtosis'] = self._qk_kurtosis
        if self._qk_threshold_fractions is not None:
            stats['threshold_fractions'] = self._qk_threshold_fractions
        
        return stats
    
    def _apply_intervention(self, tensor: torch.Tensor, target: str) -> torch.Tensor:
        """
        Apply intervention (corruption) to Q or K tensors.
        
        Args:
            tensor: [B, N, dim] Q or K tensor (before reshaping)
            target: 'q' or 'k' (which tensor this is)
        
        Returns:
            Corrupted tensor
        """
        if target not in self.intervention_config.get('targets', ['q', 'k']):
            return tensor  # Skip if this target is not configured
        
        fraction = self.intervention_config.get('fraction', 0.01)
        scale = self.intervention_config.get('scale', 5.0)
        
        B, N, D = tensor.shape
        
        # Select random elements to corrupt
        mask = torch.rand_like(tensor) < fraction
        
        # Multiply selected elements by scale factor
        corrupted = tensor.clone()
        corrupted[mask] *= scale
        
        return corrupted
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Separate Q, K, V projections
        q = self.q_proj(x)  # [B, N, dim]
        k = self.k_proj(x)  # [B, N, dim]
        v = self.v_proj(x)  # [B, N, dim]
        
        # Apply activation to Q and K (KEY MECHANISM)
        if self.qk_activation is not None:
            q = self.qk_activation(q)
            k = self.qk_activation(k)
        
        # INTERVENTION TEST: Inject artificial extremes (if enabled)
        if self.intervention_enabled and not self.training:
            q = self._apply_intervention(q, 'q')
            k = self._apply_intervention(k, 'k')
        
        # Reshape for multi-head attention
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, head_dim]
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention score computation (PRE-SOFTMAX)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        
        # Track statistics before softmax (ENHANCED TELEMETRY)
        if self.track_stats:
            with torch.no_grad():
                # Basic stats (existing)
                self._max_qk_score = attn_scores.max()
                self._qk_score_mean = attn_scores.mean()
                self._qk_score_std = attn_scores.std()
                
                # Enhanced telemetry: percentiles, kurtosis, threshold fractions
                # Flatten for easier computation
                scores_flat = attn_scores.flatten()
                
                # Percentiles (p50, p90, p99, p99.9)
                percentiles_tensor = torch.quantile(scores_flat, 
                                                    torch.tensor([0.5, 0.9, 0.99, 0.999], 
                                                                device=scores_flat.device))
                self._qk_percentiles = {
                    'p50': percentiles_tensor[0].item(),
                    'p90': percentiles_tensor[1].item(),
                    'p99': percentiles_tensor[2].item(),
                    'p999': percentiles_tensor[3].item(),
                }
                
                # Kurtosis (excess kurtosis = kurtosis - 3)
                # Kurt = E[(X - μ)^4] / σ^4 - 3
                mean = self._qk_score_mean
                std = self._qk_score_std
                if std > 1e-8:  # Avoid division by zero
                    centered = scores_flat - mean
                    fourth_moment = (centered ** 4).mean()
                    kurtosis = fourth_moment / (std ** 4)
                    excess_kurtosis = kurtosis - 3.0
                    self._qk_kurtosis = excess_kurtosis.item()
                else:
                    self._qk_kurtosis = 0.0
                
                # Threshold fractions (fraction of scores exceeding thresholds)
                self._qk_threshold_fractions = {
                    'above_10': (scores_flat > 10).float().mean().item(),
                    'above_30': (scores_flat > 30).float().mean().item(),
                    'above_100': (scores_flat > 100).float().mean().item(),
                }
        
        # Optional: apply explicit QK-Clip for comparison
        self._clip_applied = False
        self._clip_fraction = torch.tensor(0.0, device=attn_scores.device)
        if self.qk_clip_threshold is not None:
            clipped_mask = attn_scores > self.qk_clip_threshold
            self._clip_fraction = clipped_mask.float().mean()
            max_score = attn_scores.max()
            if max_score > self.qk_clip_threshold:
                # Scale down Q and K weights would be done at optimizer level
                # Here we just clip the scores directly as a fallback
                attn_scores = attn_scores.clamp(max=self.qk_clip_threshold)
                self._clip_applied = True

        if self.track_stats:
            with torch.no_grad():
                self._max_qk_score_post_clip = attn_scores.max()
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Apply attention to values
        out = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        
        return out
    
    def forward_with_scores(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that also returns pre-softmax attention scores."""
        B, N, C = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        if self.qk_activation is not None:
            q = self.qk_activation(q)
            k = self.qk_activation(k)
        
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        
        # Track statistics (ENHANCED TELEMETRY - same as forward method)
        if self.track_stats:
            with torch.no_grad():
                # Basic stats
                self._max_qk_score = attn_scores.max()
                self._qk_score_mean = attn_scores.mean()
                self._qk_score_std = attn_scores.std()
                
                # Enhanced telemetry
                scores_flat = attn_scores.flatten()
                
                # Percentiles
                percentiles_tensor = torch.quantile(scores_flat, 
                                                    torch.tensor([0.5, 0.9, 0.99, 0.999], 
                                                                device=scores_flat.device))
                self._qk_percentiles = {
                    'p50': percentiles_tensor[0].item(),
                    'p90': percentiles_tensor[1].item(),
                    'p99': percentiles_tensor[2].item(),
                    'p999': percentiles_tensor[3].item(),
                }
                
                # Kurtosis
                mean = self._qk_score_mean
                std = self._qk_score_std
                if std > 1e-8:
                    centered = scores_flat - mean
                    fourth_moment = (centered ** 4).mean()
                    kurtosis = fourth_moment / (std ** 4)
                    excess_kurtosis = kurtosis - 3.0
                    self._qk_kurtosis = excess_kurtosis.item()
                else:
                    self._qk_kurtosis = 0.0
                
                # Threshold fractions
                self._qk_threshold_fractions = {
                    'above_10': (scores_flat > 10).float().mean().item(),
                    'above_30': (scores_flat > 30).float().mean().item(),
                    'above_100': (scores_flat > 100).float().mean().item(),
                }
        
        # Apply QK-Clip if configured (same as forward)
        self._clip_applied = False
        self._clip_fraction = torch.tensor(0.0, device=attn_scores.device)
        if self.qk_clip_threshold is not None:
            clipped_mask = attn_scores > self.qk_clip_threshold
            self._clip_fraction = clipped_mask.float().mean()
            max_score = attn_scores.max()
            if max_score > self.qk_clip_threshold:
                attn_scores = attn_scores.clamp(max=self.qk_clip_threshold)
                self._clip_applied = True

        if self.track_stats:
            with torch.no_grad():
                self._max_qk_score_post_clip = attn_scores.max()
        
        # Save pre-softmax scores for return
        raw_scores = attn_scores.detach()
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        out = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        
        return out, raw_scores


class TransformerBlockMuon(nn.Module):
    """Transformer block with QK activation injection support."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        mlp_activation: nn.Module = None,
        qk_activation: nn.Module = None,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        track_stats: bool = True,
        qk_clip_threshold: Optional[float] = None,
        intervention_config: Optional[Dict[str, Any]] = None,  # NEW
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttentionWithQKActivation(
            dim=dim,
            num_heads=num_heads,
            qk_activation=qk_activation,
            dropout=dropout,
            track_stats=track_stats,
            qk_clip_threshold=qk_clip_threshold,
            intervention_config=intervention_config,  # NEW
        )
        
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            activation=mlp_activation or nn.GELU(),
            dropout=dropout
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
    
    def get_attention_stats(self) -> Dict[str, float]:
        """Get attention statistics from this block."""
        return self.attn.get_attention_stats()


class MLP(nn.Module):
    """MLP block for transformer."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        activation: nn.Module,
        dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = activation
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ViTMuonResearch(nn.Module):
    """
    Vision Transformer for Muon Optimizer Research.
    
    Key modifications from standard ViT:
    - Q/K activation injection for testing bounded activation hypothesis
    - Per-layer attention score tracking
    - Optional QK-Clip for comparison
    
    Args:
        img_size: Input image size (default: 32 for CIFAR)
        patch_size: Patch size (default: 4)
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        mlp_activation: Activation for MLP blocks
        qk_activation: Activation for Q/K projections (the key mechanism)
        dropout: Dropout rate
        drop_path: Stochastic depth rate
        track_attention_stats: Whether to track max QK scores
        qk_clip_threshold: If set, apply explicit QK-Clip at this threshold
    """
    
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 100,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        mlp_activation: nn.Module = None,
        qk_activation: nn.Module = None,  # KEY: activation for Q/K
        dropout: float = 0.1,
        drop_path: float = 0.1,
        track_attention_stats: bool = True,
        qk_clip_threshold: Optional[float] = None,
        intervention_config: Optional[Dict[str, Any]] = None,  # NEW
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.track_attention_stats = track_attention_stats
        self.intervention_config = intervention_config  # NEW
        
        # Store activation info
        self.mlp_activation = mlp_activation or nn.GELU()
        self.qk_activation = qk_activation
        self.mlp_activation_name = getattr(mlp_activation, 'name', type(mlp_activation).__name__) if mlp_activation else 'GELU'
        self.qk_activation_name = getattr(qk_activation, 'name', type(qk_activation).__name__) if qk_activation else 'None'
        
        # Patch embedding
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer encoder layers with QK activation injection
        # Each layer gets its own deepcopy of qk_activation so learnable
        # parameters (e.g. 'a') are independent per layer.
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlockMuon(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                mlp_activation=self.mlp_activation,
                qk_activation=copy.deepcopy(qk_activation) if qk_activation is not None else None,
                dropout=dropout,
                drop_path=dpr[i],
                track_stats=track_attention_stats,
                qk_clip_threshold=qk_clip_threshold,
                intervention_config=intervention_config,  # NEW
            )
            for i in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize embedding and normalization weights only.
        
        Linear layer weights are initialized by apply_initialization() in the
        sweep script, which selects kaiming/orthogonal/xavier/softcap_optimal.
        """
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        
        return x
    
    def get_all_attention_stats(self) -> Dict[str, Dict[str, float]]:
        """Get attention statistics from all layers."""
        if not self.track_attention_stats:
            return {}
        
        stats = {}
        for i, block in enumerate(self.blocks):
            stats[f'layer_{i}'] = block.get_attention_stats()
        
        return stats
    
    def get_max_qk_score(self) -> float:
        """Get the maximum QK score across all layers."""
        if not self.track_attention_stats:
            return float('nan')
        
        max_scores = []
        for block in self.blocks:
            layer_stats = block.get_attention_stats()
            if layer_stats.get('max_qk_score') is not None:
                max_scores.append(layer_stats['max_qk_score'])
        
        return max(max_scores) if max_scores else float('nan')

    def get_max_qk_score_post_clip(self) -> float:
        """Get the maximum post-clip QK score across all layers."""
        if not self.track_attention_stats:
            return float('nan')

        max_scores = []
        for block in self.blocks:
            layer_stats = block.get_attention_stats()
            if layer_stats.get('max_qk_score_post_clip') is not None:
                max_scores.append(layer_stats['max_qk_score_post_clip'])

        return max(max_scores) if max_scores else float('nan')
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'ViTMuonResearch',
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_classes': self.num_classes,
            'mlp_activation': self.mlp_activation_name,
            'qk_activation': self.qk_activation_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        }
