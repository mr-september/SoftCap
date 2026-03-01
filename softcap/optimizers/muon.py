"""
Muon Optimizer: Momentum Orthogonalized by Newton-Schulz

Implementation based on:
- Bernstein et al. (2025) arXiv:2502.16982v1
- KellerJordan/Muon reference implementation

The Muon optimizer orthogonalizes momentum updates via Newton-Schulz iterations,
providing superior convergence on 2D weight matrices. At scale, this requires
QK-Clip to prevent exploding attention logits (our hypothesis: bounded activations
may substitute for QK-Clip).

Key Properties:
- Applies orthogonalization to 2D parameters only (Linear, Conv unfolded)
- Uses Newton-Schulz iteration (5 steps optimal)
- Requires separate optimizer (AdamW) for non-2D params (embeddings, biases)

Usage:
    from softcap.optimizers.muon import Muon, create_muon_optimizer_groups
    
    # Create optimizer with automatic parameter splitting
    optimizers = create_muon_optimizer_groups(model, muon_lr=0.02, adamw_lr=0.001)
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW
from typing import List, Dict, Tuple, Optional, Any, Iterable
import math


class Muon(Optimizer):
    """
    Muon Optimizer: Momentum Orthogonalized by Newton-Schulz.
    
    Applies Newton-Schulz orthogonalization to momentum updates for 2D parameters.
    This is the core Muon algorithm without QK-Clip (we test if bounded activations
    can substitute).
    
    Args:
        params: Iterable of parameters (should be 2D only)
        lr: Learning rate (default: 0.02, higher than typical Adam)
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iteration steps (default: 5, optimal)
        weight_decay: AdamW-style weight decay (default: 0.0)
        warmup_steps: Momentum warmup steps (default: 300)
        warmup_momentum: Initial momentum during warmup (default: 0.85)
    
    References:
        - Bernstein et al. (2025): "Muon: A Momentum Optimizer"
        - https://github.com/KellerJordan/Muon
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        warmup_steps: int = 300,
        warmup_momentum: float = 0.85,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid Newton-Schulz steps: {ns_steps}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            warmup_momentum=warmup_momentum,
        )
        super().__init__(params, defaults)
        
        # Track global step for warmup
        self.global_step = 0
    
    @staticmethod
    @torch.no_grad()
    def newton_schulz_orthogonalize(
        G: torch.Tensor,
        steps: int = 5,
        eps: float = 1e-7
    ) -> torch.Tensor:
        """
        Apply Newton-Schulz iteration to orthogonalize matrix G.
        
        Uses the quintic iteration from KellerJordan/Muon reference:
            X_{k+1} = a*X + b*(X @ X^T @ X) + c*(X @ X^T @ X @ X^T @ X)
        
        With coefficients optimized for fast convergence:
            a = 3.4445, b = -4.7750, c = 2.0315
        
        This converges quadratically to the orthogonal Procrustes solution.
        
        Args:
            G: Input matrix (m x n), will be orthogonalized
            steps: Number of Newton-Schulz iterations (5 is optimal)
            eps: Small epsilon for numerical stability
        
        Returns:
            Orthogonalized matrix with same shape as G
        """
        # Ensure G is 2D
        assert G.dim() == 2, f"Expected 2D tensor, got {G.dim()}D"
        
        # Transpose to make the matrix "tall" (more rows than cols)
        # This is more numerically stable and efficient
        m, n = G.shape
        transpose = m < n
        if transpose:
            G = G.T
            m, n = n, m
        
        # Normalize to unit Frobenius norm for stable iteration
        # The Newton-Schulz iteration is sensitive to the initial scale
        G_norm = G.norm()
        if G_norm < eps:
            # Handle near-zero gradient
            if transpose:
                return G.T
            return G
        
        # Scale to unit norm
        X = G / G_norm
        
        # Quintic Newton-Schulz coefficients (from KellerJordan/Muon)
        a = 3.4445
        b = -4.7750
        c = 2.0315
        
        # Newton-Schulz iterations
        for _ in range(steps):
            # A = X @ X^T  (m x m matrix)
            A = X @ X.T
            # B = A @ X (same as X @ X^T @ X)
            B = A @ X
            # C = A @ B (same as X @ X^T @ X @ X^T @ X)
            C = A @ B
            # X_new = a*X + b*B + c*C
            X = a * X + b * B + c * C
        
        # Do NOT restore original norm — the output should be (approximately)
        # orthogonal. The learning rate controls step magnitude.
        # Ref: KellerJordan/Muon reference implementation.
        
        if transpose:
            X = X.T
        
        return X

    
    def get_current_momentum(self) -> float:
        """Get current momentum with warmup schedule."""
        group = self.param_groups[0]  # All groups should have same warmup
        warmup_steps = group['warmup_steps']
        
        if self.global_step >= warmup_steps:
            return group['momentum']
        
        # Linear warmup from warmup_momentum to momentum
        warmup_momentum = group['warmup_momentum']
        target_momentum = group['momentum']
        progress = self.global_step / warmup_steps
        return warmup_momentum + progress * (target_momentum - warmup_momentum)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        
        Returns:
            Loss value if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        current_momentum = self.get_current_momentum()
        
        for group in self.param_groups:
            lr = group['lr']
            ns_steps = group['ns_steps']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Validate 2D
                if p.dim() != 2:
                    raise RuntimeError(
                        f"Muon optimizer expects 2D parameters only. "
                        f"Got parameter with shape {p.shape}. "
                        f"Use create_muon_optimizer_groups() for automatic splitting."
                    )
                
                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                buf = state['momentum_buffer']
                
                # Update momentum buffer
                buf.mul_(current_momentum).add_(grad)
                
                # Orthogonalize the update
                if nesterov:
                    # Nesterov: use momentum + current gradient
                    update = current_momentum * buf + grad
                else:
                    update = buf
                
                # Apply Newton-Schulz orthogonalization
                # Per Bernstein et al. (2025), the orthogonalized matrix IS the update
                # direction. The learning rate alone controls step size.
                # Do NOT rescale to original norm — that undoes the orthogonalization.
                update_ortho = self.newton_schulz_orthogonalize(update, steps=ns_steps)
                
                # Apply update
                p.add_(update_ortho, alpha=-lr)
                
                # Apply weight decay AFTER update (AdamW-style: decoupled)
                if weight_decay > 0:
                    p.add_(p, alpha=-lr * weight_decay)
        
        self.global_step += 1
        return loss


def create_muon_optimizer_groups(
    model: nn.Module,
    muon_lr: float = 0.02,
    adamw_lr: float = 0.001,
    muon_momentum: float = 0.95,
    weight_decay: float = 0.01,
    ns_steps: int = 5,
    warmup_steps: int = 300,
    exclude_patterns: Optional[List[str]] = None,
) -> Tuple[Muon, AdamW]:
    """
    Create Muon + AdamW optimizer pair with automatic parameter splitting.
    
    Muon is applied to 2D weight matrices (Linear, Conv flattened).
    AdamW is applied to everything else (embeddings, biases, LayerNorm).
    
    Args:
        model: The model to optimize
        muon_lr: Learning rate for Muon (typically 0.02)
        adamw_lr: Learning rate for AdamW (typically 0.001)
        muon_momentum: Momentum for Muon
        weight_decay: Weight decay for both optimizers
        ns_steps: Newton-Schulz iteration steps
        warmup_steps: Momentum warmup steps for Muon
        exclude_patterns: Parameter name patterns to exclude from Muon
    
    Returns:
        Tuple of (Muon optimizer, AdamW optimizer)
    
    Example:
        muon_opt, adamw_opt = create_muon_optimizer_groups(model)
        
        for epoch in range(epochs):
            for batch in dataloader:
                loss = compute_loss(model, batch)
                loss.backward()
                muon_opt.step()
                adamw_opt.step()
                muon_opt.zero_grad()
                adamw_opt.zero_grad()
    """
    if exclude_patterns is None:
        exclude_patterns = ['embed', 'cls_token', 'pos_embed', 'norm', 'ln', 'layernorm']
    
    muon_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if excluded by pattern
        name_lower = name.lower()
        is_excluded = any(pat in name_lower for pat in exclude_patterns)
        
        # Check if 2D (suitable for Muon)
        is_2d = param.dim() == 2
        
        # Note: bias is 1D, so automatically goes to AdamW
        if is_2d and not is_excluded:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    
    # Create optimizers
    muon_opt = Muon(
        muon_params,
        lr=muon_lr,
        momentum=muon_momentum,
        ns_steps=ns_steps,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
    )
    
    adamw_opt = AdamW(
        adamw_params,
        lr=adamw_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    
    return muon_opt, adamw_opt


class MuonOptimizerScheduler:
    """
    Learning rate scheduler wrapper for Muon + AdamW optimizer pair.
    
    Provides cosine annealing with warmup for both optimizers.
    """
    
    def __init__(
        self,
        muon_opt: Muon,
        adamw_opt: AdamW,
        total_steps: int,
        warmup_steps: int = 500,
        min_lr_ratio: float = 0.1,
    ):
        self.muon_opt = muon_opt
        self.adamw_opt = adamw_opt
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        
        # Store initial learning rates
        self.muon_base_lr = muon_opt.param_groups[0]['lr']
        self.adamw_base_lr = adamw_opt.param_groups[0]['lr']
        
        self.current_step = 0
    
    def step(self):
        """Update learning rates for both optimizers."""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            scale = self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        
        # Update learning rates
        for group in self.muon_opt.param_groups:
            group['lr'] = self.muon_base_lr * scale
        
        for group in self.adamw_opt.param_groups:
            group['lr'] = self.adamw_base_lr * scale
    
    def get_lr(self) -> Dict[str, float]:
        """Get current learning rates."""
        return {
            'muon_lr': self.muon_opt.param_groups[0]['lr'],
            'adamw_lr': self.adamw_opt.param_groups[0]['lr'],
        }
