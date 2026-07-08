from typing import Any

import torch


def build_optimizer(
    model: Any,
    *,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """AdamW over all trainable parameters. Deliberately unconfigurable beyond LR/WD."""
    parameters = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: str = "constant",
    warmup_steps: int = 0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Constant LR, optionally with linear warmup. No cosine/decay in this phase.

    ``"constant"`` keeps the configured LR from step 0. ``"linear_warmup"``
    ramps linearly from 0 over ``warmup_steps`` optimizer steps, then stays
    constant. Anything fancier is a later extension, not a Phase-1 feature.
    """
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if name == "linear_warmup":

        def factor(step: int) -> float:
            if warmup_steps <= 0:
                return 1.0
            return min(1.0, (step + 1) / warmup_steps)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    msg = f"Unknown lr_scheduler {name!r}; known: constant, linear_warmup"
    raise ValueError(msg)


def clip_gradients(model: Any, max_grad_norm: float | None) -> float | None:
    """Clip global grad norm and return the pre-clip norm; no-op when disabled."""
    if max_grad_norm is None or max_grad_norm <= 0:
        return None
    parameters = [param for param in model.parameters() if param.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    return float(norm)
