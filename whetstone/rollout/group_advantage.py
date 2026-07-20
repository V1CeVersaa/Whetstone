from collections.abc import Sequence
from statistics import fmean, pstdev


def compute_group_advantages(
    rewards: Sequence[float],
    group_size: int,
    *,
    normalize: bool = False,
    epsilon: float = 1.0e-6,
) -> list[float]:
    """Per-prompt group-baseline advantages for grouped rollouts.

    ``rewards`` is a flat sequence of ``B * G`` values where each contiguous
    block of ``group_size`` entries came from the same prompt. The default is
    the unnormalized group-mean baseline

    ``A[g] = r[g] - mean(r[group])``

    which is deliberately simple: a group whose rewards all agree yields zero
    advantages (no learning signal), and e.g. ``[1, 0, 0, 0]`` yields
    ``[0.75, -0.25, -0.25, -0.25]``. With ``normalize=True`` each advantage is
    further divided by the in-group population std plus ``epsilon``.

    Raises:
        ValueError: If ``group_size < 1`` or ``len(rewards)`` is not a multiple of ``group_size``.
    """
    if group_size < 1:
        msg = f"group_size must be >= 1, got {group_size}"
        raise ValueError(msg)
    if len(rewards) % group_size != 0:
        msg = f"len(rewards)={len(rewards)} is not a multiple of group_size={group_size}"
        raise ValueError(msg)

    advantages: list[float] = []
    for start in range(0, len(rewards), group_size):
        group = [float(reward) for reward in rewards[start : start + group_size]]
        baseline = fmean(group)
        centered = [reward - baseline for reward in group]
        if normalize:
            std = pstdev(group) if len(group) > 1 else 0.0
            if std > 0.0:
                centered = [value / (std + epsilon) for value in centered]
        advantages.extend(centered)
    return advantages
