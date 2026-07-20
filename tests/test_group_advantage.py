import math

import pytest

from whetstone.rollout.group_advantage import compute_group_advantages


def test_group_advantage_all_equal_rewards() -> None:
    assert compute_group_advantages([1.0, 1.0, 1.0, 1.0], 4) == [0.0, 0.0, 0.0, 0.0]
    assert compute_group_advantages([0.0, 0.0, 0.0, 0.0], 4) == [0.0, 0.0, 0.0, 0.0]


def test_group_advantage_single_positive() -> None:
    advantages = compute_group_advantages([1.0, 0.0, 0.0, 0.0], 4)
    assert advantages == pytest.approx([0.75, -0.25, -0.25, -0.25])


def test_group_advantage_two_positive_two_negative() -> None:
    advantages = compute_group_advantages([1.0, 1.0, 0.0, 0.0], 4)
    assert advantages == pytest.approx([0.5, 0.5, -0.5, -0.5])


def test_group_advantage_multiple_groups_are_independent() -> None:
    advantages = compute_group_advantages([1.0, 0.0, 1.0, 1.0], 2)
    assert advantages == pytest.approx([0.5, -0.5, 0.0, 0.0])


def test_group_advantage_normalization_optional() -> None:
    # Unnormalized is the default.
    default = compute_group_advantages([1.0, 0.0, 0.0, 0.0], 4)
    assert default == pytest.approx([0.75, -0.25, -0.25, -0.25])

    normalized = compute_group_advantages([1.0, 0.0, 0.0, 0.0], 4, normalize=True)
    # Population std of [1,0,0,0] is sqrt(3)/4, so 0.75 / (sqrt(3)/4) = sqrt(3).
    assert normalized[0] == pytest.approx(math.sqrt(3), rel=1e-4)
    assert normalized[1] == pytest.approx(-math.sqrt(3) / 3, rel=1e-4)

    # All-equal groups stay exactly zero even when normalizing.
    assert compute_group_advantages([1.0, 1.0], 2, normalize=True) == [0.0, 0.0]


def test_zero_variance_group_is_total_for_any_epsilon() -> None:
    # All-equal groups short-circuit to exact zeros: no 0/0 even at epsilon=0.
    assert compute_group_advantages([0.0, 0.0, 0.0, 0.0], 4, normalize=True, epsilon=0.0) == [
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_advantage_config_rejects_nonpositive_epsilon() -> None:
    from pydantic import ValidationError

    from whetstone.train.config import AdvantageConfig

    with pytest.raises(ValidationError):
        AdvantageConfig(epsilon=0.0)


def test_group_advantage_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="multiple"):
        compute_group_advantages([1.0, 0.0, 1.0], 2)
    with pytest.raises(ValueError, match="group_size"):
        compute_group_advantages([1.0], 0)
