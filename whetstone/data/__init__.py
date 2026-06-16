from whetstone.data.base import DATASET_REGISTRY, DatasetAdapter, get_dataset_adapter
from whetstone.data.gsm8k import GSM8KAdapter
from whetstone.data.openr1_math import OpenR1MathAdapter
from whetstone.data.taco_cobalt import TacoCobaltAdapter
from whetstone.data.tiny import TinyCodeAdapter, TinyMathAdapter

# Register adapters (with their spelling aliases) here, after the classes are
# imported -- doing it in base.py would create a circular import, since each
# adapter module imports helpers from base.
for _name in ("tiny_math", "tiny-math"):
    DATASET_REGISTRY.register(_name, TinyMathAdapter)
for _name in ("tiny_code", "tiny-code"):
    DATASET_REGISTRY.register(_name, TinyCodeAdapter)
for _name in ("gsm8k",):
    DATASET_REGISTRY.register(_name, GSM8KAdapter)
for _name in ("openr1_math", "openr1-math", "openr1_math_220k"):
    DATASET_REGISTRY.register(_name, OpenR1MathAdapter)
for _name in ("taco_cobalt", "taco-cobalt"):
    DATASET_REGISTRY.register(_name, TacoCobaltAdapter)

__all__ = [
    "DATASET_REGISTRY",
    "DatasetAdapter",
    "GSM8KAdapter",
    "OpenR1MathAdapter",
    "TacoCobaltAdapter",
    "get_dataset_adapter",
]
