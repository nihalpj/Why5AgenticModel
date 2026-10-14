"""why5_synth: synthetic data factory for the Why5 agentic model.

Stages (see SYNTHETIC_DATA_PLAN.md §2.1):
    seeds -> generation -> validation -> JSONL storage
"""
__version__ = "0.1.0"

from .toolzoo import ToolZoo
from .seeds import SeedGenerator, GoldDepthRubric
from .sandbox import Sandbox, SandboxError
from .generator import TraceGenerator
from .validator import Validator
from .writer import DatasetWriter, read_dataset

__all__ = [
    "ToolZoo", "SeedGenerator", "GoldDepthRubric", "Sandbox", "SandboxError",
    "TraceGenerator", "Validator", "DatasetWriter", "read_dataset",
]
