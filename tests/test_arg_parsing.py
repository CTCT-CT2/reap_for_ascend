"""Smoke tests for CLI argument parsing (catches help-string formatting bugs)."""

import sys
import pytest
from transformers import HfArgumentParser

from reap.args import (
    ReapArgs,
    ModelArgs,
    DatasetArgs,
    ObserverArgs,
    PruneArgs,
    LayerwiseArgs,
)

# The two parser configurations used by main.py and layerwise_prune.py
MAIN_DATACLASSES = (
    ReapArgs, ModelArgs, DatasetArgs, ObserverArgs,
)
LAYERWISE_DATACLASSES = (
    ReapArgs, DatasetArgs, ObserverArgs, ModelArgs,
    PruneArgs, LayerwiseArgs,
)


@pytest.mark.parametrize(
    "dataclasses",
    [MAIN_DATACLASSES, LAYERWISE_DATACLASSES],
    ids=["main", "layerwise"],
)
def test_help_format_strings(dataclasses, monkeypatch):
    """Ensure `--help` doesn't crash (e.g. unescaped %-codes in help text)."""
    monkeypatch.setattr(sys, "argv", ["test"])
    parser = HfArgumentParser(dataclasses)
    # format_help() exercises the same %-expansion that triggers the bug
    parser.format_help()
