#!/usr/bin/env python

"""
A module to provide the boilerplate needed for all the analysis.
"""

import typing as tp

import numpy as np  # type: ignore[import]
import seaborn as sns  # type: ignore[import]

from src.types import Path


class Config:
    # constants
    figkws: tp.Final[tp.Dict] = dict(
        dpi=300, bbox_inches="tight", pad_inches=0.1, transparent=False
    )

    # directories
    metadata_dir: tp.Final[Path] = Path("metadata")
    data_dir: tp.Final[Path] = Path("data")
    processed_dir: tp.Final[Path] = Path("processed")
    results_dir: tp.Final[Path] = Path("results")
