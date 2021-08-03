#!/usr/bin/env python

"""
A module to provide the boilerplate needed for all the analysis.
"""

import json
import re
from functools import partial
from typing import List, Dict, Final

import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]
import matplotlib  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import seaborn as sns  # type: ignore[import]

from imc import Project  # type: ignore[import]
from imc.types import Path, DataFrame  # type: ignore[import]


class Config:
    # constants

    ## Major attributes to contrast when comparing observation groups
    attributes: Final[List[str]] = []

    figkws: Final[Dict] = dict(
        dpi=300, bbox_inches="tight", pad_inches=0.1, transparent=False
    )

    # directories
    metadata_dir: Final[Path] = Path("metadata")
    data_dir: Final[Path] = Path("data")
    processed_dir: Final[Path] = Path("processed")
    results_dir: Final[Path] = Path("results")

    # # Order
    cat_order: Final[Dict[str, List]] = dict(
        cat1=["val1", "val2"],
        cat2=["val1", "val2"],
    )

    # Color codes
    colors: Final[Dict[str, np.ndarray]] = dict(
        cat1=np.asarray(sns.color_palette())[[2, 0, 1, 3]],
        cat2=np.asarray(sns.color_palette())[[2, 0, 1, 5, 4, 3]],
    )

    # Output files
    metadata_file: Final[Path] = metadata_dir / "clinical_annotation.pq"
    quantification_file: Final[Path] = results_dir / "cell_type" / "quantification.pq"
    gating_file: Final[Path] = results_dir / "cell_type" / "gating.pq"
    h5ad_file: Final[Path] = (
        results_dir / "cell_type" / "anndata.all_cells.processed.h5ad"
    )
    counts_file: Final[Path] = results_dir / "cell_type" / "cell_type_counts.pq"
