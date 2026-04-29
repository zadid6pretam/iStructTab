"""
iStructTab package

Multimodal (Image + Tabular) model with
GEDS feature sequencing + OEMT (Linformer).

Typical usage:
    from iStructTab import iStructTab, set_seed
"""

from .iStructTab import (
    set_seed,
    ImageFeatureEncoder,
    TabularTokenEncoder,
    TabularEncoder,
    GEDS_GPU,
    OEMT,
    iStructTab,
)

__all__ = [
    "set_seed",
    "ImageFeatureEncoder",
    "TabularTokenEncoder",
    "TabularEncoder",
    "GEDS_GPU",
    "OEMT",
    "iStructTab",
]

__version__ = "0.1.0"