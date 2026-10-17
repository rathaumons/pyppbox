# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Compatibility imports for the Torchreid model catalog."""

from pyppbox.config.torchreidmodels import (
    TorchreidModel, TorchreidModelDict, default_model_dict_yaml,
)

__all__ = ['TorchreidModel', 'TorchreidModelDict', 'default_model_dict_yaml']
