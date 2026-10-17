# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


"""The standalone API, with pipeline imports deferred until requested or used."""

from pyppbox import ppb as _ppb

__all__ = _ppb.__all__


def __getattr__(name):
    if name in __all__:
        value = getattr(_ppb, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
