# dapper/__init__.py
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "1.0"

__all__ = [
    "Domain",
    "ERA5Adapter",
    "FluxnetAdapter",
]

# For type checkers / IDEs only (doesn't execute at runtime)
if TYPE_CHECKING:  # pragma: no cover
    from dapper.domains.domain import Domain
    from dapper.met.adapters.era5 import ERA5Adapter
    from dapper.met.adapters.fluxnet import FluxnetAdapter

_LAZY = {
    "Domain": ("dapper.domains.domain", "Domain"),
    "ERA5Adapter": ("dapper.met.adapters.era5", "ERA5Adapter"),
    "FluxnetAdapter": ("dapper.met.adapters.fluxnet", "FluxnetAdapter"),
}

def __getattr__(name: str):
    try:
        mod_name, attr = _LAZY[name]
    except KeyError as e:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from e
    mod = import_module(mod_name)
    value = getattr(mod, attr)
    globals()[name] = value  # cache
    return value

def __dir__():
    return sorted(set(list(globals()) + list(_LAZY)))
