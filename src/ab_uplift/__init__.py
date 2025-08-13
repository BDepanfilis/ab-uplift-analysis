# src/ab_uplift/__init__.py

from importlib.metadata import version, PackageNotFoundError

# Try both dist names; fall back to dev version
try:
    __version__ = version("ab-uplift")
except PackageNotFoundError:
    try:
        __version__ = version("ab_uplift")
    except PackageNotFoundError:
        __version__ = "0.0.0"

# Expose module names but DO NOT import them here
__all__ = [
    "config",
    "data_loader",
    "power",
    "ab_stats",
    "cupac",
    "causal",
    "uplift",
]
