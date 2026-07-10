"""Utilities for rendering and training prompt-based and overlapping segmentation models (like SAM). """

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("segment-everything")
except PackageNotFoundError:
    __version__ = "unknown"

__author__ = "Brian Northan"
__email__ = "bnorthan@gmail.com"
