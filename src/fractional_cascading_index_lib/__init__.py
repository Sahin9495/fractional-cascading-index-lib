# fractional-cascading-index-lib
#
# Public surface: FractionalCascadingIndex, SearchResult.
# The heavy lifting lives in core.py; this module just re-exports.

from .core import FractionalCascadingIndex, SearchResult

__all__ = ["FractionalCascadingIndex", "SearchResult"]
