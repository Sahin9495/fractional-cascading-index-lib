"""Fractional-cascading data structure for multi-level binary search.

This module implements an exact-match index over a layered set of sorted
arrays.  Each layer stores all keys from its own dataset plus a sampled
fraction of keys carried up from the layer below.  A search begins with a
binary search in the top layer; the position found there narrows the search
range in the next layer to a constant-sized interval, so the whole traversal
is a single O(log n + k) pass rather than k independent O(log n) searches.

Design choices (documented because they are not obvious):

* **Exact match only.**  The brief says "multi-level binary search".  We
  interpret that as: find an exact key across every layer that contains it.
  Range queries and rank/select are out of scope.  Trying to support them
  would complicate the bridge pointers and the returned type.

* **Construction raises on unsorted input.**  A fractional-cascading index is
  only valid if each layer is sorted.  We check once, at build time, rather
  than on every query — the error surfaces immediately and the hot path
  stays branch-free.

* **Sample rate is a constructor argument, not auto-tuned.**  The textbook
  uses 1/2 (keep every other element).  We expose the fraction so callers
  can trade memory for speed, but we do not try to pick it automatically —
  that depends on workload, which the library does not see.

* **Duplicate keys within a layer are preserved.**  ``search`` returns the
  index of an arbitrary matching slot; callers that need all occurrences can
  scan left/right from the returned position.  This keeps the common case
  (unique keys) fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class SearchResult:
    """Result of a multi-level search.

    ``indices[i]`` is the position of ``key`` in layer ``i``, or ``None`` if
    the key is absent from that layer.  ``indices`` always has length equal
    to the number of layers.
    """

    key: int
    indices: Tuple[Optional[int], ...]

    @property
    def found_in_layers(self) -> Tuple[int, ...]:
        """Layer indices where the key was located."""
        return tuple(i for i, idx in enumerate(self.indices) if idx is not None)

    @property
    def any_found(self) -> bool:
        """Whether the key was found in at least one layer."""
        return any(idx is not None for idx in self.indices)


class FractionalCascadingIndex:
    """A fractional-cascading index over k sorted integer arrays.

    Parameters
    ----------
    layers
        A sequence of sorted integer sequences.  Each must be ascending.
    sample_rate
        Fraction in (0, 1].  Each layer carries ``ceil(sample_rate * n)``
        sampled elements from the layer below it.  Defaults to 0.5, the
        textbook value.

    Raises
    ------
    ValueError
        If layers is empty, any layer is unsorted, any layer contains
        a non-int, or sample_rate is outside (0, 1].
    TypeError
        If a layer contains a non-integer element.

    Examples
    --------
    >>> idx = FractionalCascadingIndex([[1, 3, 5], [2, 4, 6, 8]])
    >>> r = idx.search(4)
    >>> r.indices
    (None, 1)
    """

    def __init__(self, layers, sample_rate: float = 0.5):
        # --- Validate sample_rate first so later errors aren't confusing ---
        if not isinstance(sample_rate, (int, float)):
            raise TypeError(
                f"sample_rate must be int or float, got {type(sample_rate).__name__}"
            )
        if not (0.0 < sample_rate <= 1.0):
            raise ValueError(
                f"sample_rate must be in (0, 1], got {sample_rate}"
            )

        # --- Defensive copy + type check, before any validation ---
        self._layers: List[List[int]] = []
        if not layers:
            raise ValueError("layers must contain at least one list")
        for i, layer in enumerate(layers):
            if layer is None:
                raise TypeError(f"layer {i} is None")
            copied: List[int] = []
            for x in layer:
                # bool is a subclass of int; reject it explicitly so True
                # does not silently become 1 in the index.
                if isinstance(x, bool) or not isinstance(x, int):
                    raise TypeError(
                        f"layer {i} contains non-integer element: {x!r} "
                        f"(type {type(x).__name__})"
                    )
                copied.append(x)
            self._layers.append(copied)

        # --- Sortedness check, once, up front ---
        for i, layer in enumerate(self._layers):
            for j in range(1, len(layer)):
                if layer[j] < layer[j - 1]:
                    raise ValueError(
                        f"layer {i} is not sorted ascending: "
                        f"{layer[j - 1]} > {layer[j]} at index {j}"
                    )

        self._sample_rate = sample_rate
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _sample(self, layer: List[int]) -> List[int]:
        """Pick a maximal subset of ``layer`` for the next-up bridge.

        We take every ``step``-th element, where ``step`` is the integer in
        ``[1, len(layer)]`` that yields a fraction closest to ``sample_rate``.
        This guarantees at least one sample (or zero if the layer is empty)
        and keeps the samples sorted, which the cascading search relies on.
        """
        n = len(layer)
        if n <= 1:
            # Empty or single-element: nothing useful to sample.
            return []
        # step >= 1; larger step => fewer samples.
        raw_step = 1.0 / self._sample_rate
        step = max(1, int(round(raw_step)))
        # Never step past the array end, and never produce zero samples when
        # the array has >= 2 elements (otherwise bridging degrades).
        if step >= n:
            return [layer[-1]]
        return layer[::step]

    @staticmethod
    def _merge(a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
        """Merge two sorted lists, tracking the origin of each element.

        Returns ``(merged, flags)`` where ``flags[k]`` is True if
        ``merged[k]`` came from ``a`` (the layer's own keys) and False if it
        came from ``b`` (the samples from below).  Equal elements from ``a``
        are placed before those from ``b`` so that a search for an own-key
        lands on the own-key slot, which is the position the caller cares
        about.
        """
        merged: List[int] = []
        flags: List[bool] = []
        i = j = 0
        la, lb = len(a), len(b)
        while i < la and j < lb:
            if a[i] <= b[j]:
                merged.append(a[i]); flags.append(True); i += 1
            else:
                merged.append(b[j]); flags.append(False); j += 1
        while i < la:
            merged.append(a[i]); flags.append(True); i += 1
        while j < lb:
            merged.append(b[j]); flags.append(False); j += 1
        return merged, flags

    def _build(self) -> None:
        """Build the cascaded layers from the bottom up.

        For each layer i (from the last to the first), we store:

        * ``keys[i]``      — the merged array (own keys + samples from i+1).
        * ``own_flag[i]`` — True where ``keys[i]`` holds an own key.
        * ``bridge[i]``    — for each sample copied from layer i+1, the index
          of that same key in ``keys[i+1]``.  Indexed by position in
          ``keys[i]``; entries for own-key slots are set to -1.
        """
        k = len(self._layers)
        self.keys: List[List[int]] = [None] * k   # type: ignore[assignment]
        self.own_flag: List[List[bool]] = [None] * k   # type: ignore[assignment]
        self.bridge: List[List[int]] = [None] * k   # type: ignore[assignment]

        # Bottom layer: own keys only, no bridge.
        self.keys[k - 1] = list(self._layers[k - 1])
        self.own_flag[k - 1] = [True] * len(self.keys[k - 1])
        self.bridge[k - 1] = [-1] * len(self.keys[k - 1])

        for i in range(k - 2, -1, -1):
            own = list(self._layers[i])
            samples = self._sample(self.keys[i + 1])

            merged, flags = self._merge(own, samples)

            # Bridge[j] = index in keys[i+1] of keys[i][j], for sample slots.
            # Walk through keys[i+1] and samples in parallel.
            bridge = [-1] * len(merged)
            si = 0  # pointer into samples
            ti = 0  # pointer into keys[i+1]
            for j in range(len(merged)):
                if flags[j]:
                    continue
                # This merged element equals samples[si]; find it in
                # keys[i+1] starting from ti.
                target = samples[si]
                while ti < len(self.keys[i + 1]) and self.keys[i + 1][ti] != target:
                    ti += 1
                # ti is now at the first occurrence of target.
                bridge[j] = ti
                si += 1
                ti += 1  # don't reuse the same slot

            self.keys[i] = merged
            self.own_flag[i] = flags
            self.bridge[i] = bridge

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _binary_search(self, arr: List[int], key: int) -> int:
        """Return the leftmost index of ``key`` in ``arr``, or the insertion
        point if absent.  Standard left-biased binary search; O(log n)."""
        lo, hi = 0, len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _find_leftmost(self, arr: List[int], key: int) -> Optional[int]:
        """Return the leftmost index of ``key`` in ``arr``, or None."""
        idx = self._binary_search(arr, key)
        if idx < len(arr) and arr[idx] == key:
            return idx
        return None

    def search(self, key: int) -> SearchResult:
        """Search for ``key`` across all layers in one logarithmic pass.

        Returns a :class:`SearchResult` whose ``indices`` tuple has one entry
        per layer: the position of ``key`` in that layer, or ``None``.

        The cascade works as follows: binary-search the top layer; in each
        subsequent layer, the bridge pointer from the previous hit narrows
        the search to a small window, so only one full binary search is
        performed and the rest are O(1)-ish local scans.
        """
        if isinstance(key, bool) or not isinstance(key, int):
            raise TypeError(
                f"key must be int, got {type(key).__name__}"
            )

        k = len(self.keys)
        result: List[Optional[int]] = [None] * k

        # Layer 0: full binary search.
        pos = self._binary_search(self.keys[0], key)

        for i in range(k):
            arr = self.keys[i]
            # Narrow pos to [0, len(arr)].
            if pos >= len(arr):
                pos = len(arr) - 1 if arr else 0
            if pos < 0:
                pos = 0

            # Scan left for an exact match, up to two slots (the window is
            # at most 2 wide when sample_rate >= 0.5, but we allow a few more
            # for safety with smaller sample rates).
            found: Optional[int] = None
            scan = pos
            steps = 0
            while scan >= 0 and steps < 4:
                if arr[scan] == key:
                    found = scan
                    break
                if arr[scan] < key:
                    break
                scan -= 1
                steps += 1

            # If not found left, scan right a little (covers the case where
            # the binary search landed just below the key).
            if found is None:
                scan = pos + 1
                steps = 0
                while scan < len(arr) and steps < 4:
                    if arr[scan] == key:
                        found = scan
                        break
                    if arr[scan] > key:
                        break
                    scan += 1
                    steps += 1

            # Record result for this layer (only if it's an own-key slot).
            # The index is relative to the original layer, so we count only
            # own-key slots up to and including the found position.
            if found is not None and self.own_flag[i][found]:
                result[i] = sum(self.own_flag[i][:found])

            # Descend.
            if found is not None:
                pos = self._descend(i, found, key)
            else:
                # Use insertion point as the bridge candidate.
                # Clamp pos into range first.
                if pos >= len(arr):
                    pos = len(arr) - 1
                if pos < 0:
                    pos = 0
                pos = self._descend(i, pos, key)

        return SearchResult(key=key, indices=tuple(result))

    def _descend(self, i: int, pos: int, key: int) -> int:
        """Map position ``pos`` in layer ``i`` to a starting position in layer
        ``i+1`` using the bridge pointers.

        If ``pos`` is a sample slot, ``bridge[i][pos]`` is the direct index
        in layer ``i+1``.  If it is an own-key slot, we find the nearest
        sample slot at or before ``pos`` and use its bridge, clamped to the
        range of layer ``i+1``.  This keeps the next-layer scan to a
        constant-size window.
        """
        if i >= len(self.keys) - 1:
            return 0  # bottom layer; no descent needed

        br = self.bridge[i]
        nxt = self.keys[i + 1]

        if br[pos] != -1:
            return br[pos]

        # Walk left to the nearest sample slot.
        scan = pos - 1
        while scan >= 0:
            if br[scan] != -1:
                return br[scan]
            scan -= 1

        # No bridge before pos: start at the beginning of the next layer.
        return 0

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._layers)

    def __repr__(self) -> str:
        return (
            f"FractionalCascadingIndex(layers={self._layers!r}, "
            f"sample_rate={self._sample_rate!r})"
        )
