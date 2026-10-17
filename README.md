# fractional-cascading-index-lib

A small, dependency-free Python library that indexes several sorted integer arrays at once using fractional cascading, so a key can be looked up across every layer in a single logarithmic pass instead of one binary search per layer.

## Usage

```python
from fractional_cascading_index_lib import FractionalCascadingIndex, SearchResult

idx = FractionalCascadingIndex(
    [[1, 5, 9], [0, 2, 4, 6, 8, 10, 12]],
    sample_rate=0.5,
)

result: SearchResult = idx.search(8)
print(result.indices)          # (None, 5)
print(result.found_in_layers)  # (1,)
print(result.any_found)        # True
```

`FractionalCascadingIndex(layers, sample_rate=0.5)` builds the structure; `search(key)` returns a `SearchResult` with `key`, `indices` (tuple, one entry per layer, `None` where absent), `found_in_layers`, and `any_found`.

## Why this exists

The problem: you have *k* sorted arrays and you want to know which of them contain a given key. Doing *k* independent binary searches costs O(k log n). Fractional cascading layers the arrays so that each one carries a sampled subset of the next layer's elements; a single binary search at the top, followed by constant-work descents, answers all *k* queries in O(log n + k). The trade-off is extra memory (each layer stores samples from below) and a more complex build, in exchange for the faster multi-query path.

This library implements **exact-match search only**. Range queries, rank, and select are out of scope; supporting them would change the bridge-pointer structure and the return type.

## Edge cases worth knowing

- Each input layer must be sorted ascending; construction raises `ValueError` otherwise.
- Keys are integers. Floats and `bool` are rejected (`bool` because it is a subclass of `int` in Python but almost never what you want in a numeric index).
- Duplicate keys within a layer are preserved; `search` returns *some* matching slot, not necessarily the first or last. If you need all occurrences, scan left and right from the returned index.
- An empty layer is allowed — it simply contributes no keys and always reports `None`.

## Running the tests

```
PYTHONPATH=src python -m unittest discover -s tests
` ``
