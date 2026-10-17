import unittest

from fractional_cascading_index_lib import FractionalCascadingIndex, SearchResult


class TestConstruction(unittest.TestCase):
    """Validation performed at build time."""

    def test_empty_layers_raises(self):
        with self.assertRaises(ValueError):
            FractionalCascadingIndex([])

    def test_unsorted_layer_raises(self):
        with self.assertRaises(ValueError):
            FractionalCascadingIndex([[1, 3, 2]])

    def test_non_integer_element_raises(self):
        with self.assertRaises(TypeError):
            FractionalCascadingIndex([[1, 2, 3.0]])

    def test_bool_rejected(self):
        # bool is a subclass of int; we reject it so True != 1 in the index.
        with self.assertRaises(TypeError):
            FractionalCascadingIndex([[1, True, 3]])

    def test_invalid_sample_rate_zero(self):
        with self.assertRaises(ValueError):
            FractionalCascadingIndex([[1, 2]], sample_rate=0.0)

    def test_invalid_sample_rate_negative(self):
        with self.assertRaises(ValueError):
            FractionalCascadingIndex([[1, 2]], sample_rate=-0.5)

    def test_invalid_sample_rate_too_large(self):
        with self.assertRaises(ValueError):
            FractionalCascadingIndex([[1, 2]], sample_rate=1.5)

    def test_single_layer(self):
        idx = FractionalCascadingIndex([[1, 3, 5, 7]])
        r = idx.search(5)
        self.assertEqual(r.indices, (2,))

    def test_empty_layer_allowed(self):
        # An empty layer is structurally fine; it just contributes no keys.
        idx = FractionalCascadingIndex([[], [1, 2, 3]])
        r = idx.search(2)
        self.assertEqual(r.indices, (None, 1))


class TestSearch(unittest.TestCase):
    """Behaviour of the search path."""

    def test_key_present_in_all_layers(self):
        idx = FractionalCascadingIndex([[2, 4, 6], [1, 2, 3, 4, 5, 6, 7]])
        r = idx.search(6)
        self.assertEqual(r.indices, (2, 5))

    def test_key_present_in_one_layer(self):
        idx = FractionalCascadingIndex([[2, 4, 6], [1, 3, 5, 7]])
        r = idx.search(6)
        self.assertEqual(r.indices, (2, None))

    def test_key_absent_everywhere(self):
        idx = FractionalCascadingIndex([[1, 3, 5], [2, 4, 6]])
        r = idx.search(99)
        self.assertEqual(r.indices, (None, None))
        self.assertFalse(r.any_found)

    def test_key_below_all(self):
        idx = FractionalCascadingIndex([[10, 20], [5, 10, 15, 20]])
        r = idx.search(1)
        self.assertEqual(r.indices, (None, None))

    def test_duplicate_within_layer_returns_a_match(self):
        # We don't promise which slot; we only promise a match is reported.
        idx = FractionalCascadingIndex([[1, 3, 3, 3, 5]])
        r = idx.search(3)
        self.assertIn(r.indices[0], (1, 2, 3))

    def test_three_layer_cascade(self):
        idx = FractionalCascadingIndex(
            [[100, 200, 300], [50, 100, 150, 200, 250, 300],
             [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
        )
        r = idx.search(100)
        self.assertEqual(r.indices, (0, 1, 9))

    def test_sample_rate_one(self):
        # sample_rate=1 keeps every element from below; still correct.
        idx = FractionalCascadingIndex(
            [[1, 5, 9], [2, 4, 6, 8]], sample_rate=1.0
        )
        r = idx.search(6)
        self.assertEqual(r.indices, (None, 2))

    def test_small_sample_rate(self):
        # A small sample rate stresses the local scan in the next layer.
        idx = FractionalCascadingIndex(
            [[1, 5, 9], list(range(0, 40, 2))],
            sample_rate=0.25,
        )
        r = idx.search(30)
        self.assertEqual(r.indices, (None, 15))


class TestResultType(unittest.TestCase):
    """SearchResult properties and shape."""

    def test_indices_length_equals_layer_count(self):
        idx = FractionalCascadingIndex([[1], [2], [3]])
        r = idx.search(1)
        self.assertEqual(len(r.indices), 3)

    def test_found_in_layers_property(self):
        idx = FractionalCascadingIndex([[1, 2], [2, 3]])
        r = idx.search(2)
        self.assertEqual(r.found_in_layers, (0, 1))

    def test_any_found_true(self):
        idx = FractionalCascadingIndex([[1, 2], [3, 4]])
        self.assertTrue(idx.search(2).any_found)

    def test_any_found_false(self):
        idx = FractionalCascadingIndex([[1, 2], [3, 4]])
        self.assertFalse(idx.search(99).any_found)

    def test_search_result_is_frozen(self):
        idx = FractionalCascadingIndex([[1, 2]])
        r = idx.search(1)
        with self.assertRaises(Exception):
            r.key = 999  # dataclass(frozen=True)

    def test_indices_is_tuple(self):
        idx = FractionalCascadingIndex([[1, 2]])
        r = idx.search(1)
        self.assertIsInstance(r.indices, tuple)


class TestInputNonMutation(unittest.TestCase):
    """The library must not mutate caller-owned lists."""

    def test_input_lists_not_mutated(self):
        original = [[1, 3, 5], [2, 4, 6, 8]]
        snapshot = [list(layer) for layer in original]
        FractionalCascadingIndex(original)
        self.assertEqual(original, snapshot)


class TestDunder(unittest.TestCase):
    """len and repr are part of the contract."""

    def test_len(self):
        idx = FractionalCascadingIndex([[1], [2], [3]])
        self.assertEqual(len(idx), 3)

    def test_repr(self):
        idx = FractionalCascadingIndex([[1, 2]], sample_rate=0.5)
        s = repr(idx)
        self.assertIn("FractionalCascadingIndex", s)
        self.assertIn("sample_rate=0.5", s)


class TestKeyValidation(unittest.TestCase):
    """search() validates its key argument."""

    def test_float_key_rejected(self):
        idx = FractionalCascadingIndex([[1, 2, 3]])
        with self.assertRaises(TypeError):
            idx.search(2.0)

    def test_bool_key_rejected(self):
        idx = FractionalCascadingIndex([[1, 2, 3]])
        with self.assertRaises(TypeError):
            idx.search(True)


if __name__ == "__main__":
    unittest.main()
