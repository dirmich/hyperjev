import unittest

from hyperjev.cache import (
    BoundedCache,
    normalize_state,
    question_cache_key,
    result_cache_key,
    state_cache_key,
)


class CacheTests(unittest.TestCase):
    def test_keys_are_deterministic_and_version_scoped(self) -> None:
        self.assertEqual(normalize_state("  a\n b  "), "a b")
        first = state_cache_key("model-1", " a\n b ")
        equivalent = state_cache_key("model-1", "a b")
        changed_model = state_cache_key("model-2", "a b")
        self.assertEqual(first, equivalent)
        self.assertNotEqual(first, changed_model)
        question = question_cache_key("memory.type@1", "type?", ("fact", "none"))
        result = result_cache_key("model-1", "cal-1", first, question)
        self.assertEqual(len(result), 64)
        self.assertNotIn("a b", result)

    def test_bounded_cache_evicts_oldest_entry(self) -> None:
        cache: BoundedCache[str] = BoundedCache(max_entries=2)
        cache.put("a", "A")
        cache.put("b", "B")
        self.assertEqual(cache.get("a"), "A")
        cache.put("c", "C")
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), "A")
        self.assertEqual(len(cache), 2)


if __name__ == "__main__":
    unittest.main()
