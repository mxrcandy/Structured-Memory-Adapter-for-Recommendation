import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recipes.train_qwen3 import _adapter_alpha_init, _distributed_any_rank_exhausted


class TestDistributedAnyRankExhausted(unittest.TestCase):
    def test_single_process_uses_local_flag(self):
        with patch("recipes.train_qwen3.dist.is_available", return_value=True), \
             patch("recipes.train_qwen3.dist.is_initialized", return_value=False):
            self.assertFalse(_distributed_any_rank_exhausted(False, torch.device("cpu")))
            self.assertTrue(_distributed_any_rank_exhausted(True, torch.device("cpu")))

    def test_distributed_all_reduce_max(self):
        def fake_all_reduce(tensor, op=None):
            tensor.fill_(1)

        with patch("recipes.train_qwen3.dist.is_available", return_value=True), \
             patch("recipes.train_qwen3.dist.is_initialized", return_value=True), \
             patch("recipes.train_qwen3.dist.all_reduce", side_effect=fake_all_reduce):
            self.assertTrue(_distributed_any_rank_exhausted(False, torch.device("cpu")))


class TestAdapterAlphaConfigCompatibility(unittest.TestCase):
    def test_old_remote_config_without_alpha_field_uses_default(self):
        self.assertEqual(_adapter_alpha_init(object()), 0.1)

    def test_configured_alpha_initial_value_is_preserved(self):
        self.assertEqual(
            _adapter_alpha_init(SimpleNamespace(adapter_alpha_init=0.125)),
            0.125,
        )


if __name__ == "__main__":
    unittest.main()
