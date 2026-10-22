import unittest
from types import SimpleNamespace
import torch
from onerec_llm.models.memory_adapter.prototype_common import hash_tokens
from onerec_llm.models.memory_adapter.structure import StructureMemoryBranch


class PrototypeTests(unittest.TestCase):
    def test_hash_matches_runtime_int64(self):
        stub = SimpleNamespace(order_salts=(0x9E3779B1 + 0x85EBCA77 * 2,),
                               head_salts=(0xC2B2AE3D,), size=65521)
        tokens = (151700, 151800, 151900, 152000)
        expected = StructureMemoryBranch.hash_ids(stub, torch.tensor([tokens]), 0, 0).item()
        self.assertEqual(hash_tokens(tokens, 2, 0, 65521, order_idx=0), expected)


if __name__ == '__main__':
    unittest.main()
