import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestAllowedMissingParameterInitialization(unittest.TestCase):
    def test_memory_adapter_norm_weights_initialize_to_one(self):
        try:
            import torch
            from onerec_llm.training.distributed import initialize_allowed_missing_parameter
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is not installed in this Python environment: {exc}")

        for name in [
            "model.layers.1.memory_adapter.query_norm.weight",
            "model.layers.1.memory_adapter.key_norm.weight",
            "model.layers.1.memory_adapter.conv_norm.weight",
        ]:
            tensor = initialize_allowed_missing_parameter(name, (8,))
            self.assertTrue(torch.equal(tensor, torch.ones_like(tensor)))

    def test_lora_b_and_other_one_dimensional_params_stay_zero(self):
        try:
            import torch
            from onerec_llm.training.distributed import initialize_allowed_missing_parameter
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is not installed in this Python environment: {exc}")

        for name in [
            "model.layers.1.self_attn.q_proj.lora_B",
            "model.layers.1.memory_adapter.some_bias",
        ]:
            tensor = initialize_allowed_missing_parameter(name, (8,))
            self.assertTrue(torch.equal(tensor, torch.zeros_like(tensor)))

    def test_memory_adapter_alpha_uses_configured_initial_value(self):
        try:
            import torch
            from onerec_llm.training.distributed import initialize_allowed_missing_parameter
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is not installed in this Python environment: {exc}")

        tensor = initialize_allowed_missing_parameter(
            "model.layers.1.memory_adapter.alpha_layer",
            (1,),
            memory_adapter_alpha_init=0.125,
        )

        self.assertTrue(torch.equal(tensor, torch.tensor([0.125])))


if __name__ == "__main__":
    unittest.main()
