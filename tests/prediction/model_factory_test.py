import pytest
import sys
import os

# 1. PATH SETUP
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if src_path not in sys.path:
    sys.path.append(src_path)

from prediction.model_factory import MODEL_REGISTRY

# 2. REGISTRY CONTRACT TEST (Your existing test)
@pytest.mark.parametrize("model_key", MODEL_REGISTRY.keys())
def test_registry_contract(model_key):
    """Checks if each model entry in the registry has the required interface."""
    spec = MODEL_REGISTRY[model_key]
    assert "loader" in spec
    assert "input_adapter" in spec
    assert "inference_fn" in spec
    assert "output_adapter" in spec
    for component in spec.values():
        assert callable(component) or component is None


# We do not test loader and inference functions here, as they involve external dependencies and are simple wrappers around library calls. Instead, we focus on the adapters which contain custom logic.