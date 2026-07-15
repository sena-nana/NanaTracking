import pytest
import torch

from nana_tracking.contracts import AdapterContract
from nana_tracking.personalization import AffineResidualAdapter, ensure_adapter_compatible


def test_adapter_starts_as_identity_residual() -> None:
    adapter = AffineResidualAdapter(3)
    values = torch.tensor([[1.0, 2.0, 3.0]])
    torch.testing.assert_close(adapter(values), values)


def test_adapter_revision_mismatch_fails_safe() -> None:
    contract = AdapterContract(
        adapter_type="affine-residual",
        base_model_family="face-basic",
        base_model_version="1.0.0",
        feature_revision="features-v1",
    )
    with pytest.raises(ValueError, match="incompatible"):
        ensure_adapter_compatible(
            contract,
            model_family="face-basic",
            model_version="1.0.0",
            feature_revision="features-v2",
        )
