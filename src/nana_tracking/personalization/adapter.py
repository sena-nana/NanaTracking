"""Small residual adapter example for the personalization contract."""

from torch import Tensor, nn

from nana_tracking.contracts import AdapterContract


class AffineResidualAdapter(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(Tensor(features).fill_(1.0))
        self.offset = nn.Parameter(Tensor(features).zero_())

    def forward(self, values: Tensor) -> Tensor:
        return values + (values * (self.scale - 1.0) + self.offset)


def ensure_adapter_compatible(
    contract: AdapterContract,
    *,
    model_family: str,
    model_version: str,
    feature_revision: str,
) -> None:
    expected = (model_family, model_version, feature_revision)
    actual = (
        contract.base_model_family,
        contract.base_model_version,
        contract.feature_revision,
    )
    if actual != expected:
        raise ValueError(
            "adapter is incompatible with the active base model or feature contract: "
            f"expected={expected!r}, actual={actual!r}"
        )
