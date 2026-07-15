"""Versioned personalization adapters kept separate from the base model."""

from nana_tracking.personalization.adapter import (
    AffineResidualAdapter,
    ensure_adapter_compatible,
)

__all__ = ["AffineResidualAdapter", "ensure_adapter_compatible"]
