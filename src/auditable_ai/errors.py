"""Explicit failures for temporal integrity, provenance and model output."""

from __future__ import annotations


class PointInTimeError(Exception):
    pass


class TemporalIntegrityError(PointInTimeError, ValueError):
    """Timing assertions violate the declared record or evidence-chain contract."""

    pass


class ProvenanceError(PointInTimeError, ValueError):
    """An evidence reference, version identity, or content fingerprint is invalid."""

    pass


class AmbiguousRevisionError(TemporalIntegrityError):
    """Multiple eligible versions share a winning timestamp; no arbitrary winner is chosen."""

    pass


class ExtractionValidationError(ValueError):
    """An extraction response cannot be validated as a batch."""
