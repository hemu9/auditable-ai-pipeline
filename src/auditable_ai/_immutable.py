"""JSON-like immutable values and stable content fingerprints."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping


def deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple((deep_freeze(v) for v in value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        f"expected a JSON-like value (str/int/float/bool/None/dict/list), got {type(value).__name__}"
    )


def to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value


def canonical_hash(value: Any) -> str:
    canonical = json.dumps(to_plain(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
