"""Historical document queries with distinct public-knowledge and system-history modes.

Versions are selected by the latest eligible timestamp per logical key. Ties
raise an explicit ambiguity. Supersession fields are validated annotations;
they do not independently truncate visibility intervals. Queries never mutate
stored versions. Exact availability timestamps must be established upstream;
this small export does not implement uncertain-availability envelopes."""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import (
    Callable,
    Generic,
    Hashable,
    Iterable,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from auditable_ai.errors import AmbiguousRevisionError, TemporalIntegrityError


class AsOfMode(Enum):
    """Choose public-source history or this system's recording history explicitly."""

    RESEARCH = "RESEARCH_AS_OF"
    OPERATIONAL_REPLAY = "OPERATIONAL_REPLAY_AS_OF"


@runtime_checkable
class TemporalRecord(Protocol):
    @property
    def publicly_available_at(self) -> dt.datetime: ...

    @property
    def recorded_at(self) -> dt.datetime: ...


T = TypeVar("T", bound=TemporalRecord)
Clock = Callable[[], dt.datetime]


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def timestamp_for_mode(record: TemporalRecord, mode: AsOfMode) -> dt.datetime:
    if mode is AsOfMode.RESEARCH:
        return record.publicly_available_at
    if mode is AsOfMode.OPERATIONAL_REPLAY:
        return record.recorded_at
    raise ValueError(f"unknown AsOfMode: {mode!r}")


def _pick_unambiguous_winner(
    candidates: list[T],
    timestamp_fn: Callable[[T], dt.datetime],
    *,
    key: Hashable,
    context_label: str,
) -> T:
    max_ts = max((timestamp_fn(v) for v in candidates))
    winners = [v for v in candidates if timestamp_fn(v) == max_ts]
    if len(winners) > 1:
        raise AmbiguousRevisionError(
            f"{len(winners)} versions for key {key!r} are tied for the latest {context_label} timestamp ({max_ts}); refusing to pick one by insertion order or identity. Tied versions: {winners!r}"
        )
    return winners[0]


def resolve_current_per_key(
    records: Iterable[T], key_fn: Callable[[T], Hashable], as_of: dt.datetime, mode: AsOfMode
) -> list[T]:
    """Resolve each logical lineage from immutable versions, independently of insertion order.

    A winning timestamp tie is an error. Supersession annotations are not
    interval filters; only mode-specific version timestamps determine selection."""
    grouped: dict[Hashable, list[T]] = {}
    for record in records:
        grouped.setdefault(key_fn(record), []).append(record)
    current: list[T] = []
    for key, versions in grouped.items():
        candidates = [v for v in versions if timestamp_for_mode(v, mode) <= as_of]
        if not candidates:
            continue
        current.append(
            _pick_unambiguous_winner(
                candidates, lambda v: timestamp_for_mode(v, mode), key=key, context_label=mode.value
            )
        )
    return current


def require_dependent_not_before_dependency(
    *,
    dependent_publicly_available_at: dt.datetime,
    dependent_recorded_at: dt.datetime,
    dependency_publicly_available_at: dt.datetime,
    dependency_recorded_at: dt.datetime,
    dependent_label: str,
    dependency_label: str,
) -> None:
    """Check availability and recording independently across a provenance edge."""
    if dependent_publicly_available_at < dependency_publicly_available_at:
        raise TemporalIntegrityError(
            f"{dependent_label}'s publicly_available_at ({dependent_publicly_available_at}) precedes {dependency_label}'s publicly_available_at ({dependency_publicly_available_at}); it cannot have been knowable to research before the evidence it depends on was"
        )
    if dependent_recorded_at < dependency_recorded_at:
        raise TemporalIntegrityError(
            f"{dependent_label}'s recorded_at ({dependent_recorded_at}) precedes {dependency_label}'s recorded_at ({dependency_recorded_at}); this engine cannot have recorded it before recording the evidence it depends on"
        )


def require_timezone_aware(name: str, value: Optional[dt.datetime]) -> None:
    if value is not None and value.tzinfo is None:
        raise TemporalIntegrityError(
            f"{name} must be timezone-aware; got a naive datetime {value!r}"
        )


def require_not_in_future(name: str, value: dt.datetime, clock: Clock) -> None:
    """Refuse a recording time later than the writer's clock; backdating stays undetectable."""
    now = clock()
    require_timezone_aware("clock()", now)
    if value > now:
        raise TemporalIntegrityError(
            f"{name} ({value}) is later than the clock ({now}); a record cannot be written before its own recording time"
        )


def validate_temporal_fields(
    *,
    ingested_at: dt.datetime,
    recorded_at: dt.datetime,
    publicly_available_at: Optional[dt.datetime] = None,
    superseded_publicly_at: Optional[dt.datetime] = None,
    superseded_in_system_at: Optional[dt.datetime] = None,
) -> None:
    """Validate aware timestamps and ordering without forbidding late historical backfills."""
    for name, value in (
        ("ingested_at", ingested_at),
        ("recorded_at", recorded_at),
        ("publicly_available_at", publicly_available_at),
        ("superseded_publicly_at", superseded_publicly_at),
        ("superseded_in_system_at", superseded_in_system_at),
    ):
        require_timezone_aware(name, value)
    if ingested_at > recorded_at:
        raise TemporalIntegrityError(
            f"ingested_at ({ingested_at}) must be <= recorded_at ({recorded_at}); a record cannot be persisted before it was ingested"
        )
    if superseded_publicly_at is not None and publicly_available_at is not None:
        if superseded_publicly_at < publicly_available_at:
            raise TemporalIntegrityError(
                f"superseded_publicly_at ({superseded_publicly_at}) precedes this record's own publicly_available_at ({publicly_available_at}); a record cannot be superseded before it was even available"
            )
    if superseded_in_system_at is not None and superseded_in_system_at < recorded_at:
        raise TemporalIntegrityError(
            f"superseded_in_system_at ({superseded_in_system_at}) precedes this record's own recorded_at ({recorded_at}); a record cannot be superseded before it was recorded"
        )


class AppendOnlyStore(Generic[T]):
    """In-memory immutable-version collection sharing the historical resolver."""

    def __init__(self, key_fn: Callable[[T], Hashable]) -> None:
        self._key_fn = key_fn
        self._records: list[T] = []

    def add(self, record: T) -> None:
        self._records.append(record)

    def all(self) -> list[T]:
        return list(self._records)

    def history_for_key(self, key: Hashable) -> list[T]:
        return [r for r in self._records if self._key_fn(r) == key]

    def as_of(self, as_of: dt.datetime, mode: AsOfMode) -> list[T]:
        return resolve_current_per_key(self._records, self._key_fn, as_of, mode)

    def as_of_matching(
        self, as_of: dt.datetime, mode: AsOfMode, predicate: Callable[[T], bool]
    ) -> list[T]:
        scoped = [r for r in self._records if predicate(r)]
        return resolve_current_per_key(scoped, self._key_fn, as_of, mode)
