from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass(frozen=True)
class QueryIdentity:
    session_id: str
    query_id: str


class QueryStore(Protocol):
    """Persistence interface for query traces.

    Implementations must be best-effort: failures in a persistence backend
    should never break the main query path.
    """

    def write_event(self, ident: QueryIdentity, event_type: str, payload: Dict[str, Any]) -> None:
        ...

    def write_snapshot(self, ident: QueryIdentity, payload: Dict[str, Any]) -> None:
        ...


class NoopQueryStore:
    def write_event(self, ident: QueryIdentity, event_type: str, payload: Dict[str, Any]) -> None:
        return

    def write_snapshot(self, ident: QueryIdentity, payload: Dict[str, Any]) -> None:
        return
