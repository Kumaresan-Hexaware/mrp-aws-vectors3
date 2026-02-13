from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import time

from nl_analytics.logging.logger import get_logger
from .types import QueryStore, QueryIdentity
from .utils import json_safe


log = get_logger("store.file")


class FileQueryStore(QueryStore):
    """Persist query traces into local files.

    Writes:
      - events: <...>_events.jsonl
      - snapshot: <...>_snapshot.json
    """

    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _base(self, ident: QueryIdentity) -> Path:
        return self.out_dir / f"{ident.session_id}__{ident.query_id}"

    def write_event(self, ident: QueryIdentity, event_type: str, payload: Dict[str, Any]) -> None:
        try:
            base = self._base(ident)
            line = {
                "ts": time.time(),
                "event_type": event_type,
                "session_id": ident.session_id,
                "query_id": ident.query_id,
                "payload": json_safe(payload),
            }
            with (base.with_name(base.name + "_events.jsonl")).open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning("Failed to write file event", extra={"error": str(e)})

    def write_snapshot(self, ident: QueryIdentity, payload: Dict[str, Any]) -> None:
        try:
            base = self._base(ident)
            snap = {
                "session_id": ident.session_id,
                "query_id": ident.query_id,
                "payload": json_safe(payload),
            }
            (base.with_name(base.name + "_snapshot.json")).write_text(
                json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            log.warning("Failed to write file snapshot", extra={"error": str(e)})
