from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
import json
import time

from nl_analytics.logging.logger import get_logger

log = get_logger("export.saved_queries")

@dataclass(frozen=True)
class SavedQuery:
    id: str
    created_at: float
    mode: str
    question: str
    plan: Dict[str, Any]

def save_query(out_dir: str, mode: str, question: str, plan: Dict[str, Any]) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    qid = str(int(time.time() * 1000))
    payload = {"id": qid, "created_at": time.time(), "mode": mode, "question": question, "plan": plan}
    path = Path(out_dir) / f"{qid}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Saved query", extra={"id": qid, "path": str(path)})
    return qid

def list_queries(out_dir: str) -> List[SavedQuery]:
    p = Path(out_dir)
    if not p.exists():
        return []
    out: List[SavedQuery] = []
    for f in sorted(p.glob("*.json"), reverse=True):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
            out.append(SavedQuery(**payload))
        except Exception:
            continue
    return out

def load_query(out_dir: str, qid: str) -> SavedQuery:
    path = Path(out_dir) / f"{qid}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SavedQuery(**payload)
