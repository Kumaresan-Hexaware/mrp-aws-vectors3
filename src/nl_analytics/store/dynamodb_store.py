from __future__ import annotations

from typing import Any, Dict, Optional
import time
from datetime import datetime, timezone
from decimal import Decimal

import boto3
from botocore.config import Config as BotoConfig

from nl_analytics.logging.logger import get_logger

from .types import QueryStore, QueryIdentity
from .utils import json_safe, scrub_large_strings, to_ddb_safe


log = get_logger("store.dynamodb")


class DynamoDBQueryStore(QueryStore):
    """Persist query traces to DynamoDB.

    Table requirements:
      - Partition key: configurable (default: session_id, type S)
      - Sort key: configurable (default: sk, type S)  [optional in compact mode]

    Modes:
      - detailed (default): one item per event + one snapshot item
      - payload / single / compact: ONE item per query (a single JSON-like payload)
      - single_row / one_row / row: ONE item per query, incrementally merged via UpdateItem
    """

    def __init__(
        self,
        table_name: str,
        region_name: str,
        endpoint_url: Optional[str] = None,
        ttl_attribute: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        pk_name: str = "session_id",
        sk_name: str = "sk",
        mode: str = "detailed",
    ):
        self.table_name = table_name
        self.ttl_attribute = ttl_attribute
        self.ttl_seconds = ttl_seconds

        self.pk_name = pk_name or "session_id"
        self.sk_name = sk_name or "sk"
        self.mode = (mode or "detailed").strip().lower().replace("-", "_")

        # Conservative retries to avoid noisy failures.
        cfg = BotoConfig(retries={"max_attempts": 3, "mode": "standard"})
        self.ddb = boto3.resource("dynamodb", region_name=region_name, endpoint_url=endpoint_url, config=cfg)
        self.table = self.ddb.Table(table_name)

        # Best-effort: align configured PK/SK with the real table schema.
        # This avoids "provided key element does not match schema" errors when users have PK-only tables.
        try:
            desc = self.table.meta.client.describe_table(TableName=table_name)
            key_schema = desc.get("Table", {}).get("KeySchema", [])
            key_names = [k.get("AttributeName") for k in key_schema if k.get("AttributeName")]
            if len(key_names) == 1:
                # PK-only table
                self.sk_name = ""
            elif len(key_names) >= 2 and self.sk_name and self.sk_name not in key_names:
                # User-configured SK doesn't match actual schema; disable SK to keep writes working.
                self.sk_name = ""
        except Exception:
            pass

    def _ttl(self) -> Optional[int]:
        if not self.ttl_attribute or not self.ttl_seconds:
            return None
        return int(time.time()) + int(self.ttl_seconds)

    def _pk_value(self, ident: QueryIdentity) -> str:
        # If PK name looks query-centric, store query_id there.
        pk = (self.pk_name or "session_id").strip().lower()
        if pk in ("query_id", "qid", "request_id", "event_id", "id"):
            return ident.query_id
        return ident.session_id

    def _sk_value(self, ident: QueryIdentity) -> Optional[str]:
        sk = (self.sk_name or "").strip()
        if not sk:
            return None
        # In payload/compact and single-row modes we store one item per query.
        if self.mode in ("payload", "single", "compact", "single_row", "one_row", "row"):
            return ident.query_id
        return None

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _iso_utc(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _key_for_query(self, ident: QueryIdentity) -> Dict[str, Any]:
        key: Dict[str, Any] = {self.pk_name: self._pk_value(ident)}
        skv = self._sk_value(ident)
        if skv is not None and (self.sk_name or "").strip():
            key[self.sk_name] = skv
        return key

    def _ensure_single_row_base(self, ident: QueryIdentity) -> None:
        """Ensure the base single-row item exists and has the required map parents."""
        now_iso = self._iso_utc()

        expr_names: Dict[str, str] = {
            "#qid": "query_id",
            "#sid": "session_id",
            "#ua": "updated_at",
            "#ca": "created_at",
            "#attempts": "attempts",
        }
        expr_vals: Dict[str, Any] = {
            ":qid": ident.query_id,
            ":sid": ident.session_id,
            ":ua": now_iso,
            ":ca": now_iso,
            ":empty_map": {},
        }

        # Only write base paths here (no nested children) to avoid overlapping paths.
        sets = [
            "#qid=:qid",
            "#sid=:sid",
            "#ua=:ua",
            "#ca=if_not_exists(#ca, :ca)",
            "#attempts=if_not_exists(#attempts, :empty_map)",
        ]

        ttl = self._ttl()
        if ttl is not None and self.ttl_attribute:
            expr_names["#ttl"] = self.ttl_attribute
            expr_vals[":ttl"] = int(ttl)
            sets.append("#ttl=:ttl")

        self.table.update_item(
            Key=self._key_for_query(ident),
            UpdateExpression="SET " + ", ".join(sets),
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_vals,
        )

    def _ensure_single_row_attempt(self, ident: QueryIdentity, attempt: int) -> None:
        """Ensure attempts[attempt] exists as a map with expected keys."""
        akey = str(int(attempt))
        expr_names: Dict[str, str] = {"#attempts": "attempts", "#a": akey}

        # Seed keys so the item always has the expected shape even on failures.
        default_attempt = {
            "raw_plan": None,
            "validated_plan": None,
            "sql": "",
            "db_type": None,
            "final": None,
            "error": None,
        }

        self.table.update_item(
            Key=self._key_for_query(ident),
            UpdateExpression="SET #attempts.#a = if_not_exists(#attempts.#a, :empty_attempt)",
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues={":empty_attempt": to_ddb_safe(default_attempt)},
        )

    def _update_single_row(self, ident: QueryIdentity, event_type: str, payload: Dict[str, Any]) -> None:
        """Merge events into a SINGLE DynamoDB item per query.

        DynamoDB doesn't allow updating a parent document path and its child path in the same request.
        So we:
          1) Ensure base parents exist (attempts map)
          2) Ensure attempts[attempt] exists (if attempt-based)
          3) Update only leaf paths in a final UpdateItem
        """

        self._ensure_single_row_base(ident)

        ts_ms = self._now_ms()
        now_iso = self._iso_utc()

        et = (event_type or "").strip().lower()
        attempt = payload.get("attempt")
        if et in ("plan", "sql", "final", "error") and attempt is not None:
            self._ensure_single_row_attempt(ident, int(attempt))

        expr_names: Dict[str, str] = {
            "#ua": "updated_at",
            "#last": "last_event",
        }
        expr_vals: Dict[str, Any] = {
            ":ua": now_iso,
            ":last": to_ddb_safe(
                {
                    "type": event_type,
                    "ts": ts_ms,
                    "payload": scrub_large_strings(json_safe(payload)),
                }
            ),
        }
        sets = ["#ua=:ua", "#last=:last"]

        # Top-level fields
        if et == "received":
            # Only set non-null top-level fields.
            q = payload.get("question")
            mh = payload.get("mode_hint")
            env = payload.get("env")
            if q is not None:
                expr_names["#q"] = "question"
                expr_vals[":q"] = to_ddb_safe(q)
                sets.append("#q=:q")
            if mh is not None:
                expr_names["#mh"] = "mode_hint"
                expr_vals[":mh"] = to_ddb_safe(mh)
                sets.append("#mh=:mh")
            if env is not None:
                expr_names["#env"] = "env"
                expr_vals[":env"] = to_ddb_safe(env)
                sets.append("#env=:env")

            expr_names["#status"] = "status"
            expr_vals[":status"] = to_ddb_safe(payload.get("status", "received"))
            sets.append("#status=:status")

        elif et == "retrieval":
            expr_names.update({"#retrieval": "retrieval", "#status": "status"})
            expr_vals.update(
                {
                    ":retrieval": to_ddb_safe(scrub_large_strings(json_safe(payload))),
                    ":status": "retrieved",
                }
            )
            sets.extend(["#retrieval=:retrieval", "#status=:status"])

        elif et in ("plan", "sql", "final", "error") and attempt is not None:
            akey = str(int(attempt))
            expr_names.update({"#attempts": "attempts", "#a": akey, "#status": "status", "#attempt": "attempt"})
            expr_vals.update({":attempt": int(attempt)})
            sets.append("#attempt=:attempt")

            if et == "plan":
                if "raw_plan" in payload:
                    expr_names["#rp"] = "raw_plan"
                    expr_vals[":rp"] = to_ddb_safe(scrub_large_strings(json_safe(payload.get("raw_plan"))))
                    sets.append("#attempts.#a.#rp=:rp")
                if "validated_plan" in payload:
                    expr_names["#vp"] = "validated_plan"
                    expr_vals[":vp"] = to_ddb_safe(scrub_large_strings(json_safe(payload.get("validated_plan"))))
                    sets.append("#attempts.#a.#vp=:vp")
                expr_vals[":status"] = "planned"
                sets.append("#status=:status")

            elif et == "sql":
                if "db_type" in payload:
                    expr_names["#dbt"] = "db_type"
                    expr_vals[":dbt"] = to_ddb_safe(payload.get("db_type"))
                    sets.append("#attempts.#a.#dbt=:dbt")
                if "sql" in payload:
                    expr_names["#sql"] = "sql"
                    expr_vals[":sql"] = to_ddb_safe(scrub_large_strings(json_safe(payload.get("sql"))))
                    sets.append("#attempts.#a.#sql=:sql")
                expr_vals[":status"] = "sql_generated"
                sets.append("#status=:status")

            elif et == "final":
                expr_names["#final"] = "final"
                expr_vals[":final"] = to_ddb_safe(scrub_large_strings(json_safe(payload)))
                sets.append("#attempts.#a.#final=:final")

                # confidence stored top-level (Decimal safe)
                if "confidence" in payload:
                    expr_names["#conf"] = "confidence"
                    expr_vals[":conf"] = to_ddb_safe(payload.get("confidence"))
                    sets.append("#conf=:conf")

                expr_vals[":status"] = payload.get("status", "ok")
                sets.append("#status=:status")

            elif et == "error":
                expr_names["#err"] = "error"
                expr_vals[":err"] = to_ddb_safe(scrub_large_strings(json_safe(payload)))
                sets.append("#attempts.#a.#err=:err")

                # Convenience: if orchestrator included these fields on the error event,
                # also surface them in their primary locations so the single-row item
                # still contains plan/sql even when the run fails early.
                if payload.get("raw_plan") is not None:
                    expr_names["#rp"] = "raw_plan"
                    expr_vals[":rp"] = to_ddb_safe(scrub_large_strings(json_safe(payload.get("raw_plan"))))
                    sets.append("#attempts.#a.#rp=:rp")
                if payload.get("validated_plan") is not None:
                    expr_names["#vp"] = "validated_plan"
                    expr_vals[":vp"] = to_ddb_safe(scrub_large_strings(json_safe(payload.get("validated_plan"))))
                    sets.append("#attempts.#a.#vp=:vp")
                if payload.get("sql") is not None:
                    expr_names["#sql"] = "sql"
                    expr_vals[":sql"] = to_ddb_safe(scrub_large_strings(json_safe(payload.get("sql"))))
                    sets.append("#attempts.#a.#sql=:sql")
                if payload.get("db_type") is not None:
                    expr_names["#dbt"] = "db_type"
                    expr_vals[":dbt"] = to_ddb_safe(payload.get("db_type"))
                    sets.append("#attempts.#a.#dbt=:dbt")

                expr_vals[":status"] = "failed"
                sets.append("#status=:status")

        # Final update (leaf-only)
        self.table.update_item(
            Key=self._key_for_query(ident),
            UpdateExpression="SET " + ", ".join(sets),
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_vals,
        )

    def write_event(self, ident: QueryIdentity, event_type: str, payload: Dict[str, Any]) -> None:
        """Write an event.

        In compact mode, ONLY event_type='payload' is persisted (single item per query).
        """
        try:
            ts_ms = int(time.time() * 1000)
            # SINGLE-ROW MODE: merge all events into one DynamoDB item per query.
            if self.mode in ("single_row", "one_row", "row"):
                self._update_single_row(ident, event_type, payload)
                return

            # COMPACT MODE: one item per query (ignore all non-payload events).
            if self.mode in ("payload", "single", "compact"):
                if (event_type or "").strip().lower() != "payload":
                    return

                # Keep the item shape intentionally simple:
                #   { <PK>, <optional SK>, query_id, ts, payload: { ... } }
                # This prevents the table from exploding with many per-step records.
                item: Dict[str, Any] = {
                    self.pk_name: self._pk_value(ident),
                    "query_id": ident.query_id,
                    "ts": ts_ms,
                }
                skv = self._sk_value(ident)
                if skv is not None:
                    item[self.sk_name] = skv

                # Always store the caller's JSON-like dict under a single attribute.
                # DynamoDB supports nested maps/lists, and this keeps the schema stable.
                item["payload"] = to_ddb_safe(scrub_large_strings(json_safe(payload)))

                ttl = self._ttl()
                if ttl is not None:
                    item[self.ttl_attribute] = ttl
                self.table.put_item(Item=item)
                return

            # DETAILED MODE: one item per event
            item = {
                self.pk_name: self._pk_value(ident),
                self.sk_name: f"{ident.query_id}#event#{event_type}#{ts_ms}",
                "query_id": ident.query_id,
                "event_type": event_type,
                "ts": ts_ms,
                "payload": to_ddb_safe(scrub_large_strings(json_safe(payload))),
            }
            ttl = self._ttl()
            if ttl is not None:
                item[self.ttl_attribute] = ttl
            self.table.put_item(Item=item)
        except Exception:
            log.exception("Failed to write DynamoDB event", extra={"table": self.table_name})

    def write_snapshot(self, ident: QueryIdentity, payload: Dict[str, Any]) -> None:
        """Write the latest snapshot.

        Snapshots are suppressed in compact & single-row modes (single item per query).
        """
        try:
            if self.mode in ("payload", "single", "compact", "single_row", "one_row", "row"):
                return

            ts_ms = int(time.time() * 1000)
            item: Dict[str, Any] = {
                self.pk_name: self._pk_value(ident),
                self.sk_name: f"{ident.query_id}#snapshot",
                "query_id": ident.query_id,
                "record_type": "snapshot",
                "ts": ts_ms,
                "payload": to_ddb_safe(scrub_large_strings(json_safe(payload))),
            }
            ttl = self._ttl()
            if ttl is not None:
                item[self.ttl_attribute] = ttl
            self.table.put_item(Item=item)
        except Exception:
            log.exception("Failed to write DynamoDB snapshot", extra={"table": self.table_name})
