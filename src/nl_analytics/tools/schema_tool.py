from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List

from nl_analytics.schema.registry import SchemaRegistry


@dataclass(frozen=True)
class SchemaContext:
    tables: Dict[str, Any]
    joins: List[Dict[str, Any]]


def build_schema_context(registry: SchemaRegistry) -> SchemaContext:
    """Build a compact, retrieval-friendly schema context.

    This context is embedded into the vector store and used to ground the planner.
    Include business aliases so user terms map correctly to real columns.
    """
    tables: Dict[str, Any] = {}
    for t in registry.list_tables():
        spec = registry.get_table(t)
        tables[t] = {
            "description": spec.description,
            "primary_key": spec.primary_key,
            "aliases": list(spec.aliases or []),
            "business_tags": list(spec.business_tags or []),
            "columns": {
                c: {
                    "type": spec.columns[c].type,
                    "description": spec.columns[c].description,
                    "aliases": list(spec.columns[c].aliases or []),
                }
                for c in spec.columns
            },
        }

    joins = [
        {
            "left_table": j.left_table,
            "right_table": j.right_table,
            "left_keys": j.left_keys,
            "right_keys": j.right_keys,
            "join_type": j.join_type,
        }
        for j in registry.joins
    ]
    return SchemaContext(tables=tables, joins=joins)
