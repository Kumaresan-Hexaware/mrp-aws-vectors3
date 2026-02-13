"""Create the DynamoDB table used by the query store.

Usage:
  python scripts/create_dynamodb_table.py

Reads config from config/<APP_ENV>.yaml and env vars (.env) via nl_analytics
Settings loader.

Table schema:
  - Partition key: configurable (default session_id, type S)
  - Sort key: configurable (default sk, type S)

If the table already exists, this script exits without error.
"""

from __future__ import annotations

import time
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def main() -> None:
    # Ensure src/ is on the module path (this repo isn't installed as a package by default).
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from nl_analytics.config.settings import load_settings

    s = load_settings()
    table_name = getattr(s, "dynamodb_table_name", "")
    region = getattr(s, "dynamodb_region", None) or getattr(s, "aws_region", "us-east-1")
    endpoint_url = getattr(s, "dynamodb_endpoint_url", "") or None
    pk_name = getattr(s, "dynamodb_pk_name", "session_id") or "session_id"
    sk_name = getattr(s, "dynamodb_sk_name", "sk") or "sk"

    if not table_name:
        raise SystemExit("DYNAMODB_TABLE_NAME is empty. Set it in config or environment.")

    ddb = boto3.client("dynamodb", region_name=region, endpoint_url=endpoint_url)

    try:
        ddb.describe_table(TableName=table_name)
        print(f"Table already exists: {table_name}")
        return
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise

    print(f"Creating table: {table_name} (region={region})")
    ddb.create_table(
        TableName=table_name,
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[
            {"AttributeName": pk_name, "AttributeType": "S"},
            {"AttributeName": sk_name, "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": pk_name, "KeyType": "HASH"},
            {"AttributeName": sk_name, "KeyType": "RANGE"},
        ],
    )

    # Wait until active
    for _ in range(60):
        desc = ddb.describe_table(TableName=table_name)
        status = desc["Table"]["TableStatus"]
        if status == "ACTIVE":
            print("Table is ACTIVE")
            return
        time.sleep(2)

    print("Timed out waiting for table to become ACTIVE")


if __name__ == "__main__":
    main()
