"""Create the DynamoDB audit table used by nl_analytics.

Table schema:
  - Partition key (PK): pk (S)
  - Sort key (SK): sk (S)
  - TTL attribute: expires_at (N)  # enabled via UpdateTimeToLive

Usage:
  python scripts/create_dynamodb_audit_table.py --table hexa-nl-analytics-audit --region us-east-1

You can also pass --endpoint-url for local dynamodb (DynamoDB Local).
"""

from __future__ import annotations

import argparse
import sys

import boto3
from botocore.exceptions import ClientError


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="DynamoDB table name")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--endpoint-url", default="", help="Optional endpoint (ex: http://localhost:8000)")
    ap.add_argument("--billing-mode", default="PAY_PER_REQUEST", choices=["PAY_PER_REQUEST", "PROVISIONED"])
    args = ap.parse_args()

    ddb = boto3.client("dynamodb", region_name=args.region, endpoint_url=args.endpoint_url or None)

    # Create table (idempotent)
    try:
        ddb.describe_table(TableName=args.table)
        print(f"Table already exists: {args.table}")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise

        params = {
            "TableName": args.table,
            "KeySchema": [
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            "AttributeDefinitions": [
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            "BillingMode": args.billing_mode,
        }
        if args.billing_mode == "PROVISIONED":
            params["ProvisionedThroughput"] = {"ReadCapacityUnits": 5, "WriteCapacityUnits": 5}

        print(f"Creating table {args.table}...")
        ddb.create_table(**params)
        waiter = ddb.get_waiter("table_exists")
        waiter.wait(TableName=args.table)
        print("Created.")

    # Enable TTL on expires_at
    try:
        ddb.update_time_to_live(
            TableName=args.table,
            TimeToLiveSpecification={"Enabled": True, "AttributeName": "expires_at"},
        )
        print("TTL enabled on attribute 'expires_at'.")
    except ClientError as e:
        print(f"Warning: could not enable TTL (you can enable it in console): {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
