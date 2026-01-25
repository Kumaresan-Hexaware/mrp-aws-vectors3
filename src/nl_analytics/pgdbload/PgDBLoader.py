import os
import sys
import csv
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_batch


# -----------------------------
# Config
# -----------------------------
DELIMITER = "Ç"
ENCODING = "latin-1"
CHUNK_LINES = 50_000  # tune based on file size & memory


def pg_connect():
    """
    Uses standard env vars:
      PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
    """
    return psycopg2.connect(
        host=os.getenv("PGHOST", "bedrock-agent-poc-db.czhycpf6acnx.us-east-1.rds.amazonaws.com"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "postgres"),
        user=os.getenv("PGUSER", "postgresadmin"),
        password=os.getenv("PGPASSWORD", "J*2swFkOx5>03]-7jTrU9XuEmWil"),
    )


# -----------------------------
# Helpers
# -----------------------------
def sanitize_identifier(name: str) -> str:
    """
    Make a safe SQL identifier. Keeps letters/numbers/_ and lowercases.
    """
    out = []
    for ch in name.strip():
        if ch.isalnum() or ch == "_":
            out.append(ch.lower())
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s or "col"


def infer_table_name(file_path: Path) -> str:
    base = file_path.stem  # PVR01500_something.nzf -> PVR01500_something
    return sanitize_identifier(base)


def read_header(file_path: Path) -> List[str]:
    """
    Reads first line header using Latin-1 + Ç delimiter.
    """
    with open(file_path, "r", encoding=ENCODING, errors="replace", newline="") as f:
        line = f.readline()

        if not line:
            raise ValueError(f"Empty file: {file_path}")
        cols = line.rstrip("\n").split(DELIMITER)
        cols = [sanitize_identifier(c) for c in cols]
        # de-dup if needed
        seen: Dict[str, int] = {}
        deduped = []
        for c in cols:
            if c not in seen:
                seen[c] = 1
                deduped.append(c)
            else:
                seen[c] += 1
                deduped.append(f"{c}_{seen[c]}")
        return deduped


def table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema, table),
        )
        return cur.fetchone() is not None


def create_table_all_text(conn, schema: str, table: str, columns: List[str]):
    """
    Creates table with all columns as TEXT.
    (Safest starting point; you can later cast/transform into typed tables.)
    """
    with conn.cursor() as cur:
        cols_sql = sql.SQL(", ").join(
            sql.SQL("{} TEXT").format(sql.Identifier(c)) for c in columns
        )
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
        cur.execute(
            sql.SQL("CREATE TABLE IF NOT EXISTS {}.{} ({})").format(
                sql.Identifier(schema), sql.Identifier(table), cols_sql
            )
        )
    conn.commit()

SOURCE_DELIM = "Ç"     # input file delimiter
TARGET_DELIM = "\t"    # output/COPY delimiter (single-byte in UTF-8)

def copy_chunk(conn, schema: str, table: str, columns: list[str], lines: list[str]):
    """
    Convert Ç-delimited text -> TSV in-memory -> COPY with delimiter TAB.
    """
    if not lines:
        return

    out = io.StringIO()

    # Read input lines using SOURCE delimiter
    reader = csv.reader(lines, delimiter=SOURCE_DELIM)

    # Write transformed rows using TARGET delimiter
    writer = csv.writer(
        out,
        delimiter=TARGET_DELIM,
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL
    )

    for row in reader:
        writer.writerow(row)

    out.seek(0)

    with conn.cursor() as cur:
        copy_sql = sql.SQL(
            "COPY {}.{} ({}) FROM STDIN WITH (FORMAT csv, DELIMITER {}, NULL '', QUOTE '\"', ESCAPE '\"')"
        ).format(
            sql.Identifier(schema),
            sql.Identifier(table),
            sql.SQL(", ").join(sql.Identifier(c) for c in columns),
            sql.Literal(TARGET_DELIM),
        )
        cur.copy_expert(copy_sql.as_string(conn), out)

    conn.commit()

def copy_chunk_2(
    conn,
    schema: str,
    table: str,
    columns: List[str],
    lines: List[str],
):
    """
    Uses COPY FROM STDIN with CSV parsing + custom delimiter.
    We pass only data lines (no header).
    """
    if not lines:
        return

    # Build an in-memory text stream for COPY
    data_stream = io.StringIO("".join(lines))

    with conn.cursor() as cur:
        copy_sql = sql.SQL(
            "COPY {}.{} ({}) FROM STDIN WITH (FORMAT csv, DELIMITER {}, NULL '', QUOTE '\"', ESCAPE '\"')"
        ).format(
            sql.Identifier(schema),
            sql.Identifier(table),
            sql.SQL(", ").join(sql.Identifier(c) for c in columns),
            sql.Literal(DELIMITER),
        )
        cur.copy_expert(copy_sql.as_string(conn), data_stream)
    conn.commit()


def load_nzf_to_postgres(
    file_path: Path,
    schema: str = "public",
    table: Optional[str] = None,
):
    table = table or infer_table_name(file_path)
    columns = read_header(file_path)

    conn = pg_connect()
    try:
        if not table_exists(conn, schema, table):
            create_table_all_text(conn, schema, table, columns)

        # Stream file in chunks (skip header)
        with open(file_path, "r", encoding=ENCODING, errors="replace", newline="") as f:
            _header = f.readline()  # skip
            buf: List[str] = []
            count = 0

            for line in f:
                # Ensure line ends with \n (COPY expects line-separated records)
                if not line.endswith("\n"):
                    line += "\n"
                buf.append(line)
                if len(buf) >= CHUNK_LINES:
                    copy_chunk(conn, schema, table, columns, buf)
                    count += len(buf)
                    print(f"Loaded {count:,} rows into {schema}.{table}")
                    buf.clear()

            # final chunk
            if buf:
                copy_chunk(conn, schema, table, columns, buf)
                count += len(buf)
                print(f"Loaded {count:,} rows into {schema}.{table}")

        print(f"✅ Done: {file_path.name} -> {schema}.{table}")

    finally:
        conn.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python loader.py <path-to-nzf-or-folder> [schema] [table]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    schema = sys.argv[2] if len(sys.argv) >= 3 else "public"
    table_arg = sys.argv[3] if len(sys.argv) >= 4 else None

    if input_path.is_dir():
        # Load all *.nzf in the folder, each to its own table (derived from filename)
        for fp in sorted(input_path.glob("*.nzf")):
            load_nzf_to_postgres(fp, schema=schema, table=None)
    else:
        load_nzf_to_postgres(input_path, schema=schema, table=table_arg)


if __name__ == "__main__":
    main()
