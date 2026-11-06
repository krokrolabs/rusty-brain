#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb==2.0.0"]
# ///
"""Independently compare an active-row SQLite source with a Parquet snapshot.

Accepts both schema generations: legacy snapshots (6 columns, no provenance)
and current ones (8 columns, agent + session). The embedding width is derived
from the source blob, never hardcoded.
"""

from __future__ import annotations

import argparse
import duckdb
from pathlib import Path
import sqlite3
import struct

LEGACY_COLUMNS = ["id", "content", "tags", "embedding_model", "created_at", "embedding"]
CURRENT_COLUMNS = LEGACY_COLUMNS[:5] + ["agent", "session", "embedding"]


def source_rows(path: Path) -> tuple[list[str], list[tuple[object, ...]]]:
    uri = path.absolute().as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
        columns = {row[1] for row in connection.execute("PRAGMA table_info(memories)")}
        provenance = "agent" in columns and "session" in columns
        selected = CURRENT_COLUMNS if provenance else LEGACY_COLUMNS
        return selected, connection.execute(
            f"SELECT {', '.join(selected)} FROM memories "
            "WHERE deleted_at IS NULL ORDER BY created_at ASC"
        ).fetchall()
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--reported-count", type=int, required=True)
    args = parser.parse_args()

    columns, expected = source_rows(args.db)
    assert args.reported_count == len(expected), (
        f"reported count {args.reported_count} != selected source rows {len(expected)}"
    )
    dimension = None
    connection = duckdb.connect(":memory:")
    try:
        assert duckdb.__version__ == "2.0.0", duckdb.__version__
        relation = "read_parquet(?)"
        described = connection.execute(
            f"DESCRIBE SELECT {', '.join(columns)} FROM {relation}", [str(args.parquet)]
        ).fetchall()
        assert [row[0] for row in described] == columns, described
        # DuckDB 2.0.0 maps Arrow FixedSizeList<Float32>(dim) to FLOAT[];
        # the Rust Arrow reader separately asserts the fixed-size declaration.
        expected_types = ["VARCHAR"] * (len(columns) - 1) + ["FLOAT[]"]
        assert [row[1] for row in described] == expected_types, described
        actual = connection.execute(
            f"SELECT {', '.join(columns)} FROM {relation}", [str(args.parquet)]
        ).fetchall()
    finally:
        connection.close()

    assert len(actual) == len(expected), f"Parquet rows {len(actual)} != source {len(expected)}"
    for row_index, (exported, source) in enumerate(zip(actual, expected, strict=True)):
        metadata = len(columns) - 1
        assert exported[:metadata] == source[:metadata], f"metadata mismatch in row {row_index}"
        blob = bytes(source[metadata])
        if dimension is None:
            assert len(blob) % 4 == 0
            dimension = len(blob) // 4
        assert len(blob) == dimension * 4, f"source vector {row_index} has {len(blob)} bytes"
        vector = exported[metadata]
        assert len(vector) == dimension, f"export vector {row_index} has {len(vector)} values"
        for value_index, value in enumerate(vector):
            expected_bytes = blob[value_index * 4 : (value_index + 1) * 4]
            assert struct.pack("<f", value) == expected_bytes, (
                f"float32 mismatch at row {row_index}, value {value_index}"
            )

    print(
        f"DuckDB 2.0.0 verified {args.parquet}: rows={len(actual)}, "
        f"schema={len(columns)} columns/FLOAT[] ({dimension} values each), all vectors exact"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
