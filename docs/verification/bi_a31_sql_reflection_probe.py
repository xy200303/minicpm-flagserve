import argparse
import json
import sqlite3
import time
from pathlib import Path

import flag_gems
from flag_gems.utils.libentry import libcache
from flag_gems.utils.models.sql import SQLPersistantModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = libcache.db_url.removeprefix("sqlite:///")
    with sqlite3.connect(path) as conn:
        name = conn.execute(
            "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone()[0]
    start = time.perf_counter()
    model = SQLPersistantModel.build_sql_model_by_db(name, libcache.model.engine)
    elapsed_ms = (time.perf_counter() - start) * 1000
    result = {
        "source": str(Path(flag_gems.__file__).resolve()),
        "table": name,
        "model_loaded": model is not None,
        "model_table": model.__table__.name if model is not None else None,
        "elapsed_ms": elapsed_ms,
    }
    text = json.dumps(result, indent=2)
    print(text)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
