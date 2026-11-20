#!/bin/sh
# Run the agent-kb tests. The store tests need PostgreSQL 18 with pgvector on 127.0.0.1:5432 (they skip when it is
# down, unless KB_REQUIRE_PG=1, which makes a missing database a failure).
cd "$(dirname "$0")/.." && exec uv run --quiet --python 3.14 --with pytest==9.2.0 \
  --with 'psycopg[binary]==3.3.9' --with pyyaml==6.0.3 --with numpy==2.5.5 \
  --with onnxruntime==1.32.0 --with tokenizers==0.23.3 --with huggingface-hub==2.4.0 -- python -B -m pytest -q tests "$@"
