#!/usr/bin/env bash
# Idempotent local pgvector setup via Homebrew. Target: a `vdm` role/db on
# localhost:5432 with the `vector` extension, reachable by the test DSN.
#
# NOTE: The pgvector Homebrew bottle ships pg17/pg18 variants only. When
# targeting postgresql@16 this script falls back to building pgvector from
# source using pg16's pg_config. The build is skipped if the extension
# control file already exists.
set -euo pipefail

brew install postgresql@16 pgvector
brew services start postgresql@16

# Wait for the server to accept connections.
PG_BIN="$(brew --prefix postgresql@16)/bin"
export PATH="$PG_BIN:$PATH"
for _ in $(seq 1 30); do
  if pg_isready -h localhost -p 5432 >/dev/null 2>&1; then break; fi
  sleep 1
done

# ── Build pgvector from source if the bottle didn't install for pg16 ────────
PG16_EXT="$(brew --prefix postgresql@16)/share/postgresql@16/extension/vector.control"
if [ ! -f "$PG16_EXT" ]; then
  echo "pgvector bottle missing pg16 support — building from source..."
  TMP_SRC="$(mktemp -d)/pgvector"
  git clone --depth 1 https://github.com/pgvector/pgvector.git "$TMP_SRC"
  pushd "$TMP_SRC" >/dev/null
  PG_CONFIG="$(brew --prefix postgresql@16)/bin/pg_config" make
  PG_CONFIG="$(brew --prefix postgresql@16)/bin/pg_config" make install
  popd >/dev/null
  echo "pgvector built and installed for postgresql@16"
fi

# Create the vdm role + database if absent (connect to default 'postgres' db
# as the bootstrap superuser, which is the current OS user under brew).
psql -d postgres -v ON_ERROR_STOP=1 <<'SQL'
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'vdm') THEN
    CREATE ROLE vdm LOGIN PASSWORD 'vdm' SUPERUSER;
  END IF;
END
$$;
SQL

psql -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'vdm'" \
  | grep -q 1 || createdb -O vdm vdm

psql -d vdm -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector"
echo "pgvector ready at postgresql://vdm:vdm@localhost:5432/vdm"
