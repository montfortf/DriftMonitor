def test_pgvector_extension_available(conn):
    row = conn.execute(
        "SELECT extname FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    assert row is not None
    assert row[0] == "vector"
