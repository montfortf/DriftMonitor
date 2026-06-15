import os

import psycopg
import pytest
from pgvector.psycopg import register_vector

DSN = os.environ.get("VDM_DSN", "postgresql://vdm:vdm@localhost:5432/vdm")


@pytest.fixture()
def conn():
    with psycopg.connect(DSN) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(c)
        yield c
        c.execute("DROP TABLE IF EXISTS items")
        c.commit()
