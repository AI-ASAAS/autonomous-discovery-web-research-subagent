"""Test-suite isolation helpers.

``tests/test_seed.py`` seeds through the process-wide Role 2 engine, which by
default points at the repository's real ``grants.db``.  That test module gets a
temporary database instead, so running the suite never mutates the developer's
data.  No Role 2 code is changed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.models import Base


@pytest.fixture(autouse=True)
def isolate_seed_db(request, tmp_path, monkeypatch):
    node_path = Path(str(getattr(request.node, "path", request.node.fspath)))
    if node_path.name != "test_seed.py":
        return

    import core.db as core_db

    engine = create_engine(
        f"sqlite:///{tmp_path / 'grants.db'}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False,
    )
    monkeypatch.setattr(core_db, "engine", engine)
    monkeypatch.setattr(core_db, "SessionLocal", session_local)
