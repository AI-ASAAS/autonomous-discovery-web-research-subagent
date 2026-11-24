"""Bridge to the two existing repositories.

The repositories are plain folders, not installed packages, and one of them
resolves its SQLite path relative to the process working directory.  This
module makes the import paths explicit and pins the database URL to an
absolute path so the platform behaves identically from any directory.
"""

from __future__ import annotations

import os
import sys

from .config import REPO_DISCOVERY, REPO_MATCHER, settings

_ready = False


def setup_repo_paths() -> None:
    global _ready
    if _ready:
        return
    seen: set[str] = set()
    for path in (REPO_MATCHER, REPO_DISCOVERY):
        resolved = str(path)
        if path.is_dir() and resolved not in seen and resolved not in sys.path:
            sys.path.insert(0, resolved)
            seen.add(resolved)
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{settings.grants_db_path}")
    _ready = True


def repos_present() -> bool:
    return REPO_MATCHER.is_dir() and REPO_DISCOVERY.is_dir()
