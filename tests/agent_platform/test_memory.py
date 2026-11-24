"""Memory keeps profiles, transcripts and artifacts per session."""

from __future__ import annotations

from pathlib import Path

from agent_platform.memory import MemoryStore
from agent_platform.schemas import DegreeLevel, StudentProfile


def test_profile_roundtrip(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.db") as store:
        profile = StudentProfile(
            name="أحمد",
            gpa=3.6,
            major="علوم الحاسوب",
            degree_level=DegreeLevel.BACHELOR,
            target_degree=DegreeLevel.MASTER,
            nationality="Yemeni",
        )
        store.save_profile("s1", profile)
        loaded = store.load_profile("s1")
        assert loaded.name == "أحمد"
        assert loaded.gpa == 3.6
        assert loaded.missing_required() == []


def test_unknown_session_returns_empty_profile(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.db") as store:
        profile = store.load_profile("missing")
        assert profile.student_id == "missing"
        assert profile.name is None


def test_transcript_and_artifacts(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.db") as store:
        store.append_message("s1", "user", "مرحبا")
        store.append_message("s1", "assistant", "أهلاً")
        store.save_artifact("s1", "match", {"top": 88.5})

        history = store.history("s1")
        assert [message.role for message in history] == ["user", "assistant"]
        assert store.message_count("s1") == 2

        artifacts = store.artifacts("s1", "match")
        assert len(artifacts) == 1
        assert artifacts[0]["payload"]["top"] == 88.5


def test_reset_clears_session(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.db") as store:
        store.save_profile("s1", StudentProfile(name="أحمد"))
        store.append_message("s1", "user", "مرحبا")
        store.save_artifact("s1", "plan", {"plans": []})

        store.reset("s1")

        assert store.load_profile("s1").name is None
        assert store.message_count("s1") == 0
        assert store.artifacts("s1") == []
