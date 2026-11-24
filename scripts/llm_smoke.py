r"""Manual real-LLM smoke test for the orchestrator platform.

Run it yourself with a provider key in the environment:

    $env:LLM_API_KEY="..."; $env:LLM_MODEL="gemini-2.5-flash"
    .\.venv\Scripts\python scripts\llm_smoke.py

The key and model are never printed.  With no ``LLM_API_KEY`` the script
prints ``skipped`` and exits 0.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

MESSAGES = [
    "I want a master's scholarship in computer science",
    "My GPA is 3.6 out of 4 and I have IELTS 6.5",
    "Match my profile",
    "Is the deadline of the first scholarship June 3?",
]

_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_MONTH_ALT = "|".join(_MONTHS)


def _extract_dates(text: str) -> list[tuple[int, int, int | None]]:
    """Return (month, day, year) tuples found in ``text``."""
    found: list[tuple[int, int, int | None]] = []

    for match in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        found.append((int(match.group(2)), int(match.group(3)), int(match.group(1))))

    for match in re.finditer(
        rf"\b({_MONTH_ALT})\s+(\d{{1,2}})(?:,\s*(\d{{4}}))?\b", text, re.IGNORECASE
    ):
        month = _MONTHS.index(match.group(1).title()) + 1
        year = int(match.group(3)) if match.group(3) else None
        found.append((month, int(match.group(2)), year))

    for match in re.finditer(
        rf"\b(\d{{1,2}})\s+({_MONTH_ALT})(?:\s+(\d{{4}}))?\b", text, re.IGNORECASE
    ):
        month = _MONTHS.index(match.group(2).title()) + 1
        year = int(match.group(3)) if match.group(3) else None
        found.append((month, int(match.group(1)), year))

    return found


def _deadlines(grants: list[dict]) -> list[tuple[int, int, int]]:
    deadlines: list[tuple[int, int, int]] = []
    for grant in grants:
        raw = grant.get("deadline")
        if not raw:
            continue
        match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", str(raw))
        if match:
            deadlines.append(
                (int(match.group(2)), int(match.group(3)), int(match.group(1)))
            )
    return deadlines


def _is_known(date: tuple[int, int, int | None], deadlines: list[tuple[int, int, int]]) -> bool:
    month, day, year = date
    for deadline_month, deadline_day, deadline_year in deadlines:
        if month != deadline_month:
            continue
        if day is not None and day != deadline_day:
            continue
        if year is not None and year != deadline_year:
            continue
        return True
    return False


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        from dotenv import load_dotenv

        load_dotenv(repo_root / ".env", override=False)
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        pass

    if not os.getenv("LLM_API_KEY", "").strip():
        print("skipped")
        return 0

    # Must be set before agent_platform.config is imported.
    os.environ["AGENT_OFFLINE"] = "0"
    os.environ.pop("AGENT_ENGINE", None)

    for path in (str(repo_root / "src"), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    if not os.getenv("LLM_MODEL", "").strip():
        print("note: LLM_MODEL is not set; the provider default will be used")

    from fastapi.testclient import TestClient

    from agent_platform.api import Runtime, create_app

    with tempfile.TemporaryDirectory() as temp_dir:
        runtime = Runtime(memory_path=Path(temp_dir) / "sessions.db")
        try:
            with TestClient(create_app(runtime)) as client:
                health = client.get("/health").json()
                print(f"engine={health.get('agent_engine')}")
                if not health.get("llm_available", True):
                    print(
                        f"circuit open, retry_after={health.get('llm_retry_after_seconds')}"
                    )
                    return 0

                session_id = client.post("/api/session").json()["session_id"]
                replies: list[dict] = []
                for index, message in enumerate(MESSAGES, 1):
                    response = client.post(
                        "/api/chat",
                        json={"session_id": session_id, "message": message},
                    )
                    if response.status_code != 200:
                        print(f"turn={index} status={response.status_code}")
                        return 1
                    data = response.json()
                    replies.append(data)
                    print(
                        f"turn={index} intent={data['intent']} "
                        f"llm_used={data['llm_used']} "
                        f"model_calls={data.get('model_calls', 0)}"
                    )
                    for match in data.get("matches", []):
                        print(
                            f"  match={match['match_percentage']} "
                            f"grant={match['title']}"
                        )

                print("reply4:", replies[3]["reply"])

                grants = client.get("/api/grants/top10").json()["grants"]
                deadlines = _deadlines(grants)
                dates = _extract_dates(replies[3]["reply"])
                unknown = [date for date in dates if not _is_known(date, deadlines)]
                print(f"reply4_dates={dates}")
                print(
                    "reply4_has_date_not_in_grant_record="
                    f"{bool(unknown)}"
                )
        finally:
            runtime.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
