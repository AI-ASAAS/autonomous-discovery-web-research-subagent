"""CLI entry point: port/host configuration precedence."""

from __future__ import annotations

import pytest

from agent_platform.main import _parse_args


def test_cli_port_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "8555")
    assert _parse_args(["--port", "8666"]).port == 8666


def test_environment_port_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "8555")
    assert _parse_args([]).port == 8555


def test_default_port_without_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PORT", raising=False)
    assert _parse_args([]).port == 8010


def test_invalid_environment_port_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "not-a-number")
    with pytest.raises(SystemExit):
        _parse_args([])


def test_host_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOST", "0.0.0.0")
    assert _parse_args([]).host == "0.0.0.0"
    assert _parse_args(["--host", "127.0.0.1"]).host == "127.0.0.1"
