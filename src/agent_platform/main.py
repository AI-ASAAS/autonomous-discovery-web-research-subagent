"""Executable entry point: ``python -m agent_platform.main``.

Configuration precedence: CLI flags > environment variables > defaults.

    python -m agent_platform.main --port 8020
    PORT=8020 python -m agent_platform.main
    make run PORT=8020
"""

from __future__ import annotations

import argparse
import os

from .api import app

__all__ = ["app"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8010


def _env_port(default: int = DEFAULT_PORT) -> int:
    raw = os.getenv("PORT")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"PORT must be an integer, got {raw!r}") from exc


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scholarship-orchestrator",
        description="Run the scholarship orchestrator API.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", DEFAULT_HOST),
        help=f"Bind address (env HOST, default {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_port(),
        help=f"TCP port (env PORT, default {DEFAULT_PORT}); use 0 to let the OS choose.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=_env_flag("RELOAD"),
        help="Reload on code changes (env RELOAD=1).",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    import uvicorn

    from .config import settings
    from .observability import configure_logging

    args = _parse_args(argv)
    # One logging setup for both the app and uvicorn (log_config=None keeps
    # uvicorn from installing a second, differently formatted handler).
    configure_logging(settings.log_level, settings.log_json)
    uvicorn.run(
        "agent_platform.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=None,
        access_log=False,  # request logging happens in RequestContextMiddleware
        proxy_headers=True,
        forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )


if __name__ == "__main__":
    run()
