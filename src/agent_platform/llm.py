"""Provider-agnostic LLM access with a strict offline fallback.

The platform is fully functional without any API key: every caller has a
deterministic path.  When a key *is* configured the same functions use it.
No credential is ever written in this file, logged, or returned to a client.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable

import httpx

from .config import Settings, settings

logger = logging.getLogger(__name__)

from enum import Enum

# Only these failures are retried; a 429 is a quota/rate-limit signal and opens
# the circuit breaker instead of burning the per-turn budget.
_RETRYABLE_STATUS_CODES = frozenset({503})
MAX_RETRIES = 2  # retries, not total attempts
_DEFAULT_QUOTA_COOLDOWN_SECONDS = 600.0


class ErrorClass(str, Enum):
    PERMANENT = "permanent"
    ACCOUNT_QUOTA = "account_quota"
    TRANSIENT_UPSTREAM = "transient_upstream"
    TRANSIENT_OTHER = "transient_other"
    OTHER = "other"


class LLMUnavailable(RuntimeError):
    """Raised when no provider is configured or the provider call fails."""


class LLMResponseError(RuntimeError):
    """Raised when the provider returns an unusable payload."""


class LLMTurnBudgetExceeded(LLMUnavailable):
    """Raised when one conversational turn exhausts its model-call budget."""


def classify_error(exc: BaseException) -> tuple[ErrorClass, float | None, str]:
    """Classify an LLM error into permanent, account quota, transient, or other."""
    status = _status_code(exc)
    payload = _error_payload(exc)
    payload_str = (
        json.dumps(payload, ensure_ascii=False)
        if isinstance(payload, (dict, list))
        else str(payload)
    )
    lowered = (payload_str + " " + str(exc)).lower()

    # 1. Permanent: 401, 402, 403, and 400 messages about model type/endpoint/tool support
    if status in {401, 402, 403}:
        return ErrorClass.PERMANENT, None, f"HTTP {status}: {str(exc)}"

    if status == 400:
        model_support_keywords = (
            "model",
            "endpoint",
            "tool",
            "tools",
            "not supported",
            "unsupported",
            "invalid model",
            "unknown model",
            "does not support",
            "not found",
        )
        if any(kw in lowered for kw in model_support_keywords):
            return ErrorClass.PERMANENT, None, f"HTTP 400 config error: {str(exc)}"

    # 2. Transient upstream: 429/503 that say "temporarily rate-limited", "retry shortly",
    # "upstream", or limit_source containing "upstream" or "shared_pool"
    transient_cues = (
        "temporarily rate-limited",
        "temporarily rate limited",
        "retry shortly",
        "upstream",
        "shared_pool",
        "shared pool",
    )
    has_transient_cue = any(cue in lowered for cue in transient_cues)
    for d in _iter_dicts(payload):
        if "limit_source" in d:
            src = str(d["limit_source"]).lower()
            if "upstream" in src or "shared_pool" in src:
                has_transient_cue = True

    if status in {429, 503} and has_transient_cue:
        return ErrorClass.TRANSIENT_UPSTREAM, None, str(exc)

    # 3. Account quota: 429 whose details contain "PerDay" or a retry delay above 60 seconds (or standard 429)
    if status == 429:
        daily, delay = quota_details(exc)
        cooldown = delay if delay is not None else _DEFAULT_QUOTA_COOLDOWN_SECONDS
        return ErrorClass.ACCOUNT_QUOTA, cooldown, str(exc)

    # 4. Other transient: 503, timeouts
    if status == 503 or isinstance(exc, (TimeoutError, httpx.TimeoutException)) or "timeout" in type(exc).__name__.lower():
        return ErrorClass.TRANSIENT_OTHER, None, str(exc)

    return ErrorClass.OTHER, None, str(exc)


def is_retryable_error(exc: BaseException) -> bool:
    """True for the transient failures we retry (transient upstream, 503 and timeouts only)."""
    err_class, _, _ = classify_error(exc)
    return err_class in (ErrorClass.TRANSIENT_UPSTREAM, ErrorClass.TRANSIENT_OTHER)


def retry_delay(attempt: int, exc: BaseException | None = None) -> float:
    """Backoff delay. Transient upstream errors back off 2s and 5s; others 1s, 2s with jitter."""
    if exc is not None:
        err_class, _, _ = classify_error(exc)
        if err_class == ErrorClass.TRANSIENT_UPSTREAM:
            return 2.0 if attempt == 0 else 5.0
    return float(min(2**attempt, 2)) + random.uniform(0.0, 0.25)


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


# ---------------------------------------------------------------------------
# Per-turn budget and call accounting
# ---------------------------------------------------------------------------


class TurnBudget:
    """Wall-clock budget shared by every model call in one turn."""

    def __init__(
        self,
        seconds: float | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.seconds = max(
            0.0, float(settings.llm_turn_budget_seconds if seconds is None else seconds)
        )
        self._clock = clock or time.monotonic
        self.started = self._clock()

    def elapsed(self) -> float:
        return self._clock() - self.started

    def remaining(self) -> float:
        return self.seconds - self.elapsed()

    def expired(self) -> bool:
        return self.remaining() <= 0.0


@dataclass
class TurnStats:
    model_calls: int = 0
    successful_calls: int = 0

    def record_call(self) -> None:
        self.model_calls += 1

    def record_success(self) -> None:
        self.successful_calls += 1


_turn_budget: ContextVar[TurnBudget | None] = ContextVar(
    "agent_platform_turn_budget", default=None
)
_turn_stats: ContextVar[TurnStats | None] = ContextVar(
    "agent_platform_turn_stats", default=None
)


def begin_turn(
    seconds: float | None = None, *, budget: TurnBudget | None = None
) -> tuple[Token, Token]:
    """Install a fresh budget/stats scope for one conversational turn."""
    scoped_budget = budget or TurnBudget(seconds)
    return _turn_budget.set(scoped_budget), _turn_stats.set(TurnStats())


def end_turn(tokens: tuple[Token, Token]) -> None:
    budget_token, stats_token = tokens
    _turn_stats.reset(stats_token)
    _turn_budget.reset(budget_token)


def current_budget() -> TurnBudget | None:
    return _turn_budget.get()


def record_model_call() -> None:
    stats = _turn_stats.get()
    if stats is not None:
        stats.record_call()


def record_successful_call() -> None:
    stats = _turn_stats.get()
    if stats is not None:
        stats.record_success()


def model_calls() -> int:
    stats = _turn_stats.get()
    return stats.model_calls if stats is not None else 0


def successful_calls() -> int:
    stats = _turn_stats.get()
    return stats.successful_calls if stats is not None else 0


def llm_used_in_turn() -> bool:
    return successful_calls() > 0


# ---------------------------------------------------------------------------
# Circuit breaker (quota errors)
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Open after a quota or configuration error; skip the model until cooldown."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clock: Callable[[], float] = time.monotonic
        self._open_until = 0.0
        self._announced_open = False
        self._config_error: str | None = None
        self._permanent = False

    def set_clock(self, clock: Callable[[], float]) -> None:
        """Inject a clock in tests."""
        with self._lock:
            self._clock = clock

    def reset(self) -> None:
        with self._lock:
            self._open_until = 0.0
            self._announced_open = False
            self._config_error = None
            self._permanent = False

    def open_permanent(self, error_message: str) -> None:
        with self._lock:
            self._permanent = True
            self._open_until = float("inf")
            self._config_error = error_message
            if not self._announced_open:
                logger.error(
                    "Permanent LLM configuration/auth error; breaker open for process: %s",
                    error_message,
                )
                self._announced_open = True

    def open(self, retry_after: float) -> float:
        cooldown = max(1.0, float(retry_after))
        with self._lock:
            self._open_until = self._clock() + cooldown
            if not self._announced_open:
                logger.warning(
                    "LLM circuit opened after a quota/rate-limit error; "
                    "deterministic engine active for %.0fs",
                    cooldown,
                )
                self._announced_open = True
        return cooldown

    def is_open(self) -> bool:
        with self._lock:
            if self._permanent:
                return True
            if self._open_until <= 0.0:
                return False
            if self._clock() < self._open_until:
                return True
            self._open_until = 0.0
            if self._announced_open:
                logger.info("LLM circuit closed; provider calls resume")
                self._announced_open = False
            return False

    def config_error(self) -> str | None:
        with self._lock:
            return self._config_error

    def retry_after_seconds(self) -> float:
        with self._lock:
            if self._permanent:
                return 0.0
            if self._open_until <= 0.0:
                return 0.0
            return max(0.0, self._open_until - self._clock())


circuit = CircuitBreaker()


def reset_runtime_state() -> None:
    """Reset process-wide LLM state (test isolation and explicit resets)."""
    circuit.reset()
    _turn_budget.set(None)
    _turn_stats.set(None)


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _error_payload(exc: BaseException) -> Any:
    body = getattr(exc, "body", None)
    if body is not None:
        if isinstance(body, str):
            try:
                return json.loads(body)
            except (TypeError, ValueError):
                return body
        return body
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            return response.json()
        except Exception:  # pragma: no cover - non-JSON provider response
            return getattr(response, "text", "")
    return str(exc)


def _parse_retry_after(value: Any) -> float | None:
    if isinstance(value, dict):
        for key in ("seconds", "retry_after", "retryDelay"):
            if key in value:
                return _parse_retry_after(value[key])
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*s\s*", value)
        if match:
            return float(match.group(1))
        try:
            return float(value)
        except ValueError:
            return None
    return None


def quota_details(exc: BaseException) -> tuple[bool, float | None]:
    """Return (daily_quota, retry_delay) for a 429 provider error."""
    if _status_code(exc) != 429:
        return False, None
    payload = _error_payload(exc)
    daily = False
    delay: float | None = None
    for item in _iter_dicts(payload):
        for key, value in item.items():
            normalized = str(key).replace("_", "").lower()
            if normalized == "quotaid" and "perday" in str(value).lower():
                daily = True
            elif normalized == "retrydelay":
                parsed = _parse_retry_after(value)
                if parsed is not None:
                    delay = parsed
    if isinstance(payload, str):
        if re.search(r"quota[_ ]?id[^,}]*perday", payload, re.IGNORECASE):
            daily = True
        match = re.search(
            r"retry[_ ]?delay\"?\s*[:=]\s*\"?(\d+(?:\.\d+)?)\s*s",
            payload,
            re.IGNORECASE,
        )
        if match:
            delay = float(match.group(1))
    return daily, delay


def quota_cooldown(exc: BaseException) -> float | None:
    """Cooldown for a 429 error, or None when the error is not a quota error."""
    if _status_code(exc) != 429:
        return None
    _, delay = quota_details(exc)
    return delay if delay is not None else _DEFAULT_QUOTA_COOLDOWN_SECONDS


def note_provider_failure(exc: BaseException) -> ErrorClass:
    """Classify failure and open circuit breaker if permanent or account quota."""
    err_class, delay, msg = classify_error(exc)
    if err_class == ErrorClass.PERMANENT:
        circuit.open_permanent(msg)
    elif err_class == ErrorClass.ACCOUNT_QUOTA:
        circuit.open(delay if delay is not None else _DEFAULT_QUOTA_COOLDOWN_SECONDS)
    return err_class


_OPENAI_COMPATIBLE_DEFAULTS = {
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "openrouter": ("https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.1-8b-instant"),
}


def strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def resolve_provider(config: Settings = settings) -> tuple[str, str, str, str]:
    """Return (provider, api_key, model, base_url) or raise LLMUnavailable."""
    if config.offline or not config.llm_api_key:
        raise LLMUnavailable("No LLM configured (offline mode or missing key).")

    provider = config.llm_provider
    if not provider:
        key = config.llm_api_key
        if key.startswith("AIza"):
            provider = "gemini"
        elif key.startswith("sk-or-"):
            provider = "openrouter"
        elif key.startswith("gsk_"):
            provider = "groq"
        elif key.startswith("sk-"):
            provider = "openai"
        else:
            provider = "custom"

    base_url = config.llm_base_url
    model = config.llm_model
    if provider in _OPENAI_COMPATIBLE_DEFAULTS:
        default_base, default_model = _OPENAI_COMPATIBLE_DEFAULTS[provider]
        base_url = base_url or default_base
        model = model or default_model

    if provider == "gemini":
        model = model or "gemini-1.5-flash"
    elif provider == "custom":
        if not base_url:
            raise LLMUnavailable("LLM_BASE_URL is required for a custom provider.")
        if not model:
            raise LLMUnavailable("LLM_MODEL is required for a custom provider.")
    elif not model:
        raise LLMUnavailable(f"No default model for provider '{provider}'.")

    return provider, config.llm_api_key, model, base_url


def is_configured(config: Settings = settings) -> bool:
    try:
        resolve_provider(config)
    except LLMUnavailable:
        return False
    return True


def provider_label(config: Settings = settings) -> str:
    try:
        provider, _, model, _ = resolve_provider(config)
    except LLMUnavailable:
        return "deterministic"
    return f"{provider}:{model}"


def _call_gemini(
    prompt: str,
    api_key: str,
    model: str,
    json_mode: bool,
    temperature: float,
    timeout: float,
    connect_timeout: float = 5.0,
) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    generation_config: dict[str, Any] = {}
    # Gemini 3 ignores temperature today and will reject it later.
    if not model.lower().startswith("gemini-3"):
        generation_config["temperature"] = temperature
    if json_mode:
        generation_config["responseMimeType"] = "application/json"
    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
    }
    if generation_config:
        payload["generationConfig"] = generation_config

    with httpx.Client(
        timeout=httpx.Timeout(timeout, connect=connect_timeout)
    ) as client:
        response = client.post(
            url,
            params={"key": api_key},
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
    try:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - provider drift
        raise LLMResponseError("Unexpected Gemini response shape.") from exc


def _call_openai_compatible(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    json_mode: bool,
    temperature: float,
    timeout: float,
    connect_timeout: float = 5.0,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    # Gemini 3 via the OpenAI-compatible surface rejects temperature.
    if not model.lower().startswith("gemini-3"):
        payload["temperature"] = temperature
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "ScholarshipOrchestrator/1.0",
    }
    with httpx.Client(
        timeout=httpx.Timeout(timeout, connect=connect_timeout)
    ) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
    try:
        return response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - provider drift
        raise LLMResponseError("Unexpected chat-completions response shape.") from exc


def _invoke_provider(
    provider: str,
    api_key: str,
    model: str,
    base_url: str,
    prompt: str,
    json_mode: bool,
    temperature: float,
    timeout: float,
    connect_timeout: float,
) -> str:
    if provider == "gemini":
        return _call_gemini(
            prompt,
            api_key,
            model,
            json_mode,
            temperature,
            timeout,
            connect_timeout,
        )
    try:
        return _call_openai_compatible(
            prompt,
            api_key,
            model,
            base_url,
            json_mode,
            temperature,
            timeout,
            connect_timeout=connect_timeout,
        )
    except TypeError:
        return _call_openai_compatible(
            prompt,
            api_key,
            model,
            base_url,
            json_mode,
            temperature,
            timeout,
        )


def _provider_targets(config: Settings) -> list[tuple[str, str, str, str]]:
    targets = [resolve_provider(config)]
    fallback_model = config.llm_fallback_model.strip()
    if fallback_model and fallback_model != targets[0][2]:
        try:
            targets.append(
                resolve_provider(replace(config, llm_model=fallback_model))
            )
        except LLMUnavailable:
            logger.warning("LLM fallback model could not be resolved; ignoring it")
    return targets


def complete(
    prompt: str,
    *,
    json_mode: bool = False,
    temperature: float = 0.2,
    timeout: float | None = None,
    config: Settings = settings,
) -> str:
    """Call the configured provider and return raw text.

    Only HTTP 503 and timeouts are retried (at most twice).  A 429 opens the
    circuit and is never retried.  The whole loop respects the per-turn budget;
    when it is exhausted :class:`LLMTurnBudgetExceeded` is raised so callers
    switch to their deterministic path.
    """
    if circuit.is_open():
        raise LLMUnavailable("LLM circuit is open after a quota error.")

    targets = _provider_targets(config)
    effective_timeout = timeout or config.llm_timeout
    budget = current_budget()
    retries = min(max(0, int(config.llm_max_retries)), MAX_RETRIES)
    last_error: BaseException | None = None

    for index, (provider, api_key, model, base_url) in enumerate(targets):
        attempts = retries + 1 if index == 0 else 1
        for attempt in range(attempts):
            if budget is not None:
                if budget.expired():
                    raise LLMTurnBudgetExceeded(
                        "The per-turn LLM time budget is exhausted."
                    )
                call_timeout = max(0.1, min(effective_timeout, budget.remaining()))
            else:
                call_timeout = effective_timeout
            logger.info(
                "LLM request provider=%s model=%s json=%s attempt=%d",
                provider,
                model,
                json_mode,
                attempt + 1,
            )
            record_model_call()
            try:
                raw = _invoke_provider(
                    provider,
                    api_key,
                    model,
                    base_url,
                    prompt,
                    json_mode,
                    temperature,
                    call_timeout,
                    config.llm_connect_timeout,
                )
                record_successful_call()
                return strip_code_fences(raw)
            except Exception as exc:  # noqa: BLE001 - classified below
                last_error = exc
                err_class = note_provider_failure(exc)
                if err_class == ErrorClass.PERMANENT:
                    raise LLMUnavailable(
                        f"Permanent LLM config error; deterministic mode active: {exc}"
                    ) from exc
                if err_class == ErrorClass.ACCOUNT_QUOTA:
                    raise LLMUnavailable(
                        "LLM quota/rate limit reached; deterministic mode active."
                    ) from exc
                if not is_retryable_error(exc) or attempt == attempts - 1:
                    break
                delay = retry_delay(attempt, exc)
                if budget is not None and budget.remaining() <= delay:
                    raise LLMTurnBudgetExceeded(
                        "The per-turn LLM time budget is exhausted."
                    )
                logger.warning(
                    "LLM request failed (%s); retrying in %.2fs",
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)
        if index == 0 and len(targets) > 1:
            if budget is not None and budget.expired():
                raise LLMTurnBudgetExceeded(
                    "The per-turn LLM time budget is exhausted."
                )
            logger.warning("Primary LLM model failed; trying the fallback model")

    raise LLMUnavailable("LLM provider failed after retries.") from last_error


def complete_json(
    prompt: str,
    *,
    temperature: float = 0.1,
    timeout: float | None = None,
    config: Settings = settings,
) -> dict[str, Any]:
    raw = complete(
        prompt,
        json_mode=True,
        temperature=temperature,
        timeout=timeout,
        config=config,
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMResponseError("LLM did not return valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise LLMResponseError("LLM JSON payload is not an object.")
    return parsed
