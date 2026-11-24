"""Build the LangChain chat model used by the supervisor.

All supported providers speak the OpenAI chat-completions protocol, so a single
``ChatOpenAI`` client covers them:

* ``openai`` / ``openrouter`` / ``groq`` / ``custom`` -> their own base URL,
* ``gemini`` -> Google's OpenAI-compatible endpoint.

Credentials come exclusively from :mod:`agent_platform.llm`
(``.env`` / environment); nothing is hard-coded here.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import Any, Callable

import httpx
from langchain_openai import ChatOpenAI

from .. import llm
from ..config import Settings, settings

logger = logging.getLogger(__name__)

# Google exposes an OpenAI-compatible surface for the Gemini API.
_GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Sent to OpenRouter so the request is attributable; harmless elsewhere.
_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/scholarship-orchestrator",
    "X-Title": "Scholarship Orchestrator",
}


class AgentModelUnavailable(RuntimeError):
    """Raised when no provider is configured for the agent engine."""


class RetryingChatModel:
    """Retry transient provider failures, then try an optional fallback model.

    Wraps any chat-model-like object (``bind_tools`` / ``invoke``) so retries
    apply to both the tool-bound and plain calls made by the graph.  When every
    attempt fails the last provider error is re-raised; the service turns that
    into a deterministic answer.
    """

    def __init__(
        self,
        primary: Any,
        fallback: Any | None = None,
        max_retries: int = 3,
        sleep: Callable[[float], Any] | None = None,
        budget: llm.TurnBudget | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.max_retries = max(0, int(max_retries))
        self._sleep = sleep or time.sleep
        self.budget = budget

    def bind_tools(self, tools: Any, **kwargs: Any) -> "RetryingChatModel":
        binder = getattr(self.primary, "bind_tools", None)
        primary = binder(tools, **kwargs) if binder else self.primary
        fallback = None
        if self.fallback is not None:
            fallback_binder = getattr(self.fallback, "bind_tools", None)
            fallback = (
                fallback_binder(tools, **kwargs)
                if fallback_binder
                else self.fallback
            )
        return RetryingChatModel(
            primary=primary,
            fallback=fallback,
            max_retries=self.max_retries,
            sleep=self._sleep,
            budget=self.budget,
        )

    @staticmethod
    def _budget() -> llm.TurnBudget | None:
        return llm.current_budget()

    def _check_budget(self, budget: llm.TurnBudget | None) -> None:
        if budget is not None and budget.expired():
            raise llm.LLMTurnBudgetExceeded(
                "The per-turn LLM time budget is exhausted."
            )

    def _retry_limit(self) -> int:
        return min(self.max_retries, llm.MAX_RETRIES)

    def _invoke_primary(
        self, messages: Any, config: Any, kwargs: dict[str, Any]
    ) -> Any:
        budget = self.budget or self._budget()
        retries = self._retry_limit()
        last_error: BaseException | None = None
        for attempt in range(retries + 1):
            self._check_budget(budget)
            llm.record_model_call()
            try:
                res = self.primary.invoke(messages, config=config, **kwargs)
                llm.record_successful_call()
                return res
            except Exception as exc:  # noqa: BLE001 - classified below
                last_error = exc
                err_class = llm.note_provider_failure(exc)
                if err_class in (llm.ErrorClass.PERMANENT, llm.ErrorClass.ACCOUNT_QUOTA):
                    raise
                if not llm.is_retryable_error(exc) or attempt == retries:
                    break
                delay = llm.retry_delay(attempt, exc)
                if budget is not None and budget.remaining() <= delay:
                    raise llm.LLMTurnBudgetExceeded(
                        "The per-turn LLM time budget is exhausted."
                    )
                logger.warning(
                    "Agent model failed (%s); retrying in %.2fs",
                    type(exc).__name__,
                    delay,
                )
                self._sleep(delay)
        if self.fallback is not None:
            self._check_budget(budget)
            logger.warning("Agent model failed; trying the fallback model")
            llm.record_model_call()
            try:
                res = self.fallback.invoke(messages, config=config, **kwargs)
                llm.record_successful_call()
                return res
            except Exception as exc:
                llm.note_provider_failure(exc)
                raise
        raise last_error  # type: ignore[misc]

    def invoke(self, messages: Any, config: Any = None, **kwargs: Any) -> Any:
        return self._invoke_primary(messages, config, kwargs)

    async def ainvoke(
        self, messages: Any, config: Any = None, **kwargs: Any
    ) -> Any:
        budget = self.budget or self._budget()
        retries = self._retry_limit()
        last_error: BaseException | None = None
        for attempt in range(retries + 1):
            self._check_budget(budget)
            llm.record_model_call()
            try:
                res = await self.primary.ainvoke(messages, config=config, **kwargs)
                llm.record_successful_call()
                return res
            except Exception as exc:  # noqa: BLE001 - classified below
                last_error = exc
                err_class = llm.note_provider_failure(exc)
                if err_class in (llm.ErrorClass.PERMANENT, llm.ErrorClass.ACCOUNT_QUOTA):
                    raise
                if not llm.is_retryable_error(exc) or attempt == retries:
                    break
                delay = llm.retry_delay(attempt, exc)
                if budget is not None and budget.remaining() <= delay:
                    raise llm.LLMTurnBudgetExceeded(
                        "The per-turn LLM time budget is exhausted."
                    )
                logger.warning(
                    "Agent model failed (%s); retrying in %.2fs",
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
        if self.fallback is not None:
            self._check_budget(budget)
            logger.warning("Agent model failed; trying the fallback model")
            llm.record_model_call()
            try:
                res = await self.fallback.ainvoke(
                    messages, config=config, **kwargs
                )
                llm.record_successful_call()
                return res
            except Exception as exc:
                llm.note_provider_failure(exc)
                raise
        raise last_error  # type: ignore[misc]


def _build_single_model(
    provider: str,
    api_key: str,
    model: str,
    base_url: str,
    config: Settings,
) -> ChatOpenAI:
    if provider == "gemini":
        base_url = _GEMINI_OPENAI_BASE
    if not base_url:  # pragma: no cover - resolve_provider guarantees this
        raise AgentModelUnavailable("No base URL resolved for the LLM provider.")

    headers: dict[str, Any] = (
        dict(_OPENROUTER_HEADERS) if provider == "openrouter" else {}
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "timeout": httpx.Timeout(
            config.llm_timeout, connect=config.llm_connect_timeout
        ),
        # RetryingChatModel owns the retry policy (LLM_MAX_RETRIES).
        "max_retries": 0,
        "default_headers": headers,
    }
    # Gemini 3 ignores temperature today and will reject it later.
    if not model.lower().startswith("gemini-3"):
        kwargs["temperature"] = 0
    return ChatOpenAI(**kwargs)


def build_chat_model(config: Settings = settings) -> RetryingChatModel:
    """Return a tool-calling chat model with retries and an opt-in fallback."""
    try:
        provider, api_key, model, base_url = llm.resolve_provider(config)
    except llm.LLMUnavailable as exc:
        raise AgentModelUnavailable(str(exc)) from exc

    logger.info("Agent model provider=%s model=%s", provider, model)
    primary = _build_single_model(provider, api_key, model, base_url, config)

    fallback = None
    fallback_model = config.llm_fallback_model.strip()
    if fallback_model and fallback_model != model:
        try:
            f_provider, f_key, f_model, f_base = llm.resolve_provider(
                replace(config, llm_model=fallback_model)
            )
            logger.info("Agent fallback model provider=%s model=%s", f_provider, f_model)
            fallback = _build_single_model(
                f_provider, f_key, f_model, f_base, config
            )
        except (llm.LLMUnavailable, AgentModelUnavailable):
            logger.warning("Agent fallback model could not be resolved; ignoring it")

    return RetryingChatModel(
        primary=primary,
        fallback=fallback,
        max_retries=config.llm_max_retries,
    )


def model_label(config: Settings = settings) -> str:
    """Human-readable provider:model label, or ``deterministic`` when unset."""
    try:
        provider, _, model, _ = llm.resolve_provider(config)
    except llm.LLMUnavailable:
        return "deterministic"
    return f"{provider}:{model}"
