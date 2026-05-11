from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from .config import ModelConfig


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool = False, status_code: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


@dataclass
class ProviderResponse:
    raw_response: str
    latency_ms: int
    token_usage: dict[str, Any] | None
    api_retry_count: int
    response_format_mode: str


class BaseProvider:
    def complete(
        self,
        model: ModelConfig,
        image_data_url: str,
        response_format_mode: str,
        timeout_seconds: float,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        schema_instruction: str | None = None,
        response_format_payload: dict[str, Any] | None = None,
        mock_prediction: dict[str, Any] | None = None,
        reference_images: list[tuple[str, str]] | None = None,
    ) -> ProviderResponse:
        raise NotImplementedError


class MockProvider(BaseProvider):
    def __init__(self, invalid_first: bool = False) -> None:
        self.invalid_first = invalid_first
        self.calls = 0

    def complete(
        self,
        model: ModelConfig,
        image_data_url: str,
        response_format_mode: str,
        timeout_seconds: float,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        schema_instruction: str | None = None,
        response_format_payload: dict[str, Any] | None = None,
        mock_prediction: dict[str, Any] | None = None,
        reference_images: list[tuple[str, str]] | None = None,
    ) -> ProviderResponse:
        self.calls += 1
        start = time.perf_counter()
        if self.invalid_first and self.calls == 1:
            raw = "not-json"
        else:
            raw = json.dumps(mock_prediction or {}, ensure_ascii=False)
        return ProviderResponse(
            raw_response=raw,
            latency_ms=max(1, round((time.perf_counter() - start) * 1000)),
            token_usage=None,
            api_retry_count=0,
            response_format_mode=response_format_mode,
        )


class OpenAICompatibleProvider(BaseProvider):
    def __init__(self, name: str, base_url: str, api_key: str | None) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def complete(
        self,
        model: ModelConfig,
        image_data_url: str,
        response_format_mode: str,
        timeout_seconds: float,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        schema_instruction: str | None = None,
        response_format_payload: dict[str, Any] | None = None,
        mock_prediction: dict[str, Any] | None = None,
        reference_images: list[tuple[str, str]] | None = None,
    ) -> ProviderResponse:
        try:
            import requests
        except ImportError as exc:
            raise ProviderError("requests is not installed; run pip install -r requirements.txt") from exc

        payload = self._make_payload(
            model,
            image_data_url,
            response_format_mode,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_instruction=schema_instruction,
            response_format_payload=response_format_payload,
            reference_images=reference_images,
        )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.name == "openrouter":
            headers["HTTP-Referer"] = "https://local.vlm-eval"
            headers["X-Title"] = "ice-cream-vlm-eval"

        api_retry_count = 0
        start = time.perf_counter()
        last_error: ProviderError | None = None
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = ProviderError(f"request failed: {exc}", retryable=True)
                if attempt < 2:
                    api_retry_count += 1
                    time.sleep(0.8 * (2**attempt))
                    continue
                raise last_error from exc

            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                api_retry_count += 1
                time.sleep(0.8 * (2**attempt))
                continue
            if response.status_code >= 400:
                raise ProviderError(
                    f"HTTP {response.status_code}: {response.text[:1000]}",
                    retryable=response.status_code in {429, 500, 502, 503, 504},
                    status_code=response.status_code,
                )

            try:
                body = response.json()
            except ValueError as exc:
                raise ProviderError(f"provider returned non-JSON HTTP body: {response.text[:500]}") from exc

            raw = _extract_message_content(body)
            latency_ms = max(1, round((time.perf_counter() - start) * 1000))
            return ProviderResponse(
                raw_response=raw,
                latency_ms=latency_ms,
                token_usage=_extract_usage(body),
                api_retry_count=api_retry_count,
                response_format_mode=response_format_mode,
            )

        if last_error:
            raise last_error
        raise ProviderError("request failed after retries", retryable=True)

    def _make_payload(
        self,
        model: ModelConfig,
        image_data_url: str,
        response_format_mode: str,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        schema_instruction: str | None = None,
        response_format_payload: dict[str, Any] | None = None,
        reference_images: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        system_text = system_prompt or ""
        user_text = user_prompt or ""
        instruction_text = schema_instruction or ""
        if response_format_mode in {"json_object", "none"}:
            user_text = f"{user_text}\n\n{instruction_text}"
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        if reference_images:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "REFERENCE CATALOG FOLLOWS. Inspect these images before the target image. "
                        "Each reference label names the KIK SKU group and the JSON field it helps decide. "
                        "Use references as positive visual definitions for KIK packaging, logos, colors, "
                        "and SKU form factors. Do not count products from reference images in "
                        "kik_sku_count, kik_share_percent, status_score, or any "
                        "target-only field."
                    ),
                }
            )
        for index, (label, reference_data_url) in enumerate(reference_images or [], start=1):
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"REFERENCE IMAGE {index}: {label}. "
                            "Use this as the visual catalog definition for the named SKU group(s). "
                            "Later, compare visible products in the target image against this reference. "
                            "Do not score this reference image."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": reference_data_url}},
                ]
            )
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "TARGET IMAGE TO ANALYZE. Return JSON for this target image only, "
                        "using the reference images above only to recognize KIK products and SKU groups. "
                        "Before setting each has_* SKU-family field, compare the target products with "
                        "the corresponding reference catalog image label. "
                        "Return only one complete JSON object. Do not explain, reason step by step, "
                        "use markdown, or output text before or after the JSON. The first character "
                        "must be { and the last character must be }."
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        )
        payload: dict[str, Any] = {
            "model": model.provider_model,
            "messages": [
                {"role": "system", "content": system_text},
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "temperature": model.temperature,
            "max_tokens": model.max_output_tokens,
        }
        if response_format_mode == "json_schema":
            if response_format_payload:
                payload["response_format"] = response_format_payload
        elif response_format_mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload


def _extract_message_content(body: dict[str, Any]) -> str:
    choice = body.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if content is None:
        for key in ("reasoning", "reasoning_content", "text"):
            value = message.get(key) or choice.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(body, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _extract_usage(body: dict[str, Any]) -> dict[str, Any]:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {
        "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
        "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def create_provider(provider_name: str) -> BaseProvider:
    if provider_name == "mock":
        return MockProvider()
    if provider_name == "deepinfra":
        api_key = os.getenv("DEEPINFRA_API_KEY")
        if not api_key:
            raise ProviderError("DEEPINFRA_API_KEY is not set")
        base_url = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
        return OpenAICompatibleProvider("deepinfra", base_url, api_key)
    if provider_name == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ProviderError("OPENROUTER_API_KEY is not set")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return OpenAICompatibleProvider("openrouter", base_url, api_key)
    if provider_name == "local_vllm":
        base_url = os.getenv("LOCAL_VLLM_BASE_URL")
        if not base_url:
            raise ProviderError("LOCAL_VLLM_BASE_URL is not set")
        return OpenAICompatibleProvider("local_vllm", base_url, os.getenv("LOCAL_VLLM_API_KEY"))
    raise ProviderError(f"Unsupported provider: {provider_name}")


def response_format_candidates(model: ModelConfig) -> list[str]:
    configured = model.response_format
    if model.provider == "openrouter" and configured == "json_schema":
        return ["json_schema", "json_object", "none"]
    if configured in {"json_schema", "json_object", "none"}:
        return [configured]
    return ["none"]
