from __future__ import annotations

import json
import os
import re
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
        image_prompt_contract: dict[str, str] | None = None,
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
        image_prompt_contract: dict[str, str] | None = None,
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
        image_prompt_contract: dict[str, str] | None = None,
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
            image_prompt_contract=image_prompt_contract,
        )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.name == "openrouter":
            headers["HTTP-Referer"] = "https://local.vlm-eval"
            headers["X-Title"] = "ice-cream-vlm-eval"

        max_attempts = max(1, int(os.getenv("VLM_EVAL_API_MAX_ATTEMPTS", "5")))
        retry_base_seconds = max(0.1, float(os.getenv("VLM_EVAL_RETRY_BASE_SECONDS", "10")))
        retry_max_seconds = max(retry_base_seconds, float(os.getenv("VLM_EVAL_RETRY_MAX_SECONDS", "120")))
        api_retry_count = 0
        start = time.perf_counter()
        last_error: ProviderError | None = None
        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = ProviderError(f"request failed: {exc}", retryable=True)
                if attempt < max_attempts - 1:
                    api_retry_count += 1
                    time.sleep(_retry_sleep_seconds(attempt, retry_base_seconds, retry_max_seconds))
                    continue
                raise last_error from exc

            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                api_retry_count += 1
                time.sleep(
                    _retry_sleep_seconds(
                        attempt,
                        retry_base_seconds,
                        retry_max_seconds,
                        retry_after_seconds=_retry_after_seconds(response),
                    )
                )
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
        image_prompt_contract: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        system_text = system_prompt or ""
        user_text = user_prompt or ""
        instruction_text = schema_instruction or ""
        if response_format_mode in {"json_object", "none"}:
            user_text = f"{user_text}\n\n{instruction_text}"

        reference_images = reference_images or []
        contract = image_prompt_contract or _default_image_prompt_contract()
        image_map_header = contract.get("image_map_header", "IMAGE MAP")
        target_map_line = contract.get("target_map_line", "TARGET_00 = target retail equipment photo to analyze.")
        image_map = "\n".join(
            [
                image_map_header,
                f"- {target_map_line}",
                *[
                    f"- {_reference_label_for_map(index, label)}"
                    for index, (label, _reference_data_url) in enumerate(reference_images, start=1)
                ],
            ]
        )
        role_rules = contract["role_rules"]
        target_blocks = [
            {"type": "text", "text": contract["target_intro"]},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
        content: list[dict[str, Any]] = [{"type": "text", "text": f"{image_map}\n\n{role_rules}"}]
        if contract.get("target_position") == "before_references":
            content.extend(target_blocks)
        for index, (label, reference_data_url) in enumerate(reference_images or [], start=1):
            reference_id = _reference_id(index, label)
            reference_map_line = _reference_label_for_map(index, label)
            content.extend(
                [
                    {
                        "type": "text",
                        "text": contract["reference_intro_template"].format(
                            reference_id=reference_id,
                            reference_map_line=reference_map_line,
                            label=label,
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": reference_data_url}},
                ]
            )
        if contract.get("target_position") != "before_references":
            content.extend(target_blocks)
        content.append(
            {
                "type": "text",
                "text": (
                    f"{contract['final_task_intro']}\n\n"
                    f"{user_text}"
                ),
            }
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


class GoogleAIStudioProvider(BaseProvider):
    def __init__(self, base_url: str, api_key: str) -> None:
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
        image_prompt_contract: dict[str, str] | None = None,
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
            reference_images=reference_images,
            image_prompt_contract=image_prompt_contract,
        )
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        max_attempts = max(1, int(os.getenv("VLM_EVAL_API_MAX_ATTEMPTS", "5")))
        retry_base_seconds = max(0.1, float(os.getenv("VLM_EVAL_RETRY_BASE_SECONDS", "10")))
        retry_max_seconds = max(retry_base_seconds, float(os.getenv("VLM_EVAL_RETRY_MAX_SECONDS", "120")))
        api_retry_count = 0
        start = time.perf_counter()
        last_error: ProviderError | None = None
        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    self._generate_content_url(model),
                    headers=headers,
                    json=payload,
                    timeout=timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = ProviderError(f"request failed: {exc}", retryable=True)
                if attempt < max_attempts - 1:
                    api_retry_count += 1
                    time.sleep(_retry_sleep_seconds(attempt, retry_base_seconds, retry_max_seconds))
                    continue
                raise last_error from exc

            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                api_retry_count += 1
                time.sleep(
                    _retry_sleep_seconds(
                        attempt,
                        retry_base_seconds,
                        retry_max_seconds,
                        retry_after_seconds=_retry_after_seconds(response),
                    )
                )
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

            raw = _extract_google_text(body)
            latency_ms = max(1, round((time.perf_counter() - start) * 1000))
            return ProviderResponse(
                raw_response=raw,
                latency_ms=latency_ms,
                token_usage=_extract_google_usage(body),
                api_retry_count=api_retry_count,
                response_format_mode=response_format_mode,
            )

        if last_error:
            raise last_error
        raise ProviderError("request failed after retries", retryable=True)

    def _generate_content_url(self, model: ModelConfig) -> str:
        model_name = model.provider_model
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"
        return f"{self.base_url}/{model_name}:generateContent"

    def _make_payload(
        self,
        model: ModelConfig,
        image_data_url: str,
        response_format_mode: str,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        schema_instruction: str | None = None,
        reference_images: list[tuple[str, str]] | None = None,
        image_prompt_contract: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        user_text = user_prompt or ""
        instruction_text = schema_instruction or ""
        if response_format_mode in {"json_object", "none"}:
            user_text = f"{user_text}\n\n{instruction_text}"

        reference_images = reference_images or []
        contract = image_prompt_contract or _default_image_prompt_contract()
        image_map_header = contract.get("image_map_header", "IMAGE MAP")
        target_map_line = contract.get("target_map_line", "TARGET_00 = target retail equipment photo to analyze.")
        image_map = "\n".join(
            [
                image_map_header,
                f"- {target_map_line}",
                *[
                    f"- {_reference_label_for_map(index, label)}"
                    for index, (label, _reference_data_url) in enumerate(reference_images, start=1)
                ],
            ]
        )
        target_parts = [
            {"text": contract["target_intro"]},
            _data_url_to_google_inline_data(image_data_url),
        ]
        parts: list[dict[str, Any]] = [{"text": f"{image_map}\n\n{contract['role_rules']}"}]
        if contract.get("target_position") == "before_references":
            parts.extend(target_parts)
        for index, (label, reference_data_url) in enumerate(reference_images, start=1):
            reference_id = _reference_id(index, label)
            reference_map_line = _reference_label_for_map(index, label)
            parts.extend(
                [
                    {
                        "text": contract["reference_intro_template"].format(
                            reference_id=reference_id,
                            reference_map_line=reference_map_line,
                            label=label,
                        )
                    },
                    _data_url_to_google_inline_data(reference_data_url),
                ]
            )
        if contract.get("target_position") != "before_references":
            parts.extend(target_parts)
        parts.append({"text": f"{contract['final_task_intro']}\n\n{user_text}"})

        generation_config: dict[str, Any] = {
            "temperature": model.temperature,
            "max_output_tokens": model.max_output_tokens,
        }
        if response_format_mode in {"json_schema", "json_object"}:
            generation_config["response_mime_type"] = "application/json"

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generation_config": generation_config,
        }
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
        return payload


def _default_image_prompt_contract() -> dict[str, str]:
    return {
        "target_id": "TARGET_00",
        "target_position": "after_references",
        "target_map_line": "TARGET_00 = target retail equipment photo to analyze.",
        "role_rules": (
            "Use only TARGET_00 and REF_01..REF_07 as image IDs. "
            "Analyze only TARGET_00 as the retail equipment photo. "
            "Use REF_* images only as visual references for KIK packaging, logo, colors, "
            "and SKU group form factors. Never count REF_* products in kik_sku_count, "
            "kik_share_percent, status_score, POSM, empty sections, or competitor mixing. "
            "Do not use positional phrases for image identity; use only canonical IDs."
        ),
        "target_intro": (
            "TARGET_00 image follows. This is the only real retail equipment photo "
            "to analyze and score."
        ),
        "reference_intro_template": (
            "{reference_id} reference image follows. {reference_map_line}. "
            "Use {reference_id} only as the visual catalog definition for its named "
            "KIK SKU group and JSON field. Do not score this reference image."
        ),
        "final_task_intro": (
            "FINAL TASK: analyze TARGET_00 only. For each has_* SKU-family field, "
            "compare visible TARGET_00 products with the matching REF_* visual definition "
            "from the IMAGE MAP. Return only one complete JSON object. Do not explain, "
            "reason step by step, use markdown, or output text before or after the JSON. "
            "The first character must be { and the last character must be }."
        ),
    }


def _reference_id(index: int, label: str) -> str:
    match = re.search(r"\bREF_[A-Za-z0-9_]+\b", label)
    if match:
        return match.group(0)
    return f"REF_{index:02d}"


def _reference_label_for_map(index: int, label: str) -> str:
    reference_id = _reference_id(index, label)
    cleaned = label.strip().rstrip(".")
    if cleaned.startswith(f"{reference_id} ="):
        return cleaned
    return f"{reference_id} = {cleaned}"


def _retry_sleep_seconds(
    attempt: int,
    base_seconds: float,
    max_seconds: float,
    retry_after_seconds: float | None = None,
) -> float:
    backoff = min(max_seconds, base_seconds * (2**attempt))
    if retry_after_seconds is None:
        return backoff
    return min(max_seconds, max(backoff, retry_after_seconds))


def _retry_after_seconds(response: Any) -> float | None:
    value = response.headers.get("retry-after") if hasattr(response, "headers") else None
    if value:
        try:
            return float(value)
        except ValueError:
            return None
    try:
        body = response.json()
    except ValueError:
        return None
    metadata = body.get("error", {}).get("metadata", {}) if isinstance(body, dict) else {}
    retry_after = metadata.get("retry_after_seconds") or metadata.get("retry_after_seconds_raw")
    try:
        return float(retry_after)
    except (TypeError, ValueError):
        return None


def _data_url_to_google_inline_data(data_url: str) -> dict[str, Any]:
    header, separator, data = data_url.partition(",")
    match = re.match(r"^data:([^;]+);base64$", header)
    if separator != "," or not match:
        raise ProviderError("Expected base64 data URL for Gemini inline image")
    return {"inline_data": {"mime_type": match.group(1), "data": data}}


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


def _extract_google_text(body: dict[str, Any]) -> str:
    for candidate in body.get("candidates", []):
        content = candidate.get("content") if isinstance(candidate, dict) else None
        parts = content.get("parts", []) if isinstance(content, dict) else []
        texts = [part.get("text") for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)]
        if texts:
            return "\n".join(texts)
    raise ProviderError(f"provider returned no text: {json.dumps(body, ensure_ascii=False)[:1000]}")


def _extract_google_usage(body: dict[str, Any]) -> dict[str, Any]:
    usage = body.get("usageMetadata")
    if not isinstance(usage, dict):
        return {}
    return {
        "input_tokens": usage.get("promptTokenCount"),
        "output_tokens": usage.get("candidatesTokenCount"),
        "total_tokens": usage.get("totalTokenCount"),
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
    if provider_name == "google_aistudio":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ProviderError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")
        base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
        return GoogleAIStudioProvider(base_url, api_key)
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
    if model.provider == "google_aistudio" and configured == "json_schema":
        return ["json_object", "none"]
    if configured in {"json_schema", "json_object", "none"}:
        return [configured]
    return ["none"]
