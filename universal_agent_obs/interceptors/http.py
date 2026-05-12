"""
interceptors/http.py — Transport-level fallback interceptor.
Patches httpx.Client and requests.Session to capture any LLM call
that isn't covered by a framework-specific callback.
"""

import json, time, uuid
from ..core import Span, emit, get_or_new_trace, _current_span, detect_provider

LLM_HOSTS = (
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.cohere.ai",
    "api.mistral.ai",
    "api.together.xyz",
    "api.groq.com",
    "openrouter.ai",
    "openai.azure.com",
    "models.inference.ai.azure.com",
)

LOCAL_LLM_HOSTS = (
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
)

LOCAL_LLM_PATHS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/api/generate",
    "/api/chat",
)

def _is_llm_url(url: str) -> bool:
    text = str(url)
    if any(h in text for h in LLM_HOSTS):
        return True
    return any(h in text for h in LOCAL_LLM_HOSTS) and any(p in text for p in LOCAL_LLM_PATHS)

def _parse_request(request) -> dict:
    try:
        if hasattr(request, "content"):
            return json.loads(request.content) or {}
        if isinstance(request, tuple):
            return (request[2] or {}).get("json", {})
    except Exception:
        pass
    return {}

def _parse_response(response) -> dict:
    try:
        return response.json()
    except Exception:
        return {}


def _detect_framework(req_body: dict, resp_body: dict) -> str:
    from ..core import _current_framework
    fw = _current_framework()
    if fw:
        return fw
    if "messages" in req_body or "choices" in resp_body or "usage" in resp_body:
        return "openai-sdk"
    if "contents" in req_body or "candidates" in resp_body or "usageMetadata" in resp_body:
        return "google-genai"
    return "custom"


def _extract_openai_response_content(resp_body: dict):
    choices = resp_body.get("choices") or []
    if not choices:
        return None
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if content is not None:
        return content
    tool_calls = message.get("tool_calls")
    if tool_calls:
        return {"tool_calls": tool_calls}
    return message or None

def _build_span(req_body: dict, resp_body: dict, latency_ms: float, status: int) -> Span:
    model   = req_body.get("model", "") or resp_body.get("modelVersion", "")
    framework = _detect_framework(req_body, resp_body)

    # ── Token usage: OpenAI format or Gemini format ──
    tokens  = resp_body.get("usage", {})
    if not tokens:
        # Gemini API uses usageMetadata with different key names
        gemini_usage = resp_body.get("usageMetadata", {})
        if gemini_usage:
            tokens = {
                "prompt_tokens": gemini_usage.get("promptTokenCount"),
                "completion_tokens": gemini_usage.get("candidatesTokenCount"),
                "total_tokens": gemini_usage.get("totalTokenCount"),
            }

    # ── Response content: OpenAI format or Gemini format ──
    content = _extract_openai_response_content(resp_body) or resp_body.get("content", [])
    if not content:
        # Gemini API uses candidates[].content.parts[].text
        candidates = resp_body.get("candidates", [])
        if candidates:
            parts = (candidates[0].get("content") or {}).get("parts", [])
            text_parts = [p.get("text", "") for p in parts if p.get("text")]
            if text_parts:
                content = "\n".join(text_parts)

    # ── Messages/contents: OpenAI uses "messages", Gemini uses "contents" ──
    messages = req_body.get("messages") or req_body.get("contents")

    return Span(
        trace_id    = get_or_new_trace(),
        parent_span = _current_span(),
        event       = "llm_end" if status == 200 else "llm_error",
        resource    = "llm",
        framework   = framework,
        model       = model,
        provider    = detect_provider(model),
        messages    = messages,
        response    = {"content": content} if content else None,
        tokens      = {
            "prompt":     tokens.get("prompt_tokens") or tokens.get("input_tokens"),
            "completion": tokens.get("completion_tokens") or tokens.get("output_tokens"),
            "total":      tokens.get("total_tokens"),
        } if tokens else None,
        latency_ms  = latency_ms,
        error       = resp_body.get("error", {}).get("message") if status != 200 else None,
        meta        = {"capture_via": "http_intercept", "status_code": status},
    )


def patch_httpx():
    try:
        import httpx
        from functools import wraps
        _orig = httpx.Client.send

        @wraps(_orig)
        def _send(self, request, **kw):
            if not _is_llm_url(request.url) or _current_span():
                return _orig(self, request, **kw)
            req_body = _parse_request(request)
            start_span_id = str(uuid.uuid4())
            trace_id = get_or_new_trace()
            emit(Span(
                span_id=start_span_id,
                trace_id=trace_id,
                event="llm_start",
                resource="llm",
                framework=_detect_framework(req_body, {}),
                model=req_body.get("model", ""),
                messages=req_body.get("messages") or req_body.get("contents"),
                meta={"capture_via": "http_intercept"},
            ))
            t0       = time.perf_counter()
            response = _orig(self, request, **kw)
            latency  = (time.perf_counter() - t0) * 1000
            resp_body= _parse_response(response)
            span     = _build_span(req_body, resp_body, latency, response.status_code)
            span.parent_span = start_span_id
            emit(span)
            return response

        httpx.Client.send = _send

        # Also patch AsyncClient
        _orig_async = httpx.AsyncClient.send

        async def _async_send(self, request, **kw):
            if not _is_llm_url(request.url) or _current_span():
                return await _orig_async(self, request, **kw)
            req_body = _parse_request(request)
            start_span_id = str(uuid.uuid4())
            trace_id = get_or_new_trace()
            emit(Span(
                span_id=start_span_id,
                trace_id=trace_id,
                event="llm_start",
                resource="llm",
                framework=_detect_framework(req_body, {}),
                model=req_body.get("model", ""),
                messages=req_body.get("messages") or req_body.get("contents"),
                meta={"capture_via": "http_intercept"},
            ))
            t0       = time.perf_counter()
            response = await _orig_async(self, request, **kw)
            latency  = (time.perf_counter() - t0) * 1000
            resp_body= _parse_response(response)
            span     = _build_span(req_body, resp_body, latency, response.status_code)
            span.parent_span = start_span_id
            emit(span)
            return response

        httpx.AsyncClient.send = _async_send
        return True
    except ImportError:
        return False


def patch_requests():
    try:
        import requests
        from functools import wraps
        _orig = requests.Session.request

        @wraps(_orig)
        def _request(self, method, url, **kw):
            if not _is_llm_url(url) or _current_span():
                return _orig(self, method, url, **kw)
            t0       = time.perf_counter()
            response = _orig(self, method, url, **kw)
            latency  = (time.perf_counter() - t0) * 1000
            req_body = (kw.get("json") or {})
            try:
                resp_body = response.json()
            except Exception:
                resp_body = {}
            span = _build_span(req_body, resp_body, latency, response.status_code)
            emit(span)
            return response

        requests.Session.request = _request
        return True
    except ImportError:
        return False


def install():
    httpx_ok = patch_httpx()
    requests_ok = patch_requests()
    return httpx_ok or requests_ok
