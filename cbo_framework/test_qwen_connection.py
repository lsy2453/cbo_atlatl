"""
Minimal Qwen connectivity test.

Set environment variables first:

    LLM_API_KEY=...
    # or use DashScope's standard variable:
    DASHSCOPE_API_KEY=...
    LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    LLM_MODEL=qwen-plus

Then run:

    python cbo_framework/test_qwen_connection.py
"""

import json
import os
import sys
import urllib.error
import urllib.request


def extract_json_object(text):
    """Extract the first JSON object from a model response."""
    text = (text or "").strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return json.loads(text[start:end])


def call_with_openai_sdk(api_key, base_url, model):
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=50,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict JSON emitter. Return only one JSON object, "
                    "with no markdown and no explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    'For a game action test, return exactly this object: '
                    '{"type":"pass"}'
                ),
            },
        ],
    )
    return response.choices[0].message.content


def call_with_raw_http(api_key, base_url, model):
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 50,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON emitter. Return only one JSON object, "
                    "with no markdown and no explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    'For a game action test, return exactly this object: '
                    '{"type":"pass"}'
                ),
            },
        ],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"]


def main():
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("Missing environment variable: LLM_API_KEY or DASHSCOPE_API_KEY")
        return 1

    base_url = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("LLM_MODEL", "qwen-plus")

    try:
        try:
            text = call_with_openai_sdk(api_key, base_url, model)
        except ImportError:
            text = call_with_raw_http(api_key, base_url, model)
    except Exception as exc:
        print(f"Qwen connection failed: {exc}")
        return 1

    print("Raw response:")
    print(text)
    try:
        parsed = extract_json_object(text)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Qwen responded, but no valid JSON action was found: {exc}")
        return 1
    if parsed != {"type": "pass"}:
        print("Qwen responded, but not with the expected action.")
        return 1
    print(f"Parsed action: {parsed}")
    print("Qwen connection OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
