"""Interoperability tests verifying OpenAI SDK wire-protocol and client compatibility."""

from __future__ import annotations

from typing import Any

import httpx

from alienese.api.app import create_app


async def test_openai_sdk_wire_protocol_interoperability() -> None:
    """Verify compatibility with the exact headers and payloads sent by the OpenAI SDK."""
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    sdk_headers = {
        "Authorization": "Bearer " + "fake_local_sdk_key_123456",
        "User-Agent": "AsyncOpenAI/Python 1.54.0",
        "X-Stainless-Lang": "python",
        "X-Stainless-Package-Version": "1.54.0",
        "X-Stainless-OS": "Linux",
        "X-Stainless-Arch": "x64",
        "X-Stainless-Runtime": "CPython",
        "Idempotency-Key": "stainless-retry-key-001",
    }

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://alienese.local/v1",
        headers=sdk_headers,
    ) as client:
        models_resp = await client.get("/models")
        assert models_resp.status_code == 200
        models_json = models_resp.json()
        assert models_json["object"] == "list"
        assert any(m["id"] == "alienese-default" for m in models_json["data"])

        chat_resp = await client.post(
            "/chat/completions",
            json={
                "model": "alienese-default",
                "messages": [
                    {"role": "system", "content": "Follow repository rules."},
                    {"role": "user", "content": "Verify OpenAI SDK wire compatibility."},
                ],
                "temperature": 0.0,
                "max_completion_tokens": 64,
            },
        )
        assert chat_resp.status_code == 200
        body = chat_resp.json()
        assert body["id"].startswith("chatcmpl-op_")
        assert body["object"] == "chat.completion"
        assert isinstance(body["created"], int)
        assert body["model"] == "alienese-default"
        assert len(body["choices"]) == 1
        assert body["choices"][0]["index"] == 0
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert "Verify OpenAI SDK wire compatibility" in body["choices"][0]["message"]["content"]
        assert set(body["usage"].keys()) == {"prompt_tokens", "completion_tokens", "total_tokens"}


async def test_openai_python_sdk_client_if_installed() -> None:
    """If the `openai` package is installed in the environment, test AsyncOpenAI directly."""
    try:
        import openai  # type: ignore[import-not-found]
    except ImportError:
        return

    app = create_app()
    http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://alienese.local/v1",
    )
    client: Any = openai.AsyncOpenAI(
        api_key="fake-local-key",
        base_url="http://alienese.local/v1",
        http_client=http_client,
    )

    async with http_client:
        models_page = await client.models.list()
        assert any(m.id == "alienese-default" for m in models_page.data)

        completion = await client.chat.completions.create(
            model="alienese-default",
            messages=[{"role": "user", "content": "Hello from real AsyncOpenAI client"}],
        )
        assert completion.id.startswith("chatcmpl-op_")
        assert completion.choices[0].finish_reason == "stop"
        assert "Hello from real AsyncOpenAI client" in completion.choices[0].message.content
