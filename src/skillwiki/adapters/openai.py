# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""ChatModel over any OpenAI-compatible chat completions API (``pip install skillwiki[openai]``). Calls tools natively
when offered."""

from __future__ import annotations

import json

from ..model import ModelRequest, ModelResponse, ToolInvocation


class OpenAIChatModel:
    supports_tools = True

    def __init__(self, model: str, *, client=None, **client_kwargs):
        import openai

        self.model = model
        self.client = client or openai.AsyncOpenAI(**client_kwargs)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        content: list[dict] = [{"type": "text", "text": request.text}]
        for image in request.images:
            content.append({"type": "image_url", "image_url": {"url": f"data:{image.media_type};base64,{image.data_base64}"}})
        kwargs: dict = {}
        if request.tools:
            kwargs["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                               for t in request.tools]
            kwargs["tool_choice"] = "required"
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=request.max_tokens,
            messages=[{"role": "system", "content": request.system}, {"role": "user", "content": content}],
            **kwargs,
        )
        choice = response.choices[0]
        calls = []
        for call in getattr(choice.message, "tool_calls", None) or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolInvocation(call.function.name, args if isinstance(args, dict) else {}))
        usage = getattr(response, "usage", None)
        return ModelResponse(text=choice.message.content or "", usage=usage.model_dump() if hasattr(usage, "model_dump") else {}, tool_calls=tuple(calls))


__all__ = ["OpenAIChatModel"]
