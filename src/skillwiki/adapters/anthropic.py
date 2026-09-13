# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""ChatModel over the Anthropic SDK (``pip install skillwiki[anthropic]``). Calls tools natively when offered."""

from __future__ import annotations

from ..model import ModelRequest, ModelResponse, ToolInvocation


class AnthropicChatModel:
    supports_tools = True

    def __init__(self, model: str, *, client=None, **client_kwargs):
        import anthropic

        self.model = model
        self.client = client or anthropic.AsyncAnthropic(**client_kwargs)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        content: list[dict] = [{"type": "text", "text": request.text}]
        for image in request.images:
            content.append({"type": "image", "source": {"type": "base64", "media_type": image.media_type, "data": image.data_base64}})
        kwargs: dict = {}
        if request.tools:
            kwargs["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in request.tools]
            kwargs["tool_choice"] = {"type": "any"}  # the proposer requires one tool call per turn
        message = await self.client.messages.create(
            model=self.model, max_tokens=request.max_tokens, system=request.system, messages=[{"role": "user", "content": content}], **kwargs
        )
        text = "".join(getattr(block, "text", "") for block in message.content)
        calls = tuple(
            ToolInvocation(block.name, dict(block.input or {})) for block in message.content if getattr(block, "type", "") == "tool_use"
        )
        usage = getattr(message, "usage", None)
        return ModelResponse(text=text, usage=usage.model_dump() if hasattr(usage, "model_dump") else {}, tool_calls=calls)


__all__ = ["AnthropicChatModel"]
