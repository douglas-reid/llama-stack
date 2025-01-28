# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List

from llama_models.llama3.api.datatypes import Role
from ollama import AsyncClient

from llama_stack.apis.inference.inference import Message, UserMessage
from llama_stack.apis.safety import Safety
from llama_stack.apis.safety.safety import (
    RunShieldResponse,
    SafetyViolation,
    ViolationLevel,
)
from llama_stack.apis.shields import Shield
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from llama_stack.providers.remote.safety.shield_gemma.config import ShieldGemmaConfig


CANNED_RESPONSE_TEXT = (
    "I can't process that request. Perhaps you could try rephrasing or submitting a different request? "
    "In the meantime, is there anything else I can help you with?"
)


SHIELD_GEMMA_MODEL_IDS = [
    "shieldgemma:2b",
    "shieldgemma:9b",
    "shieldgemma:27b",
    "shieldgemma:27b-fp16",
    "shieldgemma:27b-q2_K",
    "shieldgemma:27b-q3_K_L",
    "shieldgemma:27b-q3_K_M",
    "shieldgemma:27b-q3_K_S",
    "shieldgemma:27b-q4_0",
    "shieldgemma:27b-q4_1",
    "shieldgemma:27b-q4_K_M",
    "shieldgemma:27b-q4_K_S",
    "shieldgemma:27b-q5_0",
    "shieldgemma:27b-q5_1",
    "shieldgemma:27b-q5_K_M",
    "shieldgemma:27b-q5_K_S",
    "shieldgemma:27b-q6_K",
    "shieldgemma:27b-q8_0",
    "shieldgemma:2b-fp16",
    "shieldgemma:2b-q2_K",
    "shieldgemma:2b-q3_K_L",
    "shieldgemma:2b-q3_K_M",
    "shieldgemma:2b-q3_K_S",
    "shieldgemma:2b-q4_0",
    "shieldgemma:2b-q4_1",
    "shieldgemma:2b-q4_K_M",
    "shieldgemma:2b-q4_K_S",
    "shieldgemma:2b-q5_0",
    "shieldgemma:2b-q5_1",
    "shieldgemma:2b-q5_K_M",
    "shieldgemma:2b-q5_K_S",
    "shieldgemma:2b-q6_K",
    "shieldgemma:2b-q8_0",
    "shieldgemma:9b-fp16",
    "shieldgemma:9b-q2_K",
    "shieldgemma:9b-q3_K_L",
    "shieldgemma:9b-q3_K_M",
    "shieldgemma:9b-q3_K_S",
    "shieldgemma:9b-q4_0",
    "shieldgemma:9b-q4_1",
    "shieldgemma:9b-q4_K_M",
    "shieldgemma:9b-q4_K_S",
    "shieldgemma:9b-q5_0",
    "shieldgemma:9b-q5_1",
    "shieldgemma:9b-q5_K_M",
    "shieldgemma:9b-q5_K_S",
    "shieldgemma:9b-q6_K",
    "shieldgemma:9b-q8_0",
]


class OllamaShieldGemmaSafetyImpl(Safety, ShieldsProtocolPrivate):
    def __init__(self, config: ShieldGemmaConfig, deps):
        self.config = config

    async def register_shield(self, shield: Shield) -> None:
        if shield.provider_resource_id not in SHIELD_GEMMA_MODEL_IDS:
            raise ValueError(
                f"Unsupported ShieldGemma type: {shield.provider_resource_id}. Allowed types: {SHIELD_GEMMA_MODEL_IDS}",
            )

    async def initialize(self):
        pass

    async def run_shield(
        self,
        shield_id: str,
        messages: List[Message],
        params: Dict[str, Any] = None,
    ) -> RunShieldResponse:
        shield = await self.shield_store.get_shield(shield_id)
        if not shield:
            raise ValueError(f"Unknown shield {shield_id}")

        # TODO: instead of relabeling message, it may be better to ensure User + System ?
        messages = messages.copy()
        if len(messages) > 0 and messages[0].role != Role.user.value:
            messages[0] = UserMessage(content=messages[0].content)

        model = shield.provider_resource_id
        impl = OllamaShieldGemmaShield(model=model)
        return await impl.run(messages)


class OllamaShieldGemmaShield:
    def __init__(
        self,
        model: str,
    ):
        if model not in SHIELD_GEMMA_MODEL_IDS:
            raise ValueError(f"Unsupported model: {model}")

        self.model = model

    def validate_messages(self, messages: List[Message]) -> None:
        if len(messages) == 0:
            raise ValueError("Messages must not be empty")
        if messages[0].role != Role.user.value:
            raise ValueError("Messages must start with user")

        if len(messages) >= 2 and (
            messages[0].role == Role.user.value and messages[1].role == Role.user.value
        ):
            messages = messages[1:]

        for i in range(1, len(messages)):
            if messages[i].role == messages[i - 1].role:
                for i, m in enumerate(messages):
                    print(f"{i}: {m.role}: {m.content}")
                raise ValueError(
                    f"Messages must alternate between user and assistant. Message {i} has the same role as message {i - 1}",
                )
        return messages

    @property
    def client(self) -> AsyncClient:
        return AsyncClient(host="http://localhost:11434/")

    async def run(self, messages: List[Message]) -> RunShieldResponse:
        messages = self.validate_messages(messages)
        content = await self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,
        )
        content = content.message.content.strip()
        return self.get_shield_response(content)

    def get_shield_response(self, response: str) -> RunShieldResponse:
        response = response.strip()
        if response.lower() == "no":
            return RunShieldResponse(violation=None)

        if response.lower() == "yes":
            return RunShieldResponse(
                violation=SafetyViolation(
                    violation_level=ViolationLevel.ERROR,
                    user_message=CANNED_RESPONSE_TEXT,
                ),
            )

        raise ValueError(f"Unexpected response: {response}")
