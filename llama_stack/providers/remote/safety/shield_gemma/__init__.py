# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.providers.remote.safety.shield_gemma.config import (
    OllamaEndpoint,
    ShieldGemmaConfig,
)
from llama_stack.providers.remote.safety.shield_gemma.ollama_shield_gemma import (
    OllamaShieldGemmaSafetyImpl,
)


async def get_adapter_impl(config: ShieldGemmaConfig, deps) -> Any:
    assert isinstance(
        config, ShieldGemmaConfig
    ), f"Unexpected config type: {type(config)}"

    assert (
        config.inference_endpoint is not None
    ), "Must supply an inference endpoint in configuration"
    assert isinstance(
        config.inference_endpoint, OllamaEndpoint
    ), "Only Ollama-hosted ShieldGemma supported"

    impl = OllamaShieldGemmaSafetyImpl(config, deps=deps)
    await impl.initialize()
    return impl
