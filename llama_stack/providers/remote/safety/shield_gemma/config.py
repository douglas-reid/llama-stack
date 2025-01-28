# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Union

from pydantic import BaseModel, Field


class OllamaEndpoint(BaseModel):
    host_url: str = Field(default="http://localhost:11434/")


class ShieldGemmaConfig(BaseModel):
    inference_endpoint: Union[OllamaEndpoint, None] = Field(default=OllamaEndpoint())
