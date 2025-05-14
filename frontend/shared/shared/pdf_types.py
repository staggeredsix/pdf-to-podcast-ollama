# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pydantic import BaseModel, Field
from typing import Optional, Union, Literal
from datetime import datetime
from enum import Enum


class ConversionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class PDFConversionResult(BaseModel):
    filename: str
    content: str = ""
    status: ConversionStatus
    error: Optional[str] = None


class PDFMetadata(BaseModel):
    filename: str
    markdown: str = ""
    summary: str = ""
    status: ConversionStatus
    type: Union[Literal["target"], Literal["context"]]
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
