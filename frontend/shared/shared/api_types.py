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

from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, List
from .pdf_types import PDFMetadata
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ServiceType(str, Enum):
    PDF = "pdf"
    AGENT = "agent"
    TTS = "tts"


class StatusUpdate(BaseModel):
    job_id: str
    status: JobStatus
    message: Optional[str] = None
    service: Optional[ServiceType] = None
    timestamp: Optional[float] = None
    data: Optional[dict] = None


class StatusResponse(BaseModel):
    status: str
    result: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None


class TranscriptionParams(BaseModel):
    userId: str = Field(..., description="KAS User ID")
    name: str = Field(..., description="Name of the podcast")
    duration: int = Field(..., description="Duration in minutes")
    monologue: bool = Field(
        False, description="If True, creates a single-speaker podcast"
    )
    speaker_1_name: str = Field(
        ..., description="Name of the speaker (or first speaker if not monologue)"
    )
    speaker_2_name: Optional[str] = Field(
        None, description="Name of the second speaker (not required for monologue)"
    )
    voice_mapping: Dict[str, str] = Field(
        ...,
        description="Mapping of speaker IDs to voice IDs. For monologue, only speaker-1 is required",
        example={
            "speaker-1": "iP95p4xoKVk53GoZ742B",
            "speaker-2": "9BWtsMINqrJLrRacOk9x",
        },
    )
    guide: Optional[str] = Field(
        None, description="Optional guidance for the transcription focus and structure"
    )
    vdb_task: bool = Field(
        False,
        description="If True, creates a VDB task when running NV-Ingest allowing for retrieval abilities",
    )

    @model_validator(mode="after")
    def validate_monologue_settings(self) -> "TranscriptionParams":
        if self.monologue:
            # Check speaker_2_name is not provided
            if self.speaker_2_name is not None:
                raise ValueError(
                    "speaker_2_name should not be provided for monologue podcasts"
                )

            # Check voice_mapping only contains speaker-1
            if "speaker-2" in self.voice_mapping:
                raise ValueError(
                    "voice_mapping should only contain speaker-1 for monologue podcasts"
                )

            # Check that speaker-1 is present in voice_mapping
            if "speaker-1" not in self.voice_mapping:
                raise ValueError("voice_mapping must contain speaker-1")
        else:
            # For dialogues, ensure both speakers are present
            if not self.speaker_2_name:
                raise ValueError("speaker_2_name is required for dialogue podcasts")

            required_speakers = {"speaker-1", "speaker-2"}
            if not all(speaker in self.voice_mapping for speaker in required_speakers):
                raise ValueError(
                    "voice_mapping must contain both speaker-1 and speaker-2 for dialogue podcasts"
                )

        return self


class TranscriptionRequest(TranscriptionParams):
    pdf_metadata: List[PDFMetadata]
    job_id: str


class RAGRequest(BaseModel):
    query: str = Field(..., description="The search query to process")
    k: int = Field(..., description="Number of results to retrieve", ge=1)
    job_id: str = Field(..., description="The unique job identifier")
