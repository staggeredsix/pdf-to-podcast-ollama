import os
import subprocess
import sys
import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException, Response
from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from typing import List, Dict, Optional, Any
from shared.otel import OpenTelemetryInstrumentation, OpenTelemetryConfig
from shared.job import JobStatusManager
from shared.api_types import ServiceType, JobStatus
import random
import traceback
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure potential local Dia paths are available before attempting import
POTENTIAL_DIA_PATHS = [
    "/app/dia_model",
    "/app/dia_hf_repo",
    "/app/dia_github",
    "/app/dia",
]
for _p in POTENTIAL_DIA_PATHS:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Model definitions
class DialogueEntry(BaseModel):
    text: str
    speaker: str
    voice_id: Optional[str] = None

class TTSRequest(BaseModel):
    dialogue: List[DialogueEntry]
    job_id: str
    scratchpad: Optional[str] = ""
    voice_mapping: Optional[Dict[str, str]] = {}

# Initialize FastAPI app
app = FastAPI(title="Dia TTS Service")

# Set up OpenTelemetry instrumentation
telemetry = OpenTelemetryInstrumentation()
config = OpenTelemetryConfig(
    service_name="dia-tts-service",
    otlp_endpoint=os.getenv("OTLP_ENDPOINT", "http://jaeger:4317"),
    enable_redis=True,
    enable_requests=True,
)
telemetry.initialize(config, app)

# Initialize job manager with Redis publishing
job_manager = JobStatusManager(ServiceType.TTS, telemetry=telemetry)

# Speaker seeds storage
speaker_seeds: Dict[str, Dict[str, int]] = {}

def set_speaker_seeds(job_id: str, seeds: Dict[str, int]):
    """Store speaker seeds for consistent voice generation."""
    speaker_seeds[job_id] = seeds

def get_speaker_seeds(job_id: str) -> Optional[Dict[str, int]]:
    """Retrieve speaker seeds for a job."""
    return speaker_seeds.get(job_id)

def clear_speaker_seeds(job_id: str):
    """Remove speaker seeds for a job."""
    speaker_seeds.pop(job_id, None)

# Initialize Dia TTS
class DiaTTS:
    def __init__(self):
        self.model = None
        self.is_initialized = False
        self.initialize()

    def initialize(self):
        if self.is_initialized:
            return
        
        logger.info("Initializing Dia TTS engine")
        try:
            # First try to import Dia directly
            self._try_import_dia()
            
            # If that didn't work, try to install it
            if not self.is_initialized:
                self._try_install_dia()
                
            if not self.is_initialized:
                logger.error("Failed to initialize Dia TTS engine")
        except Exception as e:
            logger.error(f"Dia TTS engine initialization failed: {e}")
            logger.error(traceback.format_exc())
    
    def _try_import_dia(self):
        try:
            logger.info("Trying to import Dia model")
            from dia.model import Dia
            
            logger.info("Successfully imported Dia model, initializing")
            try:
                self.model = Dia.from_pretrained("nari-labs/Dia-1.6B")
                logger.info("Successfully initialized Dia model")
                self.is_initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize Dia model: {e}")
        except ImportError as e:
            logger.warning(f"Failed to import Dia module: {e}")
    
    def _try_install_dia(self):
        try:
            logger.info("Trying to install Dia package")
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                "git+https://github.com/nari-labs/dia.git",
                "--no-deps",
            ])
            
            # Try importing again after installation
            from dia.model import Dia
            
            logger.info("Successfully installed and imported Dia model, initializing")
            self.model = Dia.from_pretrained("nari-labs/Dia-1.6B")
            logger.info("Successfully initialized Dia model after installation")
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Failed to install Dia package: {e}")
    
    def format_input(self, dialogue: List[DialogueEntry]) -> str:
        """Format dialogue entries into a single text with speaker tags."""
        text = ""
        speaker_map = {}
        speaker_idx = 1
        for entry in dialogue:
            key = entry.speaker.lower()
            if key not in speaker_map:
                speaker_map[key] = f"[S{speaker_idx}]"
                speaker_idx += 1
            tag = speaker_map[key]
            text += f"{tag} {entry.text} "
        return text.strip()

    def generate_speech(self, text, speaker_seeds: Optional[Dict[str, int]] = None):
        """Generate speech from input text using optional speaker seeds."""
        if not self.is_initialized or self.model is None:
            raise RuntimeError("Dia TTS engine is not initialized")

        logger.info(f"Generating speech: {text[:50]}...")
        try:
            import torch
            import soundfile as sf
            import io

            if speaker_seeds:
                combined_seed = sum(speaker_seeds.values()) % (2**32 - 1)
                torch.manual_seed(combined_seed)
                random.seed(combined_seed)

            with torch.no_grad():
                if speaker_seeds and "speaker_seeds" in self.model.generate.__code__.co_varnames:
                    output = self.model.generate(text, speaker_seeds=list(speaker_seeds.values()))
                else:
                    if speaker_seeds:
                        seed = sum(speaker_seeds.values()) % (2**32 - 1)
                        torch.manual_seed(seed)
                        try:
                            import numpy as np
                            np.random.seed(seed)
                        except Exception:
                            pass
                    output = self.model.generate(text)
            
            # Get the sample rate, or use default
            sample_rate = getattr(self.model, "sample_rate", 44100)
            
            # Convert to audio file
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, output, sample_rate, format="mp3")
            audio_buffer.seek(0)
            
            return audio_buffer.read()
        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to generate speech: {e}")

def chunk_dialogue(dialogue: List[DialogueEntry], max_chars: int = 4000) -> List[List[DialogueEntry]]:
    """Split dialogue into chunks under a character limit."""
    chunks: List[List[DialogueEntry]] = []
    current: List[DialogueEntry] = []
    length = 0
    for entry in dialogue:
        approx = len(entry.text) + 10
        if length + approx > max_chars and current:
            chunks.append(current)
            current = [entry]
            length = approx
        else:
            current.append(entry)
            length += approx
    if current:
        chunks.append(current)
    return chunks

# Initialize TTS engine
dia_tts = DiaTTS()

# Initialize Dia when the application starts to avoid first-request delays
@app.on_event("startup")
async def startup_event():
    if not dia_tts.is_initialized:
        logger.info("Startup: initializing Dia TTS engine")
        dia_tts.initialize()
        logger.info(f"Dia initialization status: {dia_tts.is_initialized}")

# API endpoints
@app.post("/generate_tts", status_code=202)
async def generate_tts(request_body: TTSRequest, background_tasks: BackgroundTasks):
    """Generate TTS audio from dialogue text."""
    job_id = request_body.job_id
    logger.info(
        "Received TTS request for job %s with %d entries",
        job_id,
        len(request_body.dialogue),
    )
    
    if not dia_tts.is_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Dia TTS service is not available"
        )

    background_tasks.add_task(process_job, job_id=job_id, tts_request=request_body)
    return {"job_id": job_id}

async def process_job(job_id: str, tts_request: TTSRequest):
    """Process a TTS job."""
    try:
        job_manager.create_job(job_id)
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Processing TTS request")

        # Generate consistent speaker seeds
        seeds = get_speaker_seeds(job_id)
        if seeds is None:
            seeds = {"speaker-1": random.randint(0, 2**32 - 1), "speaker-2": random.randint(0, 2**32 - 1)}
            set_speaker_seeds(job_id, seeds)

        # Process dialogue in chunks
        chunks = chunk_dialogue(tts_request.dialogue)
        audio_chunks: List[bytes] = []

        logger.info("Using local Dia for TTS generation")
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Generating speech using local Dia")
        
        for idx, chunk in enumerate(chunks):
            job_manager.update_status(job_id, JobStatus.PROCESSING, f"Processing chunk {idx+1}/{len(chunks)}")
            formatted_text = dia_tts.format_input(chunk)
            audio_chunk = dia_tts.generate_speech(formatted_text, speaker_seeds=seeds)
            audio_chunks.append(audio_chunk)

        # Combine all audio chunks
        audio_data = b"".join(audio_chunks)
        clear_speaker_seeds(job_id)

        # Save the result
        logger.info("Audio generated successfully using Local Dia")
        job_manager.set_result(job_id, audio_data)
        job_manager.update_status(job_id, JobStatus.COMPLETED, "TTS complete")
    
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        job_manager.update_status(job_id, JobStatus.FAILED, str(e))

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get the status of a TTS job."""
    status = job_manager.get_status(job_id)
    if status is None:
        raise HTTPException(404, "Job not found")
    return status

@app.get("/output/{job_id}")
async def get_output(job_id: str):
    """Get the audio output of a completed TTS job."""
    logger.info(f"Getting output for job {job_id}")
    result = job_manager.get_result(job_id)
    logger.info(f"Result found: {result is not None}, size: {len(result) if result else 0}")
    if not result:
        raise HTTPException(404, "Result not found")
    return Response(
        content=result,
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"attachment; filename={job_id}.mp3"}
    )

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if dia_tts.is_initialized else "unhealthy",
        "local_dia": "up" if dia_tts.is_initialized else "down",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8889)
