from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import logging
import asyncio
import tempfile
from pathlib import Path

from dia.model import Dia
import soundfile as sf

from shared.api_types import ServiceType, JobStatus
from shared.job import JobStatusManager
from shared.otel import OpenTelemetryInstrumentation, OpenTelemetryConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dia TTS Service", debug=True)

telemetry = OpenTelemetryInstrumentation()
telemetry.initialize(
    OpenTelemetryConfig(
        service_name="dia-tts-service",
        otlp_endpoint=os.getenv("OTLP_ENDPOINT", "http://jaeger:4317"),
        enable_redis=True,
        enable_requests=True,
    ),
    app,
)

job_manager = JobStatusManager(ServiceType.TTS, telemetry=telemetry)

class DialogueEntry(BaseModel):
    text: str
    speaker: str
    voice_id: Optional[str] = None

class TTSRequest(BaseModel):
    dialogue: List[DialogueEntry]
    job_id: str
    scratchpad: Optional[str] = ""
    voice_mapping: Optional[Dict[str, str]] = {}

class DiaService:
    def __init__(self):
        self.model = None
        
    def _load_model(self):
        """Load Dia model lazily"""
        if self.model is None:
            logger.info("Loading Dia model...")
            self.model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")
            logger.info("Dia model loaded successfully")
    
    async def process_job(self, job_id: str, request: TTSRequest):
        """Process TTS job with Dia."""
        with telemetry.tracer.start_as_current_span("dia.process_job") as span:
            try:
                job_manager.create_job(job_id)
                self._load_model()
                
                # Convert dialogue to Dia format
                text = self._format_dialogue_for_dia(request.dialogue)
                logger.info(f"Generating audio for: {text[:100]}...")
                
                # Generate audio
                output = self.model.generate(text, use_torch_compile=True, verbose=True)
                
                # Save to temporary file and read back
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                    self.model.save_audio(tmp_file.name, output)
                    
                    # Read the file back
                    with open(tmp_file.name, 'rb') as f:
                        audio_content = f.read()
                    
                    # Clean up
                    os.unlink(tmp_file.name)
                
                job_manager.set_result(job_id, audio_content)
                job_manager.update_status(job_id, JobStatus.COMPLETED, "Dia TTS generation completed")
                
            except Exception as e:
                logger.error(f"[Dia TTS ERROR] Job {job_id}: {e}")
                job_manager.update_status(job_id, JobStatus.FAILED, str(e))
    
    def _format_dialogue_for_dia(self, dialogue: List[DialogueEntry]) -> str:
        """Convert dialogue entries to Dia format"""
        dia_text = ""
        speaker_mapping = {}
        speaker_count = 1
        
        for entry in dialogue:
            # Map speakers to [S1], [S2] format
            speaker_key = entry.speaker.lower()
            if speaker_key not in speaker_mapping:
                speaker_mapping[speaker_key] = f"[S{speaker_count}]"
                speaker_count += 1
            
            speaker_tag = speaker_mapping[speaker_key]
            dia_text += f"{speaker_tag} {entry.text} "
        
        return dia_text.strip()

# Initialize service
dia_service = DiaService()

@app.post("/generate_tts", status_code=202)
async def generate_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """Start TTS generation job."""
    background_tasks.add_task(dia_service.process_job, request.job_id, request)
    return {"job_id": request.job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job status."""
    status = job_manager.get_status(job_id)
    if not status:
        raise HTTPException(404, "Job not found")
    return status

@app.get("/output/{job_id}")
async def get_output(job_id: str):
    """Get the generated audio file."""
    data = job_manager.get_result(job_id)
    if not data:
        raise HTTPException(404, "Result not found")
    return Response(
        content=data,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=output.mp3"},
    )

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "model": "Dia-1.6B",
        "model_loaded": dia_service.model is not None
    }