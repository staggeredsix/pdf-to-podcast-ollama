import os
import subprocess
import sys
import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException, Response
from pydantic import BaseModel
from typing import List, Dict, Optional
import redis
import requests
import json
import tempfile
import traceback
import time
import random

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

# Service configuration
TRITON_HOST = os.getenv("TRITON_HOST", "triton")
TRITON_PORT = os.getenv("TRITON_PORT", "8000")
TRITON_URL = os.getenv("TRITON_URL", f"http://{TRITON_HOST}:{TRITON_PORT}")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
TRITON_REQUEST_TIMEOUT = int(os.getenv("TRITON_REQUEST_TIMEOUT", "5"))

# Status constants
class JobStatus:
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

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

# Job manager for tracking TTS jobs
class SimpleJobManager:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        self.voice_seeds = {}
        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
            self.redis_client.ping()
            self.use_redis = True
            logger.info("Connected to Redis")
        except Exception as e:
            self.redis_client = None
            self.use_redis = False
            logger.info(f"Redis not available, using in-memory storage: {e}")

    def create_job(self, job_id):
        data = {"status": "created", "message": "Job created"}
        if self.use_redis:
            self.redis_client.hset(f"job:{job_id}", mapping=data)
        else:
            self.jobs[job_id] = data

    def update_status(self, job_id, status, message=""):
        data = {"status": status, "message": message}
        if self.use_redis:
            self.redis_client.hset(f"job:{job_id}", mapping=data)
        else:
            self.jobs[job_id] = data
        logger.info(f"Job {job_id}: {status} - {message}")

    def get_status(self, job_id):
        if self.use_redis:
            data = self.redis_client.hgetall(f"job:{job_id}")
            return {k.decode(): v.decode() for k, v in data.items()} if data else None
        else:
            return self.jobs.get(job_id)

    def set_result(self, job_id, result):
        if self.use_redis:
            self.redis_client.set(f"result:{job_id}", result)
        else:
            self.results[job_id] = result

    def get_voice_seeds(self, job_id) -> Dict[str, int]:
        """Return persistent voice seeds for a job, generating them if needed."""
        if job_id in self.voice_seeds:
            return self.voice_seeds[job_id]

        seeds = {"speaker-1": random.randint(0, 2**32 - 1), "speaker-2": random.randint(0, 2**32 - 1)}
        self.voice_seeds[job_id] = seeds
        return seeds

    def get_result(self, job_id):
        if self.use_redis:
            return self.redis_client.get(f"result:{job_id}")
        else:
            return self.results.get(job_id)

# Initialize job manager
job_manager = SimpleJobManager()

# Initialize Dia directly
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
            import torch
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
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/nari-labs/dia.git", 
                "--no-deps"
            ])
            
            # Try importing again after installation
            import torch
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

            with torch.no_grad():
                if speaker_seeds:
                    output = self.model.generate(text, speaker_seeds=speaker_seeds)
                else:
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
    """Split dialogue into chunks that stay under a character limit."""
    chunks: List[List[DialogueEntry]] = []
    current: List[DialogueEntry] = []
    length = 0

    for entry in dialogue:
        approx_len = len(entry.text) + 10
        if length + approx_len > max_chars and current:
            chunks.append(current)
            current = [entry]
            length = approx_len
        else:
            current.append(entry)
            length += approx_len

    if current:
        chunks.append(current)

    return chunks

# Triton TTS client
class TritonTTSClient:
    def __init__(self, url=f"{TRITON_URL}/v2/models/dia-1.6b/infer"):
        self.url = url
        self.timeout = TRITON_REQUEST_TIMEOUT
        logger.info(f"Initialized Triton TTS client with URL: {self.url} (timeout: {self.timeout}s)")
    
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
    
    async def generate_speech(self, input_text: str) -> bytes:
        """Generate speech using Triton inference server."""
        logger.info(f"Sending request to Triton server: {self.url}")
        payload = {
            "inputs": [
                {
                    "name": "INPUT_TEXT",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [input_text]
                }
            ],
            "outputs": [{"name": "OUTPUT_AUDIO"}]
        }
        
        try:
            response = requests.post(
                self.url, 
                headers={"Content-Type": "application/json"}, 
                data=json.dumps(payload),
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Triton request failed: {e}")
            raise RuntimeError(f"Failed to generate speech via Triton: {e}")

# Initialize TTS engine and clients
dia_tts = DiaTTS()
local_tts_available = dia_tts.is_initialized

# Initialize Dia when the application starts to avoid first-request delays
@app.on_event("startup")
async def startup_event():
    if not dia_tts.is_initialized:
        logger.info("Startup: initializing Dia TTS engine")
        dia_tts.initialize()
        logger.info(f"Dia initialization status: {dia_tts.is_initialized}")

# Initialize Triton client
primary_tts_client = None
try:
    logger.info("Initializing Triton TTS client")
    
    # Quick connection test
    health_url = f"{TRITON_URL}/v2/health/ready"
    logger.info(f"Testing Triton connection: {health_url}")
    
    try:
        health_response = requests.get(health_url, timeout=TRITON_REQUEST_TIMEOUT)
        if health_response.status_code == 200:
            logger.info("Triton server is reachable")
            primary_tts_client = TritonTTSClient()
            
            # Test request
            test_response = requests.post(
                primary_tts_client.url,
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "inputs": [{"name": "INPUT_TEXT", "shape": [1], "datatype": "BYTES", "data": ["[S1] Test."]}],
                    "outputs": [{"name": "OUTPUT_AUDIO"}]
                }),
                timeout=TRITON_REQUEST_TIMEOUT
            )
            
            if test_response.status_code == 200:
                logger.info("Triton TTS client initialized successfully")
            else:
                logger.warning(f"Triton test request failed: {test_response.status_code}")
                primary_tts_client = None
        else:
            logger.warning(f"Triton health check failed: {health_response.status_code}")
            primary_tts_client = None
    except Exception as e:
        logger.warning(f"Triton connection test failed: {e}")
        primary_tts_client = None
except Exception as e:
    logger.warning(f"Failed to initialize Triton client: {e}")
    primary_tts_client = None

# API endpoints
@app.post("/generate_tts", status_code=202)
async def generate_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """Generate TTS audio from dialogue text."""
    job_id = request.job_id
    logger.info(f"Received TTS request for job {job_id} with {len(request.dialogue)} entries")
    
    if not primary_tts_client and not local_tts_available:
        raise HTTPException(
            status_code=503, 
            detail="No TTS services available. Both Triton and local Dia failed to initialize."
        )
    
    background_tasks.add_task(process_job, job_id, request)
    return {"job_id": job_id}

async def process_job(job_id: str, request: TTSRequest):
    """Process a TTS job."""
    try:
        job_manager.create_job(job_id)
        job_manager.update_status(job_id, JobStatus.RUNNING, "Processing TTS request")
        
        voice_seeds = job_manager.get_voice_seeds(job_id)
        chunks = chunk_dialogue(request.dialogue)
        audio_chunks: List[bytes] = []

        if primary_tts_client:
            logger.info("Using Triton for TTS generation")
            client_type = "Triton"
            for idx, chunk in enumerate(chunks):
                formatted_text = primary_tts_client.format_input(chunk)
                try:
                    job_manager.update_status(job_id, JobStatus.RUNNING, f"Generating speech chunk {idx+1}/{len(chunks)} via Triton")
                    audio_chunk = await primary_tts_client.generate_speech(formatted_text)
                    audio_chunks.append(audio_chunk)
                except Exception as e:
                    logger.warning(f"Triton chunk {idx+1} failed: {e}")
                    if local_tts_available:
                        client_type = "Local Dia"
                        formatted_text = dia_tts.format_input(chunk)
                        audio_chunk = dia_tts.generate_speech(formatted_text, speaker_seeds=voice_seeds)
                        audio_chunks.append(audio_chunk)
                    else:
                        raise
        elif local_tts_available:
            logger.info("Using local Dia for TTS generation")
            client_type = "Local Dia"
            job_manager.update_status(job_id, JobStatus.RUNNING, "Generating speech using local Dia")
            for idx, chunk in enumerate(chunks):
                formatted_text = dia_tts.format_input(chunk)
                audio_chunk = dia_tts.generate_speech(formatted_text, speaker_seeds=voice_seeds)
                audio_chunks.append(audio_chunk)
        else:
            raise RuntimeError("No TTS services available")

        audio_data = b"".join(audio_chunks)
        
        # Save the result
        logger.info(f"Audio generated successfully using {client_type}")
        job_manager.set_result(job_id, audio_data)
        job_manager.update_status(job_id, JobStatus.COMPLETED, "TTS complete")
    
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job_manager.update_status(job_id, JobStatus.FAILED, str(e))

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get the status of a TTS job."""
    status = job_manager.get_status(job_id)
    if not status:
        raise HTTPException(404, "Job not found")
    return status

@app.get("/output/{job_id}")
async def get_output(job_id: str):
    """Get the audio output of a completed TTS job."""
    result = job_manager.get_result(job_id)
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
    triton_status = "up" if primary_tts_client else "down"
    local_status = "up" if local_tts_available else "down"
    overall_status = "healthy" if triton_status == "up" or local_status == "up" else "unhealthy"
    
    return {
        "status": overall_status,
        "triton": triton_status,
        "local_dia": local_status,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8889)
