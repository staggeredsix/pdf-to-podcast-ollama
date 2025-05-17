"""
FastAPI service for text-to-speech generation using Dia TTS model.

This service provides endpoints for generating speech from text using either:
1. A Triton Inference Server (primary method)
2. Local Dia model implementation (fallback)

It supports both synchronous and asynchronous processing, with WebSocket status updates
and Redis-based job tracking.
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException, Response
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
import os
import logging
import tempfile
import requests
import json
import numpy as np
import redis
import time
import asyncio
import base64
import torch
import importlib.util
import sys
import traceback
import io
import soundfile as sf
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service configuration
TRITON_URL = os.getenv("TRITON_URL", "http://triton:8000")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
ENABLE_LOCAL_FALLBACK = os.getenv("ENABLE_LOCAL_FALLBACK", "true").lower() == "true"
FORCE_MODEL_DOWNLOAD = os.getenv("FORCE_MODEL_DOWNLOAD", "false").lower() == "true"
HF_TOKEN = os.getenv("HF_TOKEN", None)  # Hugging Face token for private models
MODEL_PATHS = [
    "/app/dia_model",
    "/app/dia_hf_repo",
    "/app/dia_github"
]

# Configure HF token if provided
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    logger.info("Hugging Face token configured")


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
        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
            self.redis_client.ping()
            self.use_redis = True
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory storage.")
            self.redis_client = None
            self.use_redis = False
            logger.info("Redis not available, using in-memory storage")

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

    def get_result(self, job_id):
        if self.use_redis:
            return self.redis_client.get(f"result:{job_id}")
        else:
            return self.results.get(job_id)

# Initialize job manager
job_manager = SimpleJobManager()

# Base TTS client class
class BaseTTSClient:
    def __init__(self):
        pass
    
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
        """Generate speech from text - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement generate_speech")

# Triton-based TTS client
class TritonTTSClient(BaseTTSClient):
    def __init__(self, url=f"{TRITON_URL}/v2/models/dia-1.6b/infer"):
        super().__init__()
        self.url = url
        logger.info(f"Initialized Triton TTS client with URL: {self.url}")
        
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
                timeout=120
            )
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            logger.error(f"Triton request failed: {e}")
            raise RuntimeError(f"Failed to generate speech via Triton: {e}")

# Dynamically load and initialize the Dia model for local inference
class LocalDiaTTSClient(BaseTTSClient):
    def __init__(self):
        super().__init__()
        self.model = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Try to initialize the Dia model from various possible locations."""
        logger.info("Attempting to initialize local Dia model...")
        
        # Find and add the Dia module to the path
        dia_module_path = None
        for path in MODEL_PATHS:
            module_path = Path(path)
            if (module_path / "model.py").exists():
                dia_module_path = module_path
                break
        
        if dia_module_path is None:
            logger.error("Could not find Dia model implementation")
            return
        
        logger.info(f"Found Dia model at {dia_module_path}")
        
        # Set up model paths for download
        model_path = Path("/app/models/dia")
        os.makedirs(model_path, exist_ok=True)
        
        # Set environment variables for Hugging Face model download
        os.environ["TRANSFORMERS_CACHE"] = str(model_path / "transformers")
        os.environ["HF_HOME"] = str(model_path / "hf")
        
        try:
            # Add Dia module path to sys.path
            sys.path.insert(0, str(dia_module_path.parent))
            
            # Try to import Dia modules
            import dia
            from dia import DiaModel
            
            # Initialize model with download if needed
            logger.info("Initializing Dia model (will download if not found)...")
            try:
                # First attempt with local model only
                self.model = DiaModel.from_pretrained(local_files_only=True)
                logger.info("Dia model loaded from local files!")
            except Exception as e_local:
                logger.warning(f"Local model not found: {e_local}. Attempting to download...")
                # If local fails, try to download
                try:
                    self.model = DiaModel.from_pretrained(
                        repo_id="nari-labs/Dia-1.6B",
                        force_download=True,
                        resume_download=True
                    )
                    logger.info("Dia model downloaded and initialized successfully!")
                except Exception as e_download:
                    logger.error(f"Download attempt failed: {e_download}")
                    raise
                
        except ImportError as e:
            logger.error(f"Failed to import Dia module: {e}")
            # Try alternative import method
            try:
                spec = importlib.util.spec_from_file_location(
                    "dia_model", 
                    str(dia_module_path / "model.py")
                )
                dia_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(dia_module)
                
                # Try to load with download capability
                try:
                    self.model = dia_module.DiaModel.from_pretrained(local_files_only=True)
                    logger.info("Dia model loaded from local files via alternative method!")
                except Exception as e_local:
                    logger.warning(f"Local model not found: {e_local}. Attempting to download...")
                    self.model = dia_module.DiaModel.from_pretrained(
                        repo_id="nari-labs/Dia-1.6B",
                        force_download=True,
                        resume_download=True
                    )
                    logger.info("Dia model downloaded and initialized successfully via alternative method!")
            except Exception as e2:
                logger.error(f"Alternative import and download failed: {e2}")
                logger.error(traceback.format_exc())
                self.model = None
        except Exception as e:
            logger.error(f"Failed to initialize Dia model: {e}")
            logger.error(traceback.format_exc())
            self.model = None
    
    async def generate_speech(self, input_text: str) -> bytes:
        """Generate speech using local Dia model."""
        if self.model is None:
            raise RuntimeError("Local Dia model is not initialized")
        
        logger.info(f"Generating speech locally for text: {input_text[:50]}...")
        try:
            # Use model to generate audio
            with torch.no_grad():
                audio_array = self.model.generate(input_text)
            
            # Convert numpy array to audio file in memory
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_array, self.model.sample_rate, format='mp3')
            audio_buffer.seek(0)
            
            return audio_buffer.read()
        except Exception as e:
            logger.error(f"Local speech generation failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to generate speech locally: {e}")

# Initialize TTS clients with fallback mechanism
try:
    logger.info("Initializing Triton TTS client as primary")
    primary_tts_client = TritonTTSClient()
    
    # Try a test request to make sure Triton is working
    dummy_request = {
        "inputs": [{"name": "INPUT_TEXT", "shape": [1], "datatype": "BYTES", "data": ["[S1] Test."]}],
        "outputs": [{"name": "OUTPUT_AUDIO"}]
    }
    test_response = requests.post(
        primary_tts_client.url, 
        headers={"Content-Type": "application/json"},
        data=json.dumps(dummy_request),
        timeout=5
    )
    
    if test_response.status_code != 200:
        logger.warning(f"Triton test request failed with status {test_response.status_code}")
        raise Exception("Triton test request failed")
        
    logger.info("Triton TTS client initialized successfully")
except Exception as e:
    logger.warning(f"Triton TTS client initialization failed: {e}")
    primary_tts_client = None

# Initialize local fallback if enabled
fallback_tts_client = None
if ENABLE_LOCAL_FALLBACK:
    try:
        logger.info("Initializing local Dia TTS client as fallback")
        fallback_tts_client = LocalDiaTTSClient()
        if fallback_tts_client.model is None and FORCE_MODEL_DOWNLOAD:
            logger.info("Attempting to download Dia model as it wasn't found locally")
            # This will be implemented by the model initialization code
            fallback_tts_client = LocalDiaTTSClient()  # Retry after forced download
        logger.info("Local Dia TTS client initialization completed")
    except Exception as e:
        logger.error(f"Failed to initialize local Dia TTS client: {e}")
        logger.error(traceback.format_exc())
        fallback_tts_client = None

# API endpoints
@app.post("/generate_tts", status_code=202)
async def generate_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Generate TTS audio from dialogue text.
    
    Args:
        request: Contains dialogue, job_id, and voice mapping info
        background_tasks: FastAPI background tasks handler
        
    Returns:
        Dict with job_id that can be used to track progress
    """
    job_id = request.job_id
    logger.info(f"Received TTS request for job {job_id} with {len(request.dialogue)} entries")
    
    if not primary_tts_client and not fallback_tts_client:
        raise HTTPException(
            status_code=503, 
            detail="No TTS clients available. Both Triton and local fallback failed to initialize."
        )
    
    background_tasks.add_task(process_job, job_id, request)
    return {"job_id": job_id}

async def process_job(job_id: str, request: TTSRequest):
    """
    Process a TTS job.
    
    Args:
        job_id: Unique identifier for the job
        request: Full TTS request with dialogue entries
    """
    try:
        job_manager.create_job(job_id)
        job_manager.update_status(job_id, JobStatus.RUNNING, "Processing TTS request")
        
        # Format the input text
        tts_client = primary_tts_client or fallback_tts_client
        formatted_text = tts_client.format_input(request.dialogue)
        
        # Generate audio
        try:
            logger.info(f"Generating audio for job {job_id}")
            job_manager.update_status(
                job_id, 
                JobStatus.RUNNING, 
                f"Generating speech using {'Triton' if tts_client == primary_tts_client else 'local Dia'}"
            )
            
            audio_data = await tts_client.generate_speech(formatted_text)
            
            # If primary client fails, try fallback
            if not audio_data and fallback_tts_client and tts_client == primary_tts_client:
                logger.warning(f"Primary TTS client failed, trying fallback for job {job_id}")
                job_manager.update_status(
                    job_id,
                    JobStatus.RUNNING,
                    "Primary TTS failed, trying fallback"
                )
                tts_client = fallback_tts_client
                audio_data = await tts_client.generate_speech(formatted_text)
        
        except Exception as e:
            # If primary client fails, try fallback
            if fallback_tts_client and tts_client == primary_tts_client:
                logger.warning(f"Primary TTS client failed: {e}, trying fallback for job {job_id}")
                job_manager.update_status(
                    job_id,
                    JobStatus.RUNNING,
                    "Primary TTS failed, trying fallback"
                )
                tts_client = fallback_tts_client
                audio_data = await tts_client.generate_speech(formatted_text)
            else:
                raise
        
        # Save result
        logger.info(f"Audio generated successfully for job {job_id}, size: {len(audio_data)} bytes")
        job_manager.set_result(job_id, audio_data)
        job_manager.update_status(job_id, JobStatus.COMPLETED, "TTS complete")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        logger.error(traceback.format_exc())
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
        headers={"Content-Disposition": "attachment; filename=output.mp3"}
    )

@app.get("/health")
async def health():
    """Health check endpoint."""
    status = "healthy"
    clients = {}
    
    # Check Triton client
    if primary_tts_client:
        try:
            test_response = requests.get(
                TRITON_URL + "/v2/health/ready",
                timeout=2
            )
            clients["triton"] = "up" if test_response.status_code == 200 else "down"
        except:
            clients["triton"] = "down"
    else:
        clients["triton"] = "not_initialized"
    
    # Check local fallback
    if fallback_tts_client:
        clients["local_dia"] = "up" if fallback_tts_client.model is not None else "down"
    else:
        clients["local_dia"] = "not_initialized"
    
    # Check Redis
    try:
        redis_status = "up" if job_manager.use_redis and job_manager.redis_client.ping() else "down"
    except:
        redis_status = "down"
    
    # Overall status is healthy only if at least one TTS client is up
    if clients.get("triton") != "up" and clients.get("local_dia") != "up":
        status = "unhealthy"
    
    return {
        "status": status,
        "triton_url": primary_tts_client.url if primary_tts_client else None,
        "clients": clients,
        "redis": redis_status,
        "timestamp": time.time()
    }