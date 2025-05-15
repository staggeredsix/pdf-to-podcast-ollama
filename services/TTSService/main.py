from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import logging
import asyncio
import tempfile
import sys
from pathlib import Path
import redis
import traceback

# Setup more detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add potential Dia locations to Python path
potential_paths = [
    '/app/dia_model',
    '/app/dia_hf_repo', 
    '/app/dia_github',
    '/app'
]

for path in potential_paths:
    if os.path.exists(path):
        sys.path.insert(0, path)
        logger.info(f"Added {path} to Python path")

# Try multiple approaches to get a working Dia implementation
Dia = None
dia_available = False

# Method 1: Try importing from installed package
try:
    from dia.model import Dia
    logger.info("SUCCESS: Imported Dia from installed package")
    dia_available = True
except ImportError as e:
    logger.info(f"Failed to import from installed package: {e}")

# Method 2: Try importing from extracted files
if not dia_available:
    try:
        from model import Dia
        logger.info("SUCCESS: Imported Dia from extracted model.py")
        dia_available = True
    except ImportError as e:
        logger.info(f"Failed to import from model.py: {e}")

# Method 3: Create a mock implementation if all else fails
if not dia_available:
    logger.warning("All import methods failed. Creating mock implementation.")
    
    class MockDia:
        def __init__(self):
            logger.warning("Using mock Dia implementation - TTS will not work!")
        
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return cls()
        
        def generate(self, text, **kwargs):
            logger.warning("Mock implementation: generating silence")
            import numpy as np
            return np.zeros(44100, dtype=np.float32)  # 1 second of silence
        
        def save_audio(self, filename, audio):
            import soundfile as sf
            sf.write(filename, audio, 44100)
    
    Dia = MockDia
    logger.warning("Using MockDia - this is a fallback implementation!")

# Import other dependencies
import soundfile as sf

# Simple job manager without external dependencies
class SimpleJobManager:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Try to connect to Redis if available
        try:
            self.redis_client = redis.Redis(host='redis', port=6379, db=0)
            self.redis_client.ping()
            self.use_redis = True
            logger.info("Connected to Redis")
        except:
            self.redis_client = None
            self.use_redis = False
            logger.info("Redis not available, using in-memory storage")
    
    def create_job(self, job_id):
        job_data = {"status": "created", "message": "Job created"}
        if self.use_redis:
            self.redis_client.hset(f"job:{job_id}", mapping=job_data)
        else:
            self.jobs[job_id] = job_data
    
    def update_status(self, job_id, status, message=""):
        job_data = {"status": str(status), "message": message}
        if self.use_redis:
            self.redis_client.hset(f"job:{job_id}", mapping=job_data)
        else:
            self.jobs[job_id] = job_data
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

# Simple status enum
class JobStatus:
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Define service type enum for compatibility
class ServiceType:
    TTS = "tts"

app = FastAPI(title="Dia TTS Service", debug=True)

# Use simplified job manager
job_manager = SimpleJobManager()

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
        self.is_mock = Dia.__name__ == 'MockDia'
        
    def _load_model(self):
        """Load Dia model lazily"""
        if self.model is None:
            logger.info("Loading Dia model...")
            if self.is_mock:
                logger.warning("Loading mock Dia model - TTS functionality will be limited")
                self.model = Dia()
            else:
                try:
                    logger.info("Attempting to load Dia model from pretrained...")
                    self.model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")
                    logger.info("Dia model loaded successfully")
                    
                    # Check if model has Triton backend configured
                    if hasattr(self.model, 'backend') or hasattr(self.model, 'triton_model'):
                        logger.info("Model appears to have Triton backend configured")
                    else:
                        logger.warning("Model may not have Triton backend configured")
                        
                except Exception as e:
                    logger.error(f"Error loading Dia model: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    logger.warning("Falling back to mock implementation")
                    self.model = MockDia()
                    self.is_mock = True
    
    async def process_job(self, job_id: str, request: TTSRequest):
        """Process TTS job with Dia."""
        try:
            logger.info(f"Starting job {job_id}")
            job_manager.create_job(job_id)
            job_manager.update_status(job_id, JobStatus.RUNNING, "Loading model...")
            self._load_model()
            
            # Convert dialogue to Dia format
            text = self._format_dialogue_for_dia(request.dialogue)
            logger.info(f"Formatted text for Dia: {text[:200]}...")
            logger.info(f"Full text length: {len(text)} characters")
            
            if self.is_mock:
                logger.warning("Using mock TTS - generating silence")
                job_manager.update_status(job_id, JobStatus.FAILED, "TTS service using mock implementation")
                return
            
            job_manager.update_status(job_id, JobStatus.RUNNING, "Generating audio...")
            logger.info("About to call model.generate()...")
            
            # Add more verbose logging around the generation call
            try:
                # Generate audio with more detailed logging
                logger.info("Calling Dia.generate() with use_torch_compile=True")
                output = self.model.generate(text, use_torch_compile=True, verbose=True)
                logger.info(f"Generation completed. Output type: {type(output)}")
                
                if output is not None:
                    logger.info(f"Generated audio shape/length: {getattr(output, 'shape', len(output) if hasattr(output, '__len__') else 'unknown')}")
                else:
                    logger.error("Generation returned None!")
                    raise Exception("Model generate() returned None")
                    
            except Exception as gen_error:
                logger.error(f"Error during generation: {gen_error}")
                logger.error(f"Generation traceback: {traceback.format_exc()}")
                raise
            
            job_manager.update_status(job_id, JobStatus.RUNNING, "Saving audio...")
            logger.info("About to save audio to temporary file...")
            
            # Save to temporary file and read back
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                logger.info(f"Saving to temporary file: {tmp_file.name}")
                self.model.save_audio(tmp_file.name, output)
                logger.info(f"Audio saved successfully")
                
                # Read the file back
                with open(tmp_file.name, 'rb') as f:
                    audio_content = f.read()
                logger.info(f"Read back {len(audio_content)} bytes from file")
                
                # Clean up
                os.unlink(tmp_file.name)
                logger.info("Temporary file cleaned up")
            
            job_manager.set_result(job_id, audio_content)
            job_manager.update_status(job_id, JobStatus.COMPLETED, "Dia TTS generation completed")
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"[Dia TTS ERROR] Job {job_id}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
        
        logger.debug(f"Speaker mapping: {speaker_mapping}")
        return dia_text.strip()

# Initialize service
dia_service = DiaService()

@app.post("/generate_tts", status_code=202)
async def generate_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """Start TTS generation job."""
    logger.info(f"Received TTS request for job {request.job_id}")
    logger.info(f"Dialogue entries: {len(request.dialogue)}")
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
    implementation = "mock" if dia_service.is_mock else "native"
    model_info = {}
    
    # Try to get more info about the loaded model
    if dia_service.model is not None and not dia_service.is_mock:
        try:
            # Check for common Triton-related attributes
            triton_info = {}
            if hasattr(dia_service.model, 'backend'):
                triton_info['backend'] = str(dia_service.model.backend)
            if hasattr(dia_service.model, 'triton_model'):
                triton_info['triton_model'] = str(dia_service.model.triton_model)
            if hasattr(dia_service.model, 'config'):
                triton_info['config'] = str(type(dia_service.model.config))
            model_info['triton_info'] = triton_info
        except Exception as e:
            model_info['triton_check_error'] = str(e)
    
    return {
        "status": "healthy", 
        "model": f"Dia-1.6B ({implementation})",
        "model_loaded": dia_service.model is not None,
        "dia_available": dia_available,
        "implementation": implementation,
        "redis_available": job_manager.use_redis,
        "opentelemetry": "disabled",
        "model_info": model_info
    }