from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import logging
import tempfile
import requests
import json
import numpy as np
import redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleJobManager:
    def __init__(self):
        self.jobs = {}
        self.results = {}
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

class JobStatus:
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class DialogueEntry(BaseModel):
    text: str
    speaker: str
    voice_id: Optional[str] = None

class TTSRequest(BaseModel):
    dialogue: List[DialogueEntry]
    job_id: str
    scratchpad: Optional[str] = ""
    voice_mapping: Optional[Dict[str, str]] = {}

app = FastAPI(title="Dia TTS Triton Client")
job_manager = SimpleJobManager()

class DiaTritonClient:
    def __init__(self, url="http://triton:8000/v2/models/dia-1.6b/infer"):
        self.url = url

    def format_input(self, dialogue: List[DialogueEntry]) -> str:
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

    def infer(self, input_text: str):
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
        logger.info(f"Sending inference to Triton for: {input_text[:50]}...")
        response = requests.post(self.url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        response.raise_for_status()
        return response.content

dia_triton_client = DiaTritonClient()

@app.post("/generate_tts", status_code=202)
async def generate_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    job_id = request.job_id
    logger.info(f"Received TTS request for job {job_id} with {len(request.dialogue)} entries")
    background_tasks.add_task(process_job, job_id, request)
    return {"job_id": job_id}

async def process_job(job_id: str, request: TTSRequest):
    try:
        job_manager.create_job(job_id)
        job_manager.update_status(job_id, JobStatus.RUNNING, "Sending to Triton")

        formatted_text = dia_triton_client.format_input(request.dialogue)
        audio_data = dia_triton_client.infer(formatted_text)

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_file.write(audio_data)
        tmp_file.close()

        with open(tmp_file.name, "rb") as f:
            result_data = f.read()

        os.unlink(tmp_file.name)

        job_manager.set_result(job_id, result_data)
        job_manager.update_status(job_id, JobStatus.COMPLETED, "TTS complete")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        job_manager.update_status(job_id, JobStatus.FAILED, str(e))

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    status = job_manager.get_status(job_id)
    if not status:
        raise HTTPException(404, "Job not found")
    return status

@app.get("/output/{job_id}")
async def get_output(job_id: str):
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
    return {
        "status": "healthy",
        "triton_url": dia_triton_client.url
    }
