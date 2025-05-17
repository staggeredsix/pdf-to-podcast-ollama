"""
Simple wrapper for Dia TTS model.
This module provides a straightforward interface to the Dia TTS model.
"""

import os
import sys
import logging
import torch
import soundfile as sf
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiaWrapper:
    """
    Wrapper for Dia TTS model that handles initialization and audio generation.
    """
    def __init__(self):
        self.model = None
        self.sample_rate = 44100  # Default sample rate
        try:
            self._initialize_model()
        except Exception as e:
            logger.error(f"Failed to initialize Dia model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    def _initialize_model(self):
        """Initialize the Dia model"""
        logger.info("Initializing Dia model")
        
        # Add potential model paths to sys.path
        potential_paths = [
            "/app/dia_model",
            "/app/dia_hf_repo",
            "/app/dia_github",
            "/app/dia"
        ]
        for path in potential_paths:
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)

        # Try importing Dia directly
        try:
            from dia.model import Dia
            logger.info("Successfully imported Dia")
            self.model = Dia.from_pretrained("nari-labs/Dia-1.6B")
            logger.info("Successfully initialized Dia model")
            return
        except Exception as e:
            logger.warning(f"Failed to import Dia directly: {e}")

        # Try importing via pip package
        try:
            logger.info("Attempting to install Dia package")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/nari-labs/dia.git"])
            
            from dia.model import Dia
            logger.info("Successfully imported Dia after installation")
            self.model = Dia.from_pretrained("nari-labs/Dia-1.6B")
            logger.info("Successfully initialized Dia model")
            return
        except Exception as e:
            logger.warning(f"Failed to install and import Dia: {e}")
            raise

    def generate_speech(self, text):
        """
        Generate speech from text
        
        Args:
            text (str): Input text with speaker tags [S1] and [S2]
            
        Returns:
            bytes: MP3 audio data
        """
        if self.model is None:
            raise RuntimeError("Dia model not initialized")
        
        logger.info(f"Generating speech for: {text[:50]}...")
        
        try:
            # Generate audio
            with torch.no_grad():
                output = self.model.generate(text)
            
            # Convert to MP3
            buffer = BytesIO()
            sf.write(buffer, output, self.sample_rate, format='mp3')
            buffer.seek(0)
            audio_data = buffer.read()
            
            logger.info(f"Successfully generated audio: {len(audio_data)} bytes")
            return audio_data
        except Exception as e:
            logger.error(f"Failed to generate speech: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

# Create a singleton instance
tts_engine = DiaWrapper()
