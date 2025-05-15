from typing import List, Dict, Any, Optional, Union
import logging
import ujson as json
from shared.otel import OpenTelemetryInstrumentation
from opentelemetry.trace.status import StatusCode
from pathlib import Path
from dataclasses import dataclass
import requests
import asyncio
import httpx
import time

# Create a custom message class for Ollama responses
class OllamaMessage:
    def __init__(self, content: str):
        self.content = content

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model.
    
    Attributes:
        name (str): Name/identifier of the model
        api_base (str): Base URL for the model's API endpoint
    """
    name: str
    api_base: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create a ModelConfig instance from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing model configuration
            
        Returns:
            ModelConfig: New ModelConfig instance
        """
        return cls(
            name=data["name"],
            api_base=data["api_base"],
        )


class LLMManager:
    """
    LLMManager that supports both NVIDIA NIM and Ollama endpoints.
    Automatically detects the endpoint type based on the API base URL.
    For Ollama, uses the more compatible /api/generate endpoint instead of /api/chat.
    No API key required for Ollama endpoints.
    """

    DEFAULT_CONFIGS = {
        "reasoning": {
            "name": "llama3.2:3b",
            "api_base": "http://ollama:11434",
        },
        "iteration": {
            "name": "llama3.2:3b",
            "api_base": "http://ollama:11434",
        },
        "json": {
            "name": "llama3.2:3b",
            "api_base": "http://ollama:11434",
        },
    }

    def __init__(
        self,
        telemetry: OpenTelemetryInstrumentation,
        api_key: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize LLMManager with telemetry.

        Args:
            telemetry (OpenTelemetryInstrumentation): Telemetry instrumentation instance
            api_key (Optional[str]): API key for NVIDIA endpoints (not needed for Ollama)
            config_path (Optional[str]): Path to custom model configurations file
        """
        try:
            self.api_key = api_key  # Only used for NVIDIA endpoints
            self.telemetry = telemetry
            self.model_configs = self._load_configurations(config_path)
            logger.info("Successfully initialized LLMManager with Ollama support")
        except Exception as e:
            logger.error(f"Failed to initialize LLMManager: {e}")
            raise

    def _load_configurations(
        self, config_path: Optional[str]
    ) -> Dict[str, ModelConfig]:
        """Load model configurations from JSON file if provided, otherwise use defaults.
        
        Args:
            config_path (Optional[str]): Path to configuration JSON file
            
        Returns:
            Dict[str, ModelConfig]: Dictionary mapping model keys to configurations
        """
        configs = self.DEFAULT_CONFIGS.copy()
        if config_path:
            try:
                config_path = Path(config_path)
                if config_path.exists():
                    with config_path.open() as f:
                        custom_configs = json.load(f)
                    configs.update(custom_configs)
                else:
                    logger.warning(
                        f"Config file {config_path} not found, using default configurations"
                    )
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
                logger.warning("Using default configurations")
        return {key: ModelConfig.from_dict(config) for key, config in configs.items()}

    def _is_ollama_endpoint(self, api_base: str) -> bool:
        """Check if the API base URL is an Ollama endpoint."""
        return "ollama" in api_base.lower() or ":11434" in api_base

    def _convert_chat_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat-style messages to a single prompt for /api/generate."""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        return "\n".join(prompt_parts) + "\nAssistant:"

    def _make_ollama_request(
        self, 
        model_config: ModelConfig, 
        messages: List[Dict[str, str]], 
        json_schema: Optional[Dict] = None,
        stream: bool = False
    ) -> requests.Response:
        """Make a request to Ollama API using /api/generate endpoint."""
        url = f"{model_config.api_base}/api/generate"
        
        # Convert chat messages to a prompt
        prompt = self._convert_chat_messages_to_prompt(messages)
        
        data = {
            "model": model_config.name,
            "prompt": prompt,
            "stream": stream
        }
        
        # Handle JSON schema for structured output
        if json_schema:
            data["format"] = "json"
            # Add schema instruction to the prompt
            schema_instruction = f"\n\nPlease respond with valid JSON following this schema: {json.dumps(json_schema)}"
            data["prompt"] += schema_instruction
        
        return requests.post(url, json=data, stream=stream)

    async def _make_ollama_request_async(
        self, 
        model_config: ModelConfig, 
        messages: List[Dict[str, str]], 
        json_schema: Optional[Dict] = None,
        stream: bool = False
    ) -> httpx.Response:
        """Make an async request to Ollama API using /api/generate endpoint."""
        url = f"{model_config.api_base}/api/generate"
        
        # Convert chat messages to a prompt
        prompt = self._convert_chat_messages_to_prompt(messages)
        
        data = {
            "model": model_config.name,
            "prompt": prompt,
            "stream": stream
        }
        
        # Handle JSON schema for structured output
        if json_schema:
            data["format"] = "json"
            # Add schema instruction to the prompt
            schema_instruction = f"\n\nPlease respond with valid JSON following this schema: {json.dumps(json_schema)}"
            data["prompt"] += schema_instruction
        
        async with httpx.AsyncClient() as client:
            return await client.post(url, json=data)

    def query_sync(
        self,
        model_key: str,
        messages: List[Dict[str, str]],
        query_name: str,
        json_schema: Optional[Dict] = None,
        retries: int = 5,
    ) -> Union[OllamaMessage, Dict[str, Any]]:
        """Send a synchronous query to the specified model."""
        with self.telemetry.tracer.start_as_current_span(
            f"agent.query.{query_name}"
        ) as span:
            span.set_attribute("model_key", model_key)
            span.set_attribute("retries", retries)
            span.set_attribute("async", False)

            try:
                model_config = self.model_configs.get(model_key)
                if not model_config:
                    raise ValueError(f"Unknown model key: {model_key}")

                if self._is_ollama_endpoint(model_config.api_base):
                    # Use Ollama API with /api/generate
                    for attempt in range(retries):
                        try:
                            response = self._make_ollama_request(
                                model_config, messages, json_schema
                            )
                            response.raise_for_status()
                            
                            result = response.json()
                            content = result.get("response", "")
                            
                            if json_schema:
                                # Parse JSON response
                                try:
                                    return json.loads(content)
                                except json.JSONDecodeError:
                                    # If JSON parsing fails, try to extract JSON from content
                                    import re
                                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                                    if json_match:
                                        return json.loads(json_match.group())
                                    raise ValueError(f"Could not parse JSON from response: {content}")
                            else:
                                return OllamaMessage(content)
                                
                        except Exception as e:
                            if attempt == retries - 1:
                                raise
                            logger.warning(f"Retry {attempt + 1}/{retries} failed: {e}")
                            time.sleep(2 ** attempt)
                else:
                    # For NVIDIA endpoints, require API key
                    if not self.api_key:
                        raise ValueError("API key required for NVIDIA endpoints")
                    # TODO: Implement NVIDIA NIM support here if needed
                    raise NotImplementedError("NVIDIA NIM support not implemented yet")

            except Exception as e:
                span.set_status(StatusCode.ERROR)
                span.record_exception(e)
                logger.error(f"Query failed: {e}")
                raise Exception(
                    f"Failed to get response after {retries} attempts"
                ) from e

    async def query_async(
        self,
        model_key: str,
        messages: List[Dict[str, str]],
        query_name: str,
        json_schema: Optional[Dict] = None,
        retries: int = 5,
    ) -> Union[OllamaMessage, Dict[str, Any]]:
        """Send an asynchronous query to the specified model."""
        with self.telemetry.tracer.start_as_current_span(
            f"agent.query.{query_name}"
        ) as span:
            span.set_attribute("model_key", model_key)
            span.set_attribute("retries", retries)
            span.set_attribute("async", True)

            try:
                model_config = self.model_configs.get(model_key)
                if not model_config:
                    raise ValueError(f"Unknown model key: {model_key}")

                if self._is_ollama_endpoint(model_config.api_base):
                    # Use Ollama API with /api/generate
                    for attempt in range(retries):
                        try:
                            response = await self._make_ollama_request_async(
                                model_config, messages, json_schema
                            )
                            response.raise_for_status()
                            
                            result = response.json()
                            content = result.get("response", "")
                            
                            if json_schema:
                                # Parse JSON response
                                try:
                                    return json.loads(content)
                                except json.JSONDecodeError:
                                    # If JSON parsing fails, try to extract JSON from content
                                    import re
                                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                                    if json_match:
                                        return json.loads(json_match.group())
                                    raise ValueError(f"Could not parse JSON from response: {content}")
                            else:
                                return OllamaMessage(content)
                                
                        except Exception as e:
                            if attempt == retries - 1:
                                raise
                            logger.warning(f"Retry {attempt + 1}/{retries} failed: {e}")
                            await asyncio.sleep(2 ** attempt)
                else:
                    # For NVIDIA endpoints, require API key
                    if not self.api_key:
                        raise ValueError("API key required for NVIDIA endpoints")
                    # TODO: Implement NVIDIA NIM support here if needed
                    raise NotImplementedError("NVIDIA NIM support not implemented yet")

            except Exception as e:
                span.set_status(StatusCode.ERROR)
                span.record_exception(e)
                logger.error(f"Async query failed: {e}")
                raise Exception(
                    f"Failed to get response after {retries} attempts"
                ) from e

    def stream_sync(
        self,
        model_key: str,
        messages: List[Dict[str, str]],
        query_name: str,
        json_schema: Optional[Dict] = None,
        retries: int = 5,
    ) -> Union[str, Dict[str, Any]]:
        """Send a synchronous streaming query to the specified model."""
        with self.telemetry.tracer.start_as_current_span(
            f"agent.stream.{query_name}"
        ) as span:
            span.set_attribute("model_key", model_key)
            span.set_attribute("retries", retries)
            span.set_attribute("async", False)

            try:
                model_config = self.model_configs.get(model_key)
                if not model_config:
                    raise ValueError(f"Unknown model key: {model_key}")

                if self._is_ollama_endpoint(model_config.api_base):
                    # Use Ollama streaming API with /api/generate
                    response = self._make_ollama_request(
                        model_config, messages, json_schema, stream=True
                    )
                    response.raise_for_status()
                    
                    full_content = ""
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            if chunk.get("response"):
                                full_content += chunk["response"]
                    
                    if json_schema and full_content:
                        return json.loads(full_content)
                    return full_content
                else:
                    # For NVIDIA endpoints, require API key
                    if not self.api_key:
                        raise ValueError("API key required for NVIDIA endpoints")
                    # TODO: Implement NVIDIA NIM support here if needed
                    raise NotImplementedError("NVIDIA NIM support not implemented yet")

            except Exception as e:
                span.set_status(StatusCode.ERROR)
                span.record_exception(e)
                logger.error(f"Streaming query failed: {e}")
                raise Exception(
                    f"Failed to get streaming response after {retries} attempts"
                ) from e

    async def stream_async(
        self,
        model_key: str,
        messages: List[Dict[str, str]],
        query_name: str,
        json_schema: Optional[Dict] = None,
        retries: int = 5,
    ) -> Union[str, Dict[str, Any]]:
        """Send an asynchronous streaming query to the specified model."""
        with self.telemetry.tracer.start_as_current_span(
            f"agent.stream.{query_name}"
        ) as span:
            span.set_attribute("model_key", model_key)
            span.set_attribute("retries", retries)
            span.set_attribute("async", True)

            try:
                model_config = self.model_configs.get(model_key)
                if not model_config:
                    raise ValueError(f"Unknown model key: {model_key}")

                if self._is_ollama_endpoint(model_config.api_base):
                    # Use Ollama streaming API with /api/generate
                    url = f"{model_config.api_base}/api/generate"
                    prompt = self._convert_chat_messages_to_prompt(messages)
                    
                    data = {
                        "model": model_config.name,
                        "prompt": prompt,
                        "stream": True
                    }
                    
                    if json_schema:
                        data["format"] = "json"
                        schema_instruction = f"\n\nPlease respond with valid JSON following this schema: {json.dumps(json_schema)}"
                        data["prompt"] += schema_instruction
                    
                    async with httpx.AsyncClient() as client:
                        async with client.stream('POST', url, json=data) as response:
                            response.raise_for_status()
                            full_content = ""
                            async for line in response.aiter_lines():
                                if line:
                                    chunk = json.loads(line)
                                    if chunk.get("response"):
                                        full_content += chunk["response"]
                            
                            if json_schema and full_content:
                                return json.loads(full_content)
                            return full_content
                else:
                    # For NVIDIA endpoints, require API key
                    if not self.api_key:
                        raise ValueError("API key required for NVIDIA endpoints")
                    # TODO: Implement NVIDIA NIM support here if needed
                    raise NotImplementedError("NVIDIA NIM support not implemented yet")

            except Exception as e:
                span.set_status(StatusCode.ERROR)
                span.record_exception(e)
                logger.error(f"Async streaming query failed: {e}")
                raise Exception(
                    f"Failed to get streaming response after {retries} attempts"
                ) from e