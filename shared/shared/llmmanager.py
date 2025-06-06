from typing import List, Dict, Any, Optional, Union
import logging
try:

    import ujson as json
except Exception:  # pragma: no cover - ujson might not be installed
    import json

from shared.otel import OpenTelemetryInstrumentation
from opentelemetry.trace.status import StatusCode
from pathlib import Path
from dataclasses import dataclass
import requests
import asyncio
import httpx
import time
import random
import textwrap

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
            "name": "llama3.1:8b-instruct-fp16-optimized",
            "api_base": "http://ollama:11434",
        },
        "iteration": {
            "name": "llama3.1:8b-instruct-fp16-optimized",
            "api_base": "http://ollama:11434",
        },
        "json": {
            "name": "llama3.1:8b-instruct-fp16-optimized",
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
        
        full_prompt = "\n".join(prompt_parts) + "\nAssistant:"
        
        # Truncate prompt if too long (approximate token counting - 1 token â 4 characters)
        max_chars = 20000  # ~5000 tokens, leaving room for response
        if len(full_prompt) > max_chars:

            removed_chars = len(full_prompt) - max_chars
            logger.warning(
                f"Prompt too long ({len(full_prompt)} chars), truncating to {max_chars} chars (removing {removed_chars} chars)"
            )
            # Try to keep the end of the prompt which is usually most important

            full_prompt = "..." + full_prompt[-max_chars:]
        
        return full_prompt

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
            "stream": stream,
            "options": {
                "num_ctx": 8192,  # Increased context window
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 2048 if json_schema else 1024,  # More tokens for JSON responses
                "stop": ["Human:", "User:"],  # Stop sequences to prevent runaway generation
            }
        }
        
        # Handle JSON schema for structured output
        if json_schema:
            data["format"] = "json"
            # Create a more focused schema instruction with concrete examples
            schema_fields = []
            if 'properties' in json_schema:
                for field, props in json_schema['properties'].items():
                    field_type = props.get('type', 'string')
                    if field_type == 'array':
                        item_type = props.get('items', {}).get('type', 'object')
                        if item_type == 'object':
                            schema_fields.append(f'"{field}": [{{"example_property": "example_value"}}]')
                        else:
                            schema_fields.append(f'"{field}": ["example_item1", "example_item2"]')
                    elif field_type == 'string':
                        # Add enum examples if available
                        if 'enum' in props:
                            example_val = props['enum'][0] if props['enum'] else "example_string"
                            schema_fields.append(f'"{field}": "{example_val}"')
                        else:
                            schema_fields.append(f'"{field}": "example_string"')
                    elif field_type == 'number' or field_type == 'integer':
                        schema_fields.append(f'"{field}": 42')
                    elif field_type == 'boolean':
                        schema_fields.append(f'"{field}": true')
                    else:
                        schema_fields.append(f'"{field}": "example_value"')
            
            schema_example = "{\n  " + ",\n  ".join(schema_fields) + "\n}"
            
            # Add a clear instruction for JSON output
            schema_instruction_template = textwrap.dedent("""

                CRITICAL: You must return ACTUAL DATA, not a schema definition!

                Do NOT return anything with "$defs", "description", "properties", "type", "title", etc.
                Do NOT return the schema structure itself.

                Instead, return REAL DATA that matches this structure:
                {schema_example}

                IMPORTANT JSON FORMATTING RULES:
                1. Use double quotes for all strings
                2. Escape any quotes within text content (use \\" for quotes in text)
                3. No trailing commas
                4. Ensure all braces and brackets are properly closed
                5: Never return a : in the text of a speaker
                   Example what not to do when returning text of a speaker: "The title of the book is \\"Cheese : Squishy"
                   Example of what to do when returning text of a speaker: "The title of the book is \\"Cheese Squishy"
                6. EVERY dialogue entry MUST have both "text" and "speaker" fields
                7. Speaker values MUST be exactly "speaker-1" or "speaker-2"

                Example of what NOT to do (schema): {{"$defs": {{"DialogueEntry": {{"properties": ...}}}}}}
                Example of what TO do (data): {{"dialogue": [{{"text": "Hello, it\\"s nice to meet you!", "speaker": "speaker-1"}}]}}
                Before returning actual JSON ensure that the JSON is valid per the instructions you have been given.
                Return only the actual JSON data now:
                """)
            schema_instruction = schema_instruction_template.format(schema_example=schema_example)
            data["prompt"] += schema_instruction
        
        # Increase timeout to 120 seconds for complex requests
        return requests.post(url, json=data, stream=stream, timeout=120)

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
            "stream": stream,
            "options": {
                "num_ctx": 8192,  # Increased context window
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 2048 if json_schema else 1024,  # More tokens for JSON responses
                "stop": ["Human:", "User:"],  # Stop sequences to prevent runaway generation
            }
        }
        
        # Handle JSON schema for structured output
        if json_schema:
            data["format"] = "json"
            # Create a more focused schema instruction with concrete examples
            schema_fields = []
            if 'properties' in json_schema:
                for field, props in json_schema['properties'].items():
                    field_type = props.get('type', 'string')
                    if field_type == 'array':
                        item_type = props.get('items', {}).get('type', 'object')
                        if item_type == 'object':
                            schema_fields.append(f'"{field}": [{{"example_property": "example_value"}}]')
                        else:
                            schema_fields.append(f'"{field}": ["example_item1", "example_item2"]')
                    elif field_type == 'string':
                        # Add enum examples if available
                        if 'enum' in props:
                            example_val = props['enum'][0] if props['enum'] else "example_string"
                            schema_fields.append(f'"{field}": "{example_val}"')
                        else:
                            schema_fields.append(f'"{field}": "example_string"')
                    elif field_type == 'number' or field_type == 'integer':
                        schema_fields.append(f'"{field}": 42')
                    elif field_type == 'boolean':
                        schema_fields.append(f'"{field}": true')
                    else:
                        schema_fields.append(f'"{field}": "example_value"')
            
            schema_example = "{\n  " + ",\n  ".join(schema_fields) + "\n}"
            
            # Add a clear instruction for JSON output
            schema_instruction_template = textwrap.dedent("""

                CRITICAL: You must return ACTUAL DATA, not a schema definition!

                Do NOT return anything with "$defs", "description", "properties", "type", "title", etc.
                Do NOT return the schema structure itself.

                Instead, return REAL DATA that matches this structure:
                {schema_example}

                IMPORTANT JSON FORMATTING RULES:
                1. Use double quotes for all strings
                2. Escape any quotes within text content (use \\" for quotes in text)
                3. No trailing commas
                4. Ensure all braces and brackets are properly closed
                5. EVERY dialogue entry MUST have both "text" and "speaker" fields
                6. Speaker values MUST be exactly "speaker-1" or "speaker-2"

                Example of what NOT to do (schema): {{"$defs": {{"DialogueEntry": {{"properties": ...}}}}}}
                Example of what TO do (data): {{"dialogue": [{{"text": "Hello, it\\"s nice to meet you!", "speaker": "speaker-1"}}]}}

                Return only the actual JSON data now:
                """)
            schema_instruction = schema_instruction_template.format(schema_example=schema_example)
            data["prompt"] += schema_instruction
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            return await client.post(url, json=data)

    def _validate_conversation_json(self, data: dict) -> dict:
        """Validate and fix conversation JSON structure."""
        if not isinstance(data, dict):
            return data
        
        if "dialogue" in data and isinstance(data["dialogue"], list):
            fixed_dialogue = []
            for i, entry in enumerate(data["dialogue"]):
                if isinstance(entry, dict) and "text" in entry:
                    # Ensure speaker field exists
                    if "speaker" not in entry:
                        entry["speaker"] = "speaker-1" if i % 2 == 0 else "speaker-2"
                        logger.warning(f"Added missing speaker to dialogue entry {i}")
                    
                    # Ensure speaker value is valid
                    if entry["speaker"] not in ["speaker-1", "speaker-2"]:
                        entry["speaker"] = "speaker-1" if i % 2 == 0 else "speaker-2"
                        logger.warning(f"Fixed invalid speaker value in dialogue entry {i}")
                    
                    fixed_dialogue.append(entry)
            
            data["dialogue"] = fixed_dialogue
        
        # Ensure scratchpad exists
        if "scratchpad" not in data:
            data["scratchpad"] = ""
        
        return data

    def _clean_and_parse_json(self, content: str) -> dict:
        """Clean and parse JSON response from Ollama."""
        import ujson as json
        import re
        
        # Remove any leading/trailing whitespace
        content = content.strip()
        
        # Remove code block markers if present
        content = content.replace('```json', '').replace('```', '')
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group()
        
        # Try to parse the JSON first without cleaning
        try:
            parsed = json.loads(content)
            # Validate that all dialogue entries have required fields
            if "dialogue" in parsed and isinstance(parsed["dialogue"], list):
                for i, entry in enumerate(parsed["dialogue"]):
                    if isinstance(entry, dict):
                        if "text" in entry and "speaker" not in entry:
                            # Try to infer speaker based on position (alternating pattern)
                            entry["speaker"] = "speaker-1" if i % 2 == 0 else "speaker-2"
                            logger.warning(f"Added missing speaker field to dialogue entry {i}")
            return parsed
        except json.JSONDecodeError:
            logger.warning("Initial JSON parse failed, attempting to clean...")
        
        # Basic cleaning - be more conservative to avoid corrupting speaker fields
        content = re.sub(r',\s*}', '}', content)  # Remove trailing commas in objects
        content = re.sub(r',\s*]', ']', content)  # Remove trailing commas in arrays
        
        # Try parsing again after basic cleaning
        try:
            parsed = json.loads(content)
            # Validate and fix dialogue entries
            if "dialogue" in parsed and isinstance(parsed["dialogue"], list):
                for i, entry in enumerate(parsed["dialogue"]):
                    if isinstance(entry, dict):
                        if "text" in entry and "speaker" not in entry:
                            entry["speaker"] = "speaker-1" if i % 2 == 0 else "speaker-2"
                            logger.warning(f"Added missing speaker field to dialogue entry {i}")
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed after basic cleaning: {e}")
            logger.error(f"Content: {content[:500]}...")
            
            # Last resort: try to extract dialogue entries manually
            try:
                dialogue_entries = []
                
                # Look for dialogue patterns in the text
                patterns = [
                    r'"text":\s*"([^"]+)",\s*"speaker":\s*"([^"]+)"',
                    r'"speaker":\s*"([^"]+)",\s*"text":\s*"([^"]+)"',
                    r'\{\s*"text":\s*"([^"]+)"\s*\}',  # Missing speaker
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        for i, match in enumerate(matches):
                            if len(match) == 2:
                                if pattern.startswith('"text"'):
                                    text, speaker = match
                                else:
                                    speaker, text = match
                            else:
                                text = match[0]
                                speaker = "speaker-1" if i % 2 == 0 else "speaker-2"
                            
                            dialogue_entries.append({
                                "text": text,
                                "speaker": speaker
                            })
                        break
                
                if dialogue_entries:
                    return {
                        "scratchpad": "",
                        "dialogue": dialogue_entries
                    }
            except Exception:
                pass
                
            raise ValueError(f"Could not parse JSON: {e}")

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
                            logger.info(f"Attempting request {attempt + 1}/{retries} for {query_name}")
                            response = self._make_ollama_request(
                                model_config, messages, json_schema
                            )
                            response.raise_for_status()
                            
                            result = response.json()
                            content = result.get("response", "")
                            
                            # Check for error in response
                            if "error" in result:
                                raise ValueError(f"Ollama error: {result['error']}")
                            
                            if json_schema:
                                # Parse JSON response with cleaning
                                try:
                                    parsed_json = self._clean_and_parse_json(content)
                                    
                                    # Check if this is a schema definition instead of actual data
                                    if '$defs' in parsed_json or 'type' in parsed_json and 'properties' in parsed_json:
                                        # Model returned the schema, not data - this is an error
                                        logger.error(f"Model returned schema instead of data: {content[:200]}...")
                                        raise ValueError("Model returned schema definition instead of actual data")
                                    
                                    # Validate conversation structure if this looks like a conversation
                                    if 'dialogue' in parsed_json:
                                        parsed_json = self._validate_conversation_json(parsed_json)
                                    
                                    return parsed_json
                                except (json.JSONDecodeError, ValueError) as e:
                                    logger.error(f"Failed to parse JSON response: {e}")
                                    logger.error(f"Raw content: {content[:500]}...")
                                    # Try one more time with a more aggressive regex
                                    import re
                                    json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
                                    for match in json_matches:
                                        try:
                                            parsed_json = self._clean_and_parse_json(match)
                                            if '$defs' not in parsed_json and not ('type' in parsed_json and 'properties' in parsed_json):
                                                if 'dialogue' in parsed_json:
                                                    parsed_json = self._validate_conversation_json(parsed_json)
                                                return parsed_json

                                        except (json.JSONDecodeError, ValueError):

                                            continue
                                    raise ValueError(f"Could not parse any valid JSON from response: {str(e)}")
                            else:
                                return OllamaMessage(content)
                                
                        except requests.exceptions.Timeout:
                            if attempt == retries - 1:
                                raise Exception(f"Request timed out after {retries} attempts")
                            wait_time = (2 ** attempt) + random.uniform(0, 1)
                            logger.warning(f"Timeout on attempt {attempt + 1}/{retries}, waiting {wait_time:.1f}s before retry...")
                            time.sleep(wait_time)
                        except requests.exceptions.HTTPError as e:
                            if e.response.status_code == 500:
                                if attempt == retries - 1:
                                    raise Exception(f"Ollama server error after {retries} attempts: {e.response.text}")
                                wait_time = (2 ** attempt) + random.uniform(0, 1)
                                logger.warning(f"Server error on attempt {attempt + 1}/{retries}, waiting {wait_time:.1f}s before retry...")
                                time.sleep(wait_time)
                            else:
                                raise
                        except Exception as e:
                            if attempt == retries - 1:
                                raise Exception(f"Failed after {retries} attempts: {str(e)}")
                            wait_time = (2 ** attempt) + random.uniform(0, 1)
                            logger.warning(f"Retry {attempt + 1}/{retries} failed with error: {e}, waiting {wait_time:.1f}s...")
                            time.sleep(wait_time)
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

    def _extract_json_block(self, text: str) -> Optional[str]:
        """Return the first balanced JSON object found in text."""
        start = text.find('{')
        while start != -1:
            depth = 0
            for idx in range(start, len(text)):
                if text[idx] == '{':
                    depth += 1
                elif text[idx] == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:idx + 1]
            start = text.find('{', start + 1)
        return None

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
                                    parsed_json = self._clean_and_parse_json(content)
                                    # Validate conversation structure if this looks like a conversation
                                    if 'dialogue' in parsed_json:
                                        parsed_json = self._validate_conversation_json(parsed_json)
                                    return parsed_json
                                except json.JSONDecodeError:
                                    # If JSON parsing fails, try to extract JSON from content
                                    import re
                                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                                    if json_match:
                                        parsed_json = json.loads(json_match.group())
                                        if 'dialogue' in parsed_json:
                                            parsed_json = self._validate_conversation_json(parsed_json)
                                        return parsed_json
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
                        parsed_json = self._clean_and_parse_json(full_content)
                        if 'dialogue' in parsed_json:
                            parsed_json = self._validate_conversation_json(parsed_json)
                        return parsed_json
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
                        "stream": True,
                        "options": {
                            "num_ctx": 8192,  # Increased context window
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "top_k": 40,
                            "num_predict": 2048 if json_schema else 1024,  # More tokens for JSON responses
                            "stop": ["Human:", "User:"],  # Stop sequences to prevent runaway generation
                        }
                    }
                    
                    if json_schema:
                        data["format"] = "json"
                        # Create a more focused schema instruction with concrete examples
                        schema_fields = []
                        if 'properties' in json_schema:
                            for field, props in json_schema['properties'].items():
                                field_type = props.get('type', 'string')
                                if field_type == 'array':
                                    item_type = props.get('items', {}).get('type', 'object')
                                    if item_type == 'object':
                                        schema_fields.append(f'"{field}": [{{"example_property": "example_value"}}]')
                                    else:
                                        schema_fields.append(f'"{field}": ["example_item1", "example_item2"]')
                                elif field_type == 'string':
                                    # Add enum examples if available
                                    if 'enum' in props:
                                        example_val = props['enum'][0] if props['enum'] else "example_string"
                                        schema_fields.append(f'"{field}": "{example_val}"')
                                    else:
                                        schema_fields.append(f'"{field}": "example_string"')
                                elif field_type == 'number' or field_type == 'integer':
                                    schema_fields.append(f'"{field}": 42')
                                elif field_type == 'boolean':
                                    schema_fields.append(f'"{field}": true')
                                else:
                                    schema_fields.append(f'"{field}": "example_value"')
                        
                        schema_example = "{\n  " + ",\n  ".join(schema_fields) + "\n}"
                        
                        # Add a clear instruction for JSON output
                        schema_instruction_template = textwrap.dedent("""

                            CRITICAL: You must return ACTUAL DATA, not a schema definition!

                            Do NOT return anything with "$defs", "description", "properties", "type", "title", etc.
                            Do NOT return the schema structure itself.

                            Instead, return REAL DATA that matches this structure:
                            {schema_example}

                            IMPORTANT JSON FORMATTING RULES:
                            1. Use double quotes for all strings
                            2. Escape any quotes within text content (use \\" for quotes in text)
                            3. No trailing commas
                            4. Ensure all braces and brackets are properly closed
                            5. EVERY dialogue entry MUST have both "text" and "speaker" fields
                            6. Speaker values MUST be exactly "speaker-1" or "speaker-2"

                            Example of what NOT to do (schema): {{"$defs": {{"DialogueEntry": {{"properties": ...}}}}}}
                            Example of what TO do (data): {{"dialogue": [{{"text": "Hello, it\\"s nice to meet you!", "speaker": "speaker-1"}}]}}

                            Return only the actual JSON data now:
                            """)
                        schema_instruction = schema_instruction_template.format(schema_example=schema_example)
                        data["prompt"] += schema_instruction
                    
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        async with client.stream('POST', url, json=data) as response:
                            response.raise_for_status()
                            full_content = ""
                            async for line in response.aiter_lines():
                                if line:
                                    chunk = json.loads(line)
                                    if chunk.get("response"):
                                        full_content += chunk["response"]
                            
                            if json_schema and full_content:
                                try:
                                    parsed_json = self._clean_and_parse_json(full_content)
                                    # Check if this is a schema definition instead of actual data
                                    if '$defs' in parsed_json or 'type' in parsed_json and 'properties' in parsed_json:
                                        logger.error(f"Model returned schema instead of data in streaming: {full_content[:200]}...")
                                        raise ValueError("Model returned schema definition instead of actual data")
                                    
                                    # Validate conversation structure if this looks like a conversation
                                    if 'dialogue' in parsed_json:
                                        parsed_json = self._validate_conversation_json(parsed_json)
                                        
                                    return parsed_json
                                except (json.JSONDecodeError, ValueError) as e:
                                    logger.error(f"Failed to parse JSON from streaming response: {e}")
                                    logger.error(f"Raw content: {full_content[:500]}...")
                                    raise ValueError(f"Could not parse JSON from streaming response: {str(e)}")
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