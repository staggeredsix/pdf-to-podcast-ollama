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
        
        # Truncate prompt if too long (approximate token counting - 1 token â 4 characters)
        max_chars = 20000  # ~5000 tokens, leaving room for response
        if len(full_prompt) > max_chars:
            removed = len(full_prompt) - max_chars
            logger.warning(
                f"Prompt too long ({len(full_prompt)} chars), truncating by {removed} chars"
            )
            # Keep the end of the prompt which is usually most important
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

                Example of what NOT to do (schema): {{"$defs": {{"DialogueEntry": {{"properties": ...}}}}}}
                Example of what TO do (data): {{"entries": [{{"text": "Hello, it\\"s nice to meet you!", "speaker": "speaker-1"}}]}}

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

                Example of what NOT to do (schema): {{"$defs": {{"DialogueEntry": {{"properties": ...}}}}}}
                Example of what TO do (data): {{"entries": [{{"text": "Hello, it\\"s nice to meet you!", "speaker": "speaker-1"}}]}}

                Return only the actual JSON data now:
                """)
            schema_instruction = schema_instruction_template.format(schema_example=schema_example)
            data["prompt"] += schema_instruction
        
        async with httpx.AsyncClient(timeout=120.0) as client:
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
                                                return parsed_json
                                        except Exception:
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

    def _clean_and_parse_json(self, content: str) -> dict:
        """Clean and parse JSON response from Ollama."""
        # Explicitly import ujson here to avoid any local scope issues that could
        # lead to "local variable 'json' referenced before assignment" errors
        import ujson as json
        # Remove any leading/trailing whitespace
        content = content.strip()
        
        # Remove code block markers if present
        content = content.replace('```json', '').replace('```', '')
        
        # Try to find JSON in the response
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group()
        
        # Special handling for podcast JSON where the dialogue field is missing
        if '"scratchpad"' in content and re.search(r'"scratchpad"\s*:\s*"[^"]*"[^{]*\[\s*(?:INTRO|MUSIC|HOST|NARRATOR|BOB|KATE)', content, re.IGNORECASE):
            try:
                # Extract scratchpad
                scratchpad_match = re.search(r'"scratchpad"\s*:\s*"([^"]*)"', content)
                scratchpad = scratchpad_match.group(1) if scratchpad_match else ""
                
                # Extract the dialogue content (everything after scratchpad)
                dialogue_text = re.sub(r'^.*?"scratchpad"\s*:\s*"[^"]*"[^[]*', '', content, flags=re.DOTALL)
                
                # Reconstruct proper JSON
                fixed_json = '{' + f'"scratchpad": "{scratchpad}", "dialogue": {dialogue_text}'
                
                # Try to parse it
                import json
                return json.loads(fixed_json)
            except Exception as e:
                logger.warning(f"Special handling for podcast JSON failed: {e}")
        
        # Common JSON fixes
        content = content.replace('\n', ' ')  # Remove newlines
        content = re.sub(r',\s*}', '}', content)  # Remove trailing commas
        content = re.sub(r',\s*]', ']', content)  # Remove trailing commas in arrays
        content = re.sub(r'"\s+("|\[)', '": \1', content)  # Add missing colons
        content = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', content)  # Fix missing quotes
        
        # Try to parse the JSON first
        try:
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError as e:
            # If parsing fails, try more aggressive cleaning
            logger.warning(f"Initial JSON parse failed: {e}, attempting to clean...")
            
            # Fix escaped quotes in string values but preserve JSON structure
            # This regex finds string values and properly escapes quotes within them
            def fix_quotes_in_strings(match):
                key_part = match.group(1)  # "key":
                string_content = match.group(2)  # the content between quotes
                
                # Escape any unescaped quotes in the string content
                fixed_content = string_content.replace('\\"', '__ESCAPED_QUOTE__')  # Preserve already escaped quotes
                fixed_content = fixed_content.replace('"', '\\"')  # Escape unescaped quotes
                fixed_content = fixed_content.replace('__ESCAPED_QUOTE__', '\\"')  # Restore escaped quotes
                
                return f'{key_part}"{fixed_content}"'
            
            # Apply the fix to all string values in the JSON
            content = re.sub(r'("[^"]+"):\s*"([^"]*(?:\\"[^"]*)*)"', fix_quotes_in_strings, content)
            
            # Final attempt to parse
            try:
                parsed = json.loads(content)
                return parsed
            except json.JSONDecodeError as e2:
                # Try to extract a balanced JSON block if available
                extracted = self._extract_json_block(content)
                if extracted:
                    return json.loads(extracted)
                logger.error(f"JSON parsing failed even after cleaning. Error: {e2}")
                logger.error(f"Content: {content[:500]}...")
                
                # Last resort for podcast content - extract anything that looks like a dialogue
                try:
                    if "scratchpad" in content or "dialogue" in content:
                        dialogue_content = []
                        text_chunks = re.findall(r'(?:HOST|GUEST|NARRATOR|BOB|KATE):\s*([^\n]+)', content)
                        if text_chunks:
                            for i, chunk in enumerate(text_chunks):
                                speaker = "speaker-1" if i % 2 == 0 else "speaker-2"
                                dialogue_content.append({"text": chunk.strip(), "speaker": speaker})
                            
                            return {
                                "scratchpad": "",
                                "dialogue": dialogue_content
                            }
                except Exception:
                    pass
                    
                raise ValueError(f"Could not parse JSON: {e2}")

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
                        return self._clean_and_parse_json(full_content)
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

                            Example of what NOT to do (schema): {{"$defs": {{"DialogueEntry": {{"properties": ...}}}}}}
                            Example of what TO do (data): {{"entries": [{{"text": "Hello, it\\"s nice to meet you!", "speaker": "speaker-1"}}]}}

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