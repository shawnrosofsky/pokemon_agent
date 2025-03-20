"""
Gemini Client Module - Asynchronous client for Google's Gemini API.
"""

import time
import uuid
import queue
import json
import base64
from typing import Dict, Any, Optional
from google import genai
from google.genai import types
from google.genai.types import FunctionDeclaration, Tool, GenerateContentConfig

from llm_client_base import AsyncLLMClient


class AsyncGeminiClient(AsyncLLMClient):
    """Asynchronous client for calling Google Gemini API without blocking the main thread."""
    
    def __init__(self, api_key, model="gemini-2.0-flash", temperature=0.7, output_manager=None):
        """Initialize the async Gemini client."""
        super().__init__(model_name=model, temperature=temperature, output_manager=output_manager)
        self.api_key = api_key
        # Initialize Google Gemini
        self.client = genai.Client(api_key=api_key)
        # Generation config (will be used in actual calls)
        self.generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
        }
    
    def _process_requests(self):
        """Process requests from the queue in a separate thread."""
        while self.running:
            try:
                # Get a request from the queue with a timeout
                try:
                    request = self.request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                try:
                    # Process the request
                    system = request.get('system', '')
                    messages = request.get('messages', [])
                    max_tokens = request.get('max_tokens', 1000)
                    temperature = request.get('temperature', self.temperature)
                    tools = request.get('tools', None)
                    request_id = request.get('request_id', None)
                    request_type = request.get('request_type', 'action')  # Default to action
                    
                    # Make the API call
                    try:
                        if self.output:
                            self.output.print(f"Processing {request_type} request ID: {request_id}")
                        
                        # Convert Anthropic-style messages to Gemini format
                        gemini_messages = self._convert_to_gemini_messages(system, messages)
                        
                        # Create model instance
                        # model = self.client.models
                        # model = genai.GenerativeModel(
                        #     model_name=self.model,
                        #     generation_config={
                        #         "temperature": temperature,
                        #         "max_output_tokens": max_tokens,
                        #         "top_p": 0.95,
                        #         "top_k": 64,
                        #     }
                        # )

                        # Prepare API call
                        if tools is not None and request_type == 'action':
                            # Convert tools format from Anthropic to Gemini
                            gemini_tools = self._convert_to_gemini_tools(tools)
                            tool_config={"function_calling_config": {"mode": "auto"}}
                            config = GenerateContentConfig(tools=gemini_tools, tool_config=tool_config)
                            response = self.client.models.generate_content(model=self.model, contents=gemini_messages, config=config)
                            # response = model.generate_content(
                            #     gemini_messages,
                            #     tools=gemini_tools,
                            #     tool_config={"function_calling_config": {"mode": "auto"}}
                            # )
                        else:
                            response = self.client.models.generate_content(contents=gemini_messages)
                        
                        # Convert Gemini response to Anthropic-like format
                        anthropic_response = self._convert_to_anthropic_response(response, request_type)
                        
                        if self.output:
                            self.output.print(f"Got response for {request_type} request ID: {request_id}")
                        
                        # Put the response in the response queue
                        self.response_queue.put({
                            'success': True,
                            'response': anthropic_response,
                            'request_id': request_id,
                            'request_type': request_type
                        })
                        
                    except Exception as e:
                        # Put the error in the response queue
                        error_msg = f"API Error: {str(e)}"
                        if self.output:
                            self.output.print(error_msg)
                        
                        self.response_queue.put({
                            'success': False,
                            'error': error_msg,
                            'request_id': request_id,
                            'request_type': request_type
                        })
                
                finally:
                    # Mark the request as done
                    self.request_queue.task_done()
                    
            except Exception as e:
                if self.output:
                    self.output.print(f"Error in async client thread: {str(e)}")
    
    def call_llm(self, system, messages, tools=None, max_tokens=1000, temperature=None, request_id=None, request_type='action'):
        """
        Queue an API call to Gemini.
        Returns immediately, response will be available in the response queue.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
            
        # Create request dictionary
        request = {
            'system': system,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature if temperature is not None else self.temperature,
            'request_id': request_id,
            'request_type': request_type
        }
        
        # Only include tools for action requests, not for summaries
        if tools is not None and request_type == 'action':
            request['tools'] = tools
        
        self.request_queue.put(request)
        return request_id
    
    def _convert_to_gemini_messages(self, system, messages):
        """Convert Anthropic-style messages to Gemini format."""
        gemini_messages = []
        
        # Add system message as a user message with specific indicator
        if system:
            gemini_messages.append({
                "role": "user",
                "parts": [f"System instruction: {system}"]
            })
            # Add a model response to acknowledge system instruction
            gemini_messages.append({
                "role": "model",
                "parts": ["I'll follow these instructions."]
            })
        
        # Convert each message
        for msg in messages:
            role = msg.get("role")
            
            # Map Anthropic roles to Gemini roles
            if role == "user":
                gemini_role = "user"
            elif role == "assistant":
                gemini_role = "model"
            else:
                continue  # Skip unknown roles
            
            # Process content
            parts = []
            for content_item in msg.get("content", []):
                if isinstance(content_item, dict):
                    content_type = content_item.get("type")
                    
                    if content_type == "text":
                        parts.append(content_item.get("text", ""))
                    elif content_type == "image":
                        # Handle base64 encoded images
                        image_data = content_item.get("source", {}).get("data", "")
                        parts.append({
                            "inline_data": {
                                "mime_type": content_item.get("source", {}).get("media_type", "image/png"),
                                "data": image_data
                            }
                        })
                    elif content_type == "tool_use":
                        # For tool uses in previous assistant messages
                        # We'll handle these differently for Gemini
                        if gemini_role == "model":
                            parts.append(f"I want to use the tool: {content_item.get('name')} with parameters: {json.dumps(content_item.get('input', {}))}")
                    elif content_type == "tool_result":
                        # For tool results in user messages
                        parts.append(f"Tool result: {content_item.get('content', '')}")
                else:
                    # Simple string content (should not happen with Anthropic format)
                    parts.append(str(content_item))
            
            # Add the processed message to the list if it has parts
            if parts:
                gemini_messages.append({
                    "role": gemini_role,
                    "parts": parts
                })
        
        return gemini_messages
    
    def _convert_to_gemini_tools(self, tools):
        """Convert Anthropic-style tools to Gemini format."""
        gemini_tools = []
        
        for tool in tools:
            tool_name = tool.get("name")
            description = tool.get("description", "")
            params = tool.get("input_schema", {})
            
            # Create FunctionDeclaration for this tool
            function_declaration = FunctionDeclaration(
                name=tool_name,
                description=description,
                parameters=params
            )
            
            gemini_tools.append(Tool(function_declarations=[function_declaration]))
        
        return gemini_tools
    
    def _convert_to_anthropic_response(self, gemini_response, request_type):
        """
        Convert Gemini response to Anthropic-like format for consistent handling.
        Creates a compatible response object with the same interface as Anthropic's.
        """
        # Create a response class to mimic Anthropic's response structure
        class AnthropicContent:
            def __init__(self, type, text=None, name=None, input=None, id=None):
                self.type = type
                self.text = text
                self.name = name 
                self.input = input
                self.id = id
        
        class AnthropicResponse:
            def __init__(self, content):
                self.content = content
        
        anthropic_content = []
        
        # Check if response has function calls
        function_calls = []
        try:
            if hasattr(gemini_response, 'candidates') and gemini_response.candidates:
                candidate = gemini_response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    # Extract text content
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    
                    if text_parts:
                        anthropic_content.append(AnthropicContent(
                            type="text",
                            text="\n".join(text_parts)
                        ))
                    
                    # Extract function calls if present
                    if hasattr(candidate, 'function_calls') and candidate.function_calls:
                        for func_call in candidate.function_calls:
                            function_name = func_call.name
                            # Parse arguments
                            try:
                                args = json.loads(func_call.args)
                            except:
                                args = {}
                            
                            anthropic_content.append(AnthropicContent(
                                type="tool_use",
                                name=function_name,
                                input=args,
                                id=str(uuid.uuid4())  # Gemini doesn't provide IDs, so we generate one
                            ))
        except Exception as e:
            # Fallback if parsing fails
            anthropic_content.append(AnthropicContent(
                type="text",
                text=f"Error parsing Gemini response: {str(e)}\n\nRaw response: {str(gemini_response)}"
            ))
        
        return AnthropicResponse(anthropic_content)