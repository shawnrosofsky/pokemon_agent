"""
OpenAI Client Module - Asynchronous client for OpenAI API.
"""

import time
import uuid
import queue
import json
from typing import Dict, Any, Optional
from openai import OpenAI

from llm_client_base import AsyncLLMClient


class AsyncOpenAIClient(AsyncLLMClient):
    """Asynchronous client for calling OpenAI API without blocking the main thread."""
    
    def __init__(self, api_key, model="gpt-4o", temperature=0.7, output_manager=None):
        """Initialize the async OpenAI client."""
        super().__init__(model_name=model, temperature=temperature, output_manager=output_manager)
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
    
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
                        
                        # Convert Anthropic-style messages to OpenAI format
                        openai_messages = self._convert_to_openai_messages(system, messages)
                        
                        # Build API call parameters
                        api_params = {
                            "model": self.model,
                            "messages": openai_messages,
                            "max_tokens": max_tokens,
                            "temperature": temperature
                        }
                        
                        # Only include tools if provided and not None
                        if tools is not None and request_type == 'action':
                            # Convert tools format from Anthropic to OpenAI
                            openai_tools = self._convert_to_openai_tools(tools)
                            api_params["tools"] = openai_tools
                            # We need to include tool_choice parameter
                            api_params["tool_choice"] = "auto"
                        
                        # Make the API call with the appropriate parameters
                        response = self.client.chat.completions.create(**api_params)
                        
                        # Convert OpenAI response to Anthropic-like format
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
        Queue an API call to OpenAI.
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
    
    def _convert_to_openai_messages(self, system, messages):
        """Convert Anthropic-style messages to OpenAI format."""
        openai_messages = []
        
        # Add system message
        if system:
            openai_messages.append({"role": "system", "content": system})
        
        # Convert each message
        for msg in messages:
            role = msg.get("role")
            if role == "user":
                openai_msg = {"role": "user", "content": []}
            elif role == "assistant":
                openai_msg = {"role": "assistant", "content": []}
            else:
                continue  # Skip unknown roles
            
            # Process content
            for content_item in msg.get("content", []):
                if isinstance(content_item, dict):
                    content_type = content_item.get("type")
                    
                    if content_type == "text":
                        openai_msg["content"].append({
                            "type": "text", 
                            "text": content_item.get("text", "")
                        })
                    elif content_type == "image":
                        # Handle base64 encoded images
                        image_data = content_item.get("source", {}).get("data", "")
                        media_type = content_item.get("source", {}).get("media_type", "image/png")
                        openai_msg["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            }
                        })
                    elif content_type == "tool_use":
                        # For tool uses in previous assistant messages
                        # OpenAI doesn't need this in the message history
                        pass
                    elif content_type == "tool_result":
                        # Handle tool results - in OpenAI this is a different message type
                        tool_use_id = content_item.get("tool_use_id", "")
                        tool_content = content_item.get("content", "")
                        
                        # For OpenAI, tool results are separate messages with role="tool"
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": str(tool_content)
                        })
                        
                        # Skip adding the current message as it's converted to a tool response
                        continue
                else:
                    # Simple string content (should not happen with Anthropic format)
                    openai_msg["content"].append({
                        "type": "text", 
                        "text": str(content_item)
                    })
            
            # If we have multiple content items but they're all text, collapse into a single string
            # This helps with compatibility for simpler OpenAI client versions
            if all(isinstance(c, dict) and c.get("type") == "text" for c in openai_msg["content"]):
                openai_msg["content"] = " ".join(c.get("text", "") for c in openai_msg["content"])
            
            # Add the processed message to the list
            if "content" in openai_msg and openai_msg["content"]:
                openai_messages.append(openai_msg)
        
        return openai_messages
    
    def _convert_to_openai_tools(self, tools):
        """Convert Anthropic-style tools to OpenAI format."""
        openai_tools = []
        
        for tool in tools:
            # The main difference is that OpenAI uses "function" instead of "input_schema"
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {})
                }
            }
            openai_tools.append(openai_tool)
        
        return openai_tools
    
    def _convert_to_anthropic_response(self, openai_response, request_type):
        """
        Convert OpenAI response to Anthropic-like format for consistent handling.
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
        
        # Extract the message from OpenAI response
        message = openai_response.choices[0].message
        anthropic_content = []
        
        # Handle text content
        if message.content:
            anthropic_content.append(AnthropicContent(
                type="text",
                text=message.content
            ))
        
        # Handle tool calls if present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                # Extract tool details
                function_call = tool_call.function
                tool_name = function_call.name
                
                try:
                    # Parse the arguments from JSON string
                    tool_input = json.loads(function_call.arguments)
                except:
                    # Fallback if JSON parsing fails
                    tool_input = function_call.arguments
                
                anthropic_content.append(AnthropicContent(
                    type="tool_use",
                    name=tool_name,
                    input=tool_input,
                    id=tool_call.id
                ))
        
        return AnthropicResponse(anthropic_content)