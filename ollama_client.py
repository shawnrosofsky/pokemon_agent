"""
Ollama Client Module - Asynchronous client for local Ollama models.
"""

import time
import uuid
import queue
import json
import base64
import requests
from typing import Dict, Any, Optional

from llm_client_base import AsyncLLMClient


class AsyncOllamaClient(AsyncLLMClient):
    """Asynchronous client for calling local Ollama models without blocking the main thread."""
    
    def __init__(self, model="llama3", temperature=0.7, api_base="http://localhost:11434", output_manager=None):
        """Initialize the async Ollama client."""
        super().__init__(model_name=model, temperature=temperature, output_manager=output_manager)
        self.api_base = api_base
    
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
                        
                        # Convert Anthropic-style messages to Ollama format
                        ollama_messages = self._convert_to_ollama_messages(system, messages)
                        
                        # For Ollama, tools are embedded in the prompt since function calling
                        # is not natively supported in most local models
                        if tools is not None and request_type == 'action':
                            # Add tools description to the system prompt
                            tools_desc = self._format_tools_for_prompt(tools)
                            ollama_messages = self._add_tools_to_messages(ollama_messages, tools_desc)
                        
                        # Prepare API call parameters
                        api_params = {
                            "model": self.model,
                            "messages": ollama_messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "stream": False
                        }
                        
                        # Make the API call
                        response = requests.post(
                            f"{self.api_base}/api/chat",
                            json=api_params
                        )
                        response.raise_for_status()
                        ollama_response = response.json()
                        
                        # Convert Ollama response to Anthropic-like format
                        anthropic_response = self._convert_to_anthropic_response(ollama_response, tools, request_type)
                        
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
        Queue an API call to Ollama.
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
    
    def _convert_to_ollama_messages(self, system, messages):
        """Convert Anthropic-style messages to Ollama format."""
        ollama_messages = []
        
        # Add system message
        if system:
            ollama_messages.append({
                "role": "system",
                "content": system
            })
        
        # Convert each message
        for msg in messages:
            role = msg.get("role")
            
            # Map Anthropic roles to Ollama roles
            if role == "user":
                ollama_role = "user"
            elif role == "assistant":
                ollama_role = "assistant"
            else:
                continue  # Skip unknown roles
            
            # Process content
            content_parts = []
            for content_item in msg.get("content", []):
                if isinstance(content_item, dict):
                    content_type = content_item.get("type")
                    
                    if content_type == "text":
                        content_parts.append(content_item.get("text", ""))
                    elif content_type == "image":
                        # Handle base64 encoded images - most Ollama models aren't multimodal
                        # so we'll add a note about this
                        content_parts.append("[Image was provided but this model may not support images]")
                    elif content_type == "tool_use":
                        # For tool uses in previous assistant messages
                        tool_name = content_item.get("name", "")
                        tool_input = content_item.get("input", {})
                        content_parts.append(f"I'll use the {tool_name} tool with these parameters: {json.dumps(tool_input)}")
                    elif content_type == "tool_result":
                        # For tool results in user messages
                        content_parts.append(f"Tool result: {content_item.get('content', '')}")
                else:
                    # Simple string content (should not happen with Anthropic format)
                    content_parts.append(str(content_item))
            
            # Add the processed message to the list if it has content
            if content_parts:
                ollama_messages.append({
                    "role": ollama_role,
                    "content": "\n".join(content_parts)
                })
        
        return ollama_messages
    
    def _format_tools_for_prompt(self, tools):
        """Format tools as a text description for embedding in the prompt."""
        tools_desc = "You have the following tools available:\n\n"
        
        for i, tool in enumerate(tools):
            tool_name = tool.get("name")
            description = tool.get("description", "")
            params = tool.get("input_schema", {})
            
            tools_desc += f"Tool {i+1}: {tool_name}\n"
            tools_desc += f"Description: {description}\n"
            
            # Format parameters
            if "properties" in params:
                tools_desc += "Parameters:\n"
                for param_name, param_info in params["properties"].items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    required = "required" if "required" in params and param_name in params["required"] else "optional"
                    
                    tools_desc += f"- {param_name} ({param_type}, {required}): {param_desc}\n"
            
            tools_desc += "\n"
        
        tools_desc += "\nWhen you want to use a tool, respond with JSON in this exact format:\n"
        tools_desc += '{"tool": "tool_name", "parameters": {"param1": "value1", "param2": "value2"}}\n'
        tools_desc += "\nMake sure to use valid JSON with double quotes. After your reasoning, always include the JSON tool call in this format if you want to use a tool.\n\n"
        
        return tools_desc
    
    def _add_tools_to_messages(self, messages, tools_desc):
        """Add tools description to the messages."""
        # If there's a system message, append the tools description
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += "\n\n" + tools_desc
        else:
            # Otherwise, create a system message with tools description
            messages.insert(0, {
                "role": "system",
                "content": tools_desc
            })
        
        return messages
    
    def _convert_to_anthropic_response(self, ollama_response, tools, request_type):
        """
        Convert Ollama response to Anthropic-like format for consistent handling.
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
        
        # Extract the text response
        text = ollama_response.get("message", {}).get("content", "")
        
        # Parse tool use from the response text
        # Look for JSON formatted tool calls in the text
        text_parts = []
        tool_call = None
        
        # Try to find JSON-formatted tool calls
        import re
        json_pattern = r'\{[\s\S]*?"tool"[\s\S]*?\}'
        json_matches = re.findall(json_pattern, text)
        
        if json_matches:
            # Use the last match which is most likely to be the final decision
            json_str = json_matches[-1]
            try:
                tool_data = json.loads(json_str)
                if "tool" in tool_data and "parameters" in tool_data:
                    # Found a valid tool call
                    tool_name = tool_data["tool"]
                    parameters = tool_data["parameters"]
                    
                    # Validate the tool exists
                    if tools and any(t.get("name") == tool_name for t in tools):
                        tool_call = {
                            "name": tool_name,
                            "parameters": parameters
                        }
                        
                        # Remove the tool call JSON from the text
                        text = text.replace(json_str, "")
            except json.JSONDecodeError:
                # If JSON parsing fails, just keep the text as is
                pass
        
        # Add text content first
        if text.strip():
            anthropic_content.append(AnthropicContent(
                type="text",
                text=text.strip()
            ))
        
        # Add tool call if found
        if tool_call:
            anthropic_content.append(AnthropicContent(
                type="tool_use",
                name=tool_call["name"],
                input=tool_call["parameters"],
                id=str(uuid.uuid4())  # Generate a unique ID
            ))
        
        return AnthropicResponse(anthropic_content)