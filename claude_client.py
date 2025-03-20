"""
Claude Client Module - Asynchronous client for the Anthropic Claude API.
"""

import time
import uuid
import queue
import anthropic
from typing import Dict, Any, Optional

from llm_client_base import AsyncLLMClient


class AsyncClaudeClient(AsyncLLMClient):
    """Asynchronous client for calling Claude API without blocking the main thread."""
    
    def __init__(self, api_key, model="claude-3-7-sonnet-20250219", temperature=0.7, output_manager=None):
        """Initialize the async Claude client."""
        super().__init__(model_name=model, temperature=temperature, output_manager=output_manager)
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=api_key)
    
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
                        
                        # Build API call parameters
                        api_params = {
                            "model": self.model,
                            "system": system,
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": temperature
                        }
                        
                        # Only include tools if provided and not None - this is especially 
                        # important for summary requests which don't need tools
                        if tools is not None:
                            api_params["tools"] = tools
                        
                        # Make the API call with the appropriate parameters
                        response = self.client.messages.create(**api_params)
                        
                        if self.output:
                            self.output.print(f"Got response for {request_type} request ID: {request_id}")
                        
                        # Put the response in the response queue
                        self.response_queue.put({
                            'success': True,
                            'response': response,
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
        Queue an API call to Claude.
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