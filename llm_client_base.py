"""
Base LLM Client Module - Abstract base class for asynchronous LLM clients.
Provides a common interface for different LLM providers.
"""

import abc
import queue
import threading
import uuid
import time
from typing import Dict, List, Any, Optional, Tuple


class AsyncLLMClient(abc.ABC):
    """Abstract base class for asynchronous LLM API clients."""
    
    def __init__(self, model_name, temperature=0.7, output_manager=None):
        """Initialize the async LLM client."""
        self.model = model_name
        self.temperature = temperature
        self.output = output_manager
        
        # Queue for API call requests
        self.request_queue = queue.Queue()
        # Queue for API call responses
        self.response_queue = queue.Queue()
        
        # Flag to control the thread
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the async client thread."""
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self.thread = threading.Thread(target=self._process_requests, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop the async client thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
    
    @abc.abstractmethod
    def _process_requests(self):
        """
        Process requests from the queue in a separate thread.
        This method must be implemented by concrete subclasses.
        """
        pass
    
    @abc.abstractmethod
    def call_llm(self, system, messages, tools=None, max_tokens=1000, temperature=None, request_id=None, request_type='action'):
        """
        Queue an API call to the LLM.
        Returns immediately, response will be available in the response queue.
        
        Returns:
            request_id (str): ID of the queued request
        """
        pass
    
    def get_response(self, request_id=None, request_type=None, block=False, timeout=None):
        """
        Get a response from the response queue.
        If request_id is provided, only return a response with that ID.
        If request_type is provided, only return a response of that type.
        If block is True, wait until a response is available or timeout is reached.
        If block is False, return None if no response is available.
        """
        # If we want a specific response
        if request_id is not None or request_type is not None:
            # If both are provided, match both
            if request_id is not None and request_type is not None:
                # Look for a specific request_id AND request_type
                while True:
                    try:
                        response = self.response_queue.get(block=block, timeout=timeout)
                        if response['request_id'] == request_id and response['request_type'] == request_type:
                            return response
                        else:
                            # Put it back in the queue and try again
                            self.response_queue.put(response)
                            if not block:
                                return None
                    except queue.Empty:
                        return None
            
            # If only request_id is provided
            elif request_id is not None:
                # Look for a specific request_id
                while True:
                    try:
                        response = self.response_queue.get(block=block, timeout=timeout)
                        if response['request_id'] == request_id:
                            return response
                        else:
                            # Put it back in the queue and try again
                            self.response_queue.put(response)
                            if not block:
                                return None
                    except queue.Empty:
                        return None
            
            # If only request_type is provided
            else:  # request_type is not None
                # Look for a specific request_type
                while True:
                    try:
                        response = self.response_queue.get(block=block, timeout=timeout)
                        if response['request_type'] == request_type:
                            return response
                        else:
                            # Put it back in the queue and try again
                            self.response_queue.put(response)
                            if not block:
                                return None
                    except queue.Empty:
                        return None
        else:
            # Just get any response
            try:
                return self.response_queue.get(block=block, timeout=timeout)
            except queue.Empty:
                return None