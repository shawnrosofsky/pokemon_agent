import os
import json
import time
import base64
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

class ClaudeDebugger:
    """
    Debugging tool for Claude API interactions that logs all inputs and outputs.
    Can be enabled/disabled as needed.
    """
    
    def __init__(self, 
                 enabled: bool = False, 
                 log_dir: str = "claude_logs",
                 console_output: bool = True,
                 log_level: int = logging.DEBUG,
                 max_image_log_size: int = 1000  # Characters to log for base64 images
                ):
        """
        Initialize the Claude debugger.
        
        Args:
            enabled (bool): Whether debugging is enabled
            log_dir (str): Directory to save logs
            console_output (bool): Whether to also output logs to console
            log_level (int): Logging level
            max_image_log_size (int): Maximum characters to log for base64 images
        """
        self.enabled = enabled
        self.log_dir = log_dir
        self.console_output = console_output
        self.max_image_log_size = max_image_log_size
        
        if enabled:
            self._setup_logging(log_level)
    
    def _setup_logging(self, log_level: int):
        """Set up the logger."""
        # Create log directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Create a timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"claude_debug_{timestamp}.log")
        
        # Configure logging
        self.logger = logging.getLogger("claude_debug")
        self.logger.setLevel(log_level)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        
        # Add console handler if requested
        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.logger.info("Claude Debugger initialized")
        self.log_file_path = log_file
    
    def enable(self):
        """Enable debugging."""
        if not self.enabled:
            self.enabled = True
            self._setup_logging(logging.DEBUG)
            self.logger.info("Claude Debugger enabled")
    
    def disable(self):
        """Disable debugging."""
        if self.enabled:
            self.logger.info("Claude Debugger disabled")
            self.enabled = False
            
            # Close all handlers
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
    
    def log_request(self, 
                    system_message: Optional[str] = None,
                    user_messages: List[Dict[str, Any]] = None,
                    model: str = None,
                    temperature: float = None,
                    max_tokens: int = None,
                    request_id: Optional[str] = None):
        """
        Log a request to the Claude API.
        
        Args:
            system_message: The system message
            user_messages: List of user message content objects
            model: The model name
            temperature: Temperature setting
            max_tokens: Max tokens setting
            request_id: Optional request ID for tracking
        """
        if not self.enabled:
            return
        
        # Create a request ID if none provided
        if request_id is None:
            request_id = f"req_{int(time.time())}"
        
        self.logger.info(f"REQUEST START [{request_id}] ==============================")
        self.logger.info(f"Model: {model}, Temp: {temperature}, Max Tokens: {max_tokens}")
        
        # Log system message
        if system_message:
            self.logger.info(f"SYSTEM MESSAGE [{request_id}]:")
            self.logger.info(f"\n{system_message}")
        
        # Log user messages, handling both text and images
        if user_messages:
            self.logger.info(f"USER MESSAGES [{request_id}]:")
            for i, msg in enumerate(user_messages):
                if isinstance(msg, dict) and 'content' in msg:
                    content = msg['content']
                    if isinstance(content, list):
                        # Handle content arrays (text and images)
                        for j, item in enumerate(content):
                            if isinstance(item, dict):
                                if item.get('type') == 'text':
                                    self.logger.info(f"Message {i+1}.{j+1} (text): {item.get('text')}")
                                elif item.get('type') == 'image':
                                    # Handle images - log truncated base64 data
                                    if 'image_url' in item and 'url' in item['image_url']:
                                        url = item['image_url']['url']
                                        if url.startswith('data:image'):
                                            # Extract base64 part and truncate
                                            base64_data = url.split(',')[1]
                                            truncated = f"{base64_data[:self.max_image_log_size]}..."
                                            self.logger.info(f"Message {i+1}.{j+1} (image): [base64 image data, length={len(base64_data)}]")
                                            # Save full image data to separate file
                                            self._save_image_data(base64_data, request_id, i, j)
                                        else:
                                            self.logger.info(f"Message {i+1}.{j+1} (image): {url}")
                    elif isinstance(content, str):
                        self.logger.info(f"Message {i+1} (text): {content}")
                else:
                    self.logger.info(f"Message {i+1}: {msg}")
        
        self.logger.info(f"REQUEST END [{request_id}] ================================")
        return request_id
    
    def log_response(self, response: Any, request_id: Optional[str] = None, timing_ms: Optional[float] = None):
        """
        Log a response from the Claude API.
        
        Args:
            response: The response object from Claude
            request_id: Request ID for tracking (should match request)
            timing_ms: Optional timing information in milliseconds
        """
        if not self.enabled:
            return
        
        # Use the provided request ID or create one
        if request_id is None:
            request_id = f"resp_{int(time.time())}"
        
        self.logger.info(f"RESPONSE START [{request_id}] {timing_ms}ms ===============================")
        
        # Handle different response formats
        try:
            if hasattr(response, 'content'):
                # Handle LangChain response objects
                if isinstance(response.content, list):
                    for i, item in enumerate(response.content):
                        self.logger.info(f"Content {i+1}: {item}")
                else:
                    self.logger.info(f"Content: {response.content}")
            elif hasattr(response, 'text'):
                # Handle raw Anthropic response objects with text attribute
                self.logger.info(f"Response text: {response.text}")
            elif isinstance(response, str):
                # Handle string responses
                self.logger.info(f"Response text: {response}")
            elif isinstance(response, dict):
                # Handle dictionary responses
                if 'content' in response:
                    self.logger.info(f"Response content: {response['content']}")
                else:
                    # Pretty print the dictionary
                    self.logger.info(f"Response dict: {json.dumps(response, indent=2)}")
            else:
                # Default case
                self.logger.info(f"Response (type {type(response)}): {response}")
        except Exception as e:
            self.logger.error(f"Error logging response: {e}")
            self.logger.info(f"Raw response: {response}")
        
        self.logger.info(f"RESPONSE END [{request_id}] =================================")
    
    def _save_image_data(self, base64_data: str, request_id: str, msg_idx: int, item_idx: int):
        """Save image data to a separate file to avoid cluttering the log."""
        if not self.enabled:
            return
        
        try:
            # Create an images directory under logs
            img_dir = os.path.join(self.log_dir, 'images')
            os.makedirs(img_dir, exist_ok=True)
            
            # Create a filename
            filename = f"{request_id}_msg{msg_idx+1}_{item_idx+1}.png"
            filepath = os.path.join(img_dir, filename)
            
            # Decode and save the image
            image_data = base64.b64decode(base64_data)
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            self.logger.info(f"Saved image to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving image data: {e}")
    
    def wrap_anthropic_client(self, client):
        """
        Wrap an Anthropic client to log all interactions.
        
        Args:
            client: The Anthropic client to wrap
            
        Returns:
            A wrapped client with logging
        """
        if not self.enabled:
            return client
        
        original_messages_create = client.messages.create
        
        def wrapped_messages_create(*args, **kwargs):
            # Log the request
            system = kwargs.get('system')
            messages = kwargs.get('messages', [])
            model = kwargs.get('model')
            temperature = kwargs.get('temperature')
            max_tokens = kwargs.get('max_tokens')
            
            start_time = time.time()
            request_id = self.log_request(
                system_message=system,
                user_messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Make the actual request
            try:
                response = original_messages_create(*args, **kwargs)
                # Calculate timing
                end_time = time.time()
                timing_ms = round((end_time - start_time) * 1000)
                
                # Log the response
                self.log_response(response, request_id, timing_ms)
                return response
            except Exception as e:
                self.logger.error(f"ERROR in API call [{request_id}]: {e}")
                raise
        
        # Replace the create method with our wrapped version
        client.messages.create = wrapped_messages_create
        return client
    
    def wrap_langchain_llm(self, llm):
        """
        Wrap a LangChain LLM to log all interactions.
        
        Args:
            llm: The LangChain LLM to wrap
            
        Returns:
            A wrapped LLM with logging
        """
        if not self.enabled:
            return llm
        
        original_invoke = llm.invoke
        
        def wrapped_invoke(messages, *args, **kwargs):
            # Log the request
            system_message = None
            user_messages = []
            
            # Extract system message and user messages
            for msg in messages:
                if hasattr(msg, 'type') and msg.type == 'system':
                    system_message = msg.content
                elif hasattr(msg, 'type') and msg.type == 'human':
                    user_messages.append({"content": msg.content})
            
            start_time = time.time()
            request_id = self.log_request(
                system_message=system_message,
                user_messages=user_messages,
                model=getattr(llm, 'model_name', None),
                temperature=getattr(llm, 'temperature', None)
            )
            
            # Make the actual request
            try:
                response = original_invoke(messages, *args, **kwargs)
                # Calculate timing
                end_time = time.time()
                timing_ms = round((end_time - start_time) * 1000)
                
                # Log the response
                self.log_response(response, request_id, timing_ms)
                return response
            except Exception as e:
                self.logger.error(f"ERROR in LangChain LLM invoke [{request_id}]: {e}")
                raise
        
        # Replace the invoke method with our wrapped version
        llm.invoke = wrapped_invoke
        return llm

# Example usage
if __name__ == "__main__":
    # Create a debugger and enable it
    debugger = ClaudeDebugger(enabled=True)
    
    # Example with Anthropic client
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    
    # Wrap the client with the debugger
    wrapped_client = debugger.wrap_anthropic_client(client)
    
    # Now all calls to wrapped_client.messages.create will be logged
    
    # Example with LangChain
    from langchain_anthropic import ChatAnthropic
    
    llm = ChatAnthropic(model_name="claude-3-7-sonnet-20250219", anthropic_api_key=api_key)
    wrapped_llm = debugger.wrap_langchain_llm(llm)
    
    # Now all calls to wrapped_llm.invoke will be logged