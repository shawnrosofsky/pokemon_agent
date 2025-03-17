#!/usr/bin/env python3
"""
Pokemon Agent - Fully Asynchronous version where everything runs in the background.
All API calls (including summarization) are done asynchronously while the game runs.
Fixed API issues with summary requests.
"""

import os
import time
import json
import uuid
import queue
import threading
import traceback
import anthropic
from typing import Dict, List, Any, Optional, Tuple

# Import our modules
# Make sure these modules are in the same directory or properly installed
from pokemon_emulator import GameEmulator
from pokemon_knowledge import KnowledgeBase
from pokemon_tools import PokemonTools


class OutputManager:
    """Manages output of Claude's thoughts and agent actions."""
    
    def __init__(self, output_to_file=False, log_file=None):
        self.output_to_file = output_to_file
        self.log_file = log_file or "pokemon_agent.log"
        self.lock = threading.Lock()
        
        # Create log file if needed
        if self.output_to_file:
            with open(self.log_file, 'w') as f:
                f.write(f"=== Pokémon Agent Log - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    def print(self, text, end="\n"):
        """Print text to console and optionally to file (thread-safe)."""
        with self.lock:
            print(text, end=end)
            
            if self.output_to_file:
                with open(self.log_file, 'a') as f:
                    f.write(text)
                    if end:
                        f.write(end)
    
    def print_section(self, title, content=None):
        """Print a formatted section with title (thread-safe)."""
        separator = "="*50
        self.print(f"\n{separator}")
        self.print(f"{title}")
        self.print(f"{separator}")
        
        if content:
            self.print(content)
            self.print(separator)


class AsyncClaudeClient:
    """Asynchronous client for calling Claude API without blocking the main thread."""
    
    def __init__(self, api_key, model="claude-3-7-sonnet-20250219", temperature=0.7, output_manager=None):
        """Initialize the async Claude client."""
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.client = anthropic.Anthropic(api_key=api_key)
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
    
    def call_claude(self, system, messages, tools=None, max_tokens=1000, temperature=None, request_id=None, request_type='action'):
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


class PokemonAgent:
    """LLM Agent for playing Pokémon using fully asynchronous Claude API calls."""
    
    def __init__(self, rom_path, model_name="claude-3-7-sonnet-20250219", 
                 api_key=None, temperature=0.7, headless=False, speed=1,
                 sound=True, output_to_file=False, log_file=None, summary_interval=10):
        """Initialize the agent."""
        # Setup API
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set ANTHROPIC_API_KEY env var or pass directly.")
        
        # Setup output manager first so we can use it for logging
        self.output = OutputManager(output_to_file=output_to_file, log_file=log_file)
        
        # Setup async Claude client
        self.model = model_name
        self.temperature = temperature
        self.claude_client = AsyncClaudeClient(
            api_key=self.api_key,
            model=model_name,
            temperature=temperature,
            output_manager=self.output
        )
        
        # Setup components
        self.emulator = GameEmulator(rom_path, headless=headless, speed=speed, sound=sound)
        self.kb = KnowledgeBase()
        self.tools_manager = PokemonTools(self.emulator, self.kb)
        
        # Conversation management
        self.message_history = []
        self.turn_count = 0
        self.max_turns_before_summary = summary_interval
        self.last_summary = ""
        
        # Current state tracking
        self.last_error = None
        
        # Action request tracking
        self.waiting_for_action = False
        self.action_request_id = None
        
        # Summary request tracking
        self.waiting_for_summary = False
        self.summary_request_id = None
        self.summary_due = False
        
        # Frame timing
        self.target_fps = 60
        self.frame_time = 1.0 / self.target_fps
    
    def start(self):
        """Start the game and async client."""
        # Start the game
        self.emulator.start_game(skip_intro=True)
        
        # Start the async client
        self.claude_client.start()
    
    def validate_message_history(self):
        """
        Validate and clean up message history to ensure valid conversation.
        Returns a proper sequence of messages ready for API use.
        """
        # Start with empty validated history
        valid_history = []
        
        # Keep track of whether the last message has a tool_use
        last_had_tool_use = False
        
        # Create valid message pairs
        for i in range(len(self.message_history)):
            msg = self.message_history[i]
            
            # Check for assistant messages with tool_use
            if msg["role"] == "assistant":
                has_tool_use = False
                for content in msg.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "tool_use":
                        has_tool_use = True
                        break
                
                # If this is a normal assistant message without tool_use, just add it
                if not has_tool_use:
                    valid_history.append(msg)
                    last_had_tool_use = False
                else:
                    # This is a tool_use message
                    # Check if next message is the tool_result
                    if i + 1 < len(self.message_history) and self.message_history[i+1]["role"] == "user":
                        next_msg = self.message_history[i+1]
                        has_tool_result = False
                        for content in next_msg.get("content", []):
                            if isinstance(content, dict) and content.get("type") == "tool_result":
                                has_tool_result = True
                                break
                        
                        # Only add the pair if next message has tool_result
                        if has_tool_result:
                            valid_history.append(msg)
                            valid_history.append(next_msg)
                            # Skip the next message since we already added it
                            i += 1
                            last_had_tool_use = False
                        else:
                            # If next message doesn't have tool_result, add only this message
                            valid_history.append(msg)
                            last_had_tool_use = True
                    else:
                        # No next message or not a user message
                        valid_history.append(msg)
                        last_had_tool_use = True
            
            # For user messages
            elif msg["role"] == "user":
                has_tool_result = False
                for content in msg.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "tool_result":
                        has_tool_result = True
                        break
                
                # Only add user messages with tool_result if last message had tool_use
                if has_tool_result and not last_had_tool_use:
                    # Skip this message as it contains a tool_result without a preceding tool_use
                    continue
                else:
                    valid_history.append(msg)
                    last_had_tool_use = False
        
        # Ensure we end with a message that doesn't require a response
        if valid_history and valid_history[-1]["role"] == "assistant" and last_had_tool_use:
            valid_history.pop()  # Remove the last message as it requires a tool_result
        
        # Take only the last few messages to keep context manageable
        return valid_history[-6:] if len(valid_history) > 6 else valid_history
    
    def request_next_action(self) -> bool:
        """
        Request the next action from Claude asynchronously.
        Returns True if request was made, False otherwise.
        """
        try:
            # If we're already waiting for an action response, don't make a new request
            if self.waiting_for_action:
                return False
            
            # Get game state
            game_state = self.emulator.get_game_state()
            screen_base64 = self.emulator.get_screen_base64()
            state_text = self.emulator.format_game_state(game_state)
            recent_actions = self.kb.get_recent_actions()
            
            # System prompt
            system_prompt = """
You are an expert Pokémon player. Analyze the game state and use a tool to take the best action.

First, think step-by-step about:
1. Where the player is currently and what's visible on screen
2. What progress has been made in the game so far
3. What the immediate goal should be
4. Available actions and their potential outcomes

Then, use a tool to execute the best action. Always use a tool - do not just describe what to do.
If you're unsure what to do, use wait_frames instead of pressing buttons randomly.
"""

            # Build message content
            message_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screen_base64
                    }
                },
                {
                    "type": "text",
                    "text": f"""
Current game state:
{state_text}

Recent actions:
{recent_actions}

{f"Note about previous action: {self.last_error}" if self.last_error else ""}

First, analyze what you see on screen and the current situation.
Then, use a tool to take the most appropriate action.
"""
                }
            ]

            # Validate message history
            valid_history = self.validate_message_history()
            
            # Add current message
            current_message = {
                "role": "user",
                "content": message_content
            }
            
            # Use validated history plus current message for API call
            messages = valid_history + [current_message]
            
            # Get tool definitions
            tools = self.tools_manager.define_tools()
            
            # Store message for history (we'll only keep it if the API call succeeds)
            self.temp_current_message = current_message
            
            # Generate a unique request ID
            request_id = str(uuid.uuid4())
            
            # Make asynchronous API call
            self.output.print_section("CALLING CLAUDE FOR ACTION")
            self.output.print(f"Waiting for Claude's action response (request ID: {request_id})")
            self.output.print("Game continues running while Claude is thinking...")
            
            self.claude_client.call_claude(
                system=system_prompt,
                messages=messages,
                tools=tools,
                max_tokens=1000,
                temperature=self.temperature,
                request_id=request_id,
                request_type='action'
            )
            
            # Start waiting for API response
            self.waiting_for_action = True
            self.action_request_id = request_id
            
            return True
            
        except Exception as e:
            self.output.print_section("ACTION REQUEST ERROR")
            self.output.print(f"Error: {str(e)}")
            traceback_str = traceback.format_exc()
            self.output.print(traceback_str)
            self.last_error = str(e)
            
            # Not waiting for API anymore
            self.waiting_for_action = False
            self.action_request_id = None
            
            return False
    
    def check_for_action_response(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if a response for an action request is available.
        Returns (success, action, result) tuple.
        If no response is available yet, returns (False, None, None).
        """
        if not self.waiting_for_action or self.action_request_id is None:
            return False, None, None
        
        # Check for response without blocking, but only for our specific action request
        api_result = self.claude_client.get_response(
            request_id=self.action_request_id, 
            request_type='action',
            block=False
        )
        
        if api_result is None:
            # No response yet for our action request
            return False, None, None
        
        # We got a response, so we're no longer waiting for an action
        self.waiting_for_action = False
        
        if not api_result['success']:
            # API call failed
            error_msg = api_result.get('error', 'Unknown error')
            self.output.print_section("ACTION API ERROR")
            self.output.print(f"Error: {error_msg}")
            self.last_error = error_msg
            
            # Execute a wait action as fallback
            result = self.emulator.wait_frames(30)
            self.kb.add_action("wait", "Error fallback: " + error_msg)
            
            return True, "wait", result
        
        # API call succeeded
        response = api_result['response']
        
        # Add the current message to history now that the API call succeeded
        self.message_history.append(self.temp_current_message)
        
        # Add Claude's response to history (convert from API response to storable dict)
        assistant_content = []
        for content in response.content:
            if content.type == "text":
                assistant_content.append({
                    "type": "text",
                    "text": content.text
                })
            elif content.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "name": content.name,
                    "input": content.input,
                    "id": content.id
                })
        
        assistant_message = {
            "role": "assistant",
            "content": assistant_content
        }
        self.message_history.append(assistant_message)
        
        # Print Claude's thought process
        self.output.print_section("CLAUDE'S THOUGHT PROCESS")
        
        # Process and print each content block
        tool_use = None
        for content in response.content:
            if content.type == "text":
                self.output.print(content.text)
            elif content.type == "tool_use":
                tool_use = content
                self.output.print_section("TOOL USE")
                self.output.print(f"Tool: {content.name}")
                self.output.print(f"Parameters: {content.input}")
        
        # Execute the tool if one was used
        if tool_use:
            # Extract tool information
            tool_name = tool_use.name
            tool_params = tool_use.input
            tool_id = tool_use.id
            
            # Execute the tool
            self.output.print_section("EXECUTING ACTION")
            self.output.print(f"Executing {tool_name} with parameters: {tool_params}")
            result = self.tools_manager.execute_tool(tool_name, tool_params)
            self.output.print(f"Result: {result}")
            
            # Determine the action name for tracking
            action = tool_name
            if tool_name == "press_button" and "button" in tool_params:
                action = tool_params["button"]
            
            # Add to knowledge base
            self.kb.add_action(action, result)
            
            # Add tool result to conversation history
            tool_result_message = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result
                    }
                ]
            }
            self.message_history.append(tool_result_message)
            
            return True, action, result
        else:
            # No tool was used, fall back to wait
            self.output.print_section("NO TOOL USED")
            self.output.print("Claude didn't use a tool. Falling back to wait...")
            result = self.emulator.wait_frames(30)
            self.kb.add_action("wait", "No tool used")
            return True, "wait", result
    
    def request_summarization(self) -> bool:
        """
        Request a game progress summary from Claude asynchronously.
        Returns True if request was made, False otherwise.
        """
        if self.waiting_for_summary:
            return False
            
        try:
            # Get game state for additional context
            game_state = self.emulator.get_game_state()
            state_text = self.emulator.format_game_state(game_state)
            
            # Make the summarization request
            self.output.print_section("REQUESTING SUMMARY")
            self.output.print(f"Requesting summary after {self.turn_count} turns")
            self.output.print("Game continues running during summarization...")
            
            # Generate a unique request ID
            request_id = str(uuid.uuid4())
            
            # Make asynchronous API call for summary - note that tools is INTENTIONALLY NOT included
            self.claude_client.call_claude(
                system="You are a Pokémon game expert. Provide a concise summary of the player's progress.",
                messages=[
                    {
                        "role": "user",
                        "content": f"""
Please summarize the recent game progress:

Current game state:
{state_text}

Recent actions:
{self.kb.get_recent_actions(20)}

Previous summary (if any):
{self.last_summary}

Summarize:
1. Current location and objective
2. Recent battles and encounters
3. Party status
4. Progress towards game goals

Keep the summary concise but informative.
"""
                    }
                ],
                max_tokens=500,
                temperature=self.temperature,
                request_id=request_id,
                request_type='summary'
                # NO tools parameter for summary requests - this is key!
            )
            
            # Start waiting for summary response
            self.waiting_for_summary = True
            self.summary_request_id = request_id
            
            return True
            
        except Exception as e:
            self.output.print_section("SUMMARY REQUEST ERROR")
            self.output.print(f"Error: {str(e)}")
            traceback_str = traceback.format_exc()
            self.output.print(traceback_str)
            
            # Not waiting for summary anymore
            self.waiting_for_summary = False
            self.summary_request_id = None
            
            return False
    
    def check_for_summary_response(self) -> bool:
        """
        Check if a summary response is available.
        Returns True if a summary was processed, False otherwise.
        """
        if not self.waiting_for_summary or self.summary_request_id is None:
            return False
        
        # Check for response without blocking, but only for our summary request
        api_result = self.claude_client.get_response(
            request_id=self.summary_request_id,
            request_type='summary',
            block=False
        )
        
        if api_result is None:
            # No response yet for our summary request
            return False
        
        # We got a response, so we're no longer waiting for a summary
        self.waiting_for_summary = False
        self.summary_due = False
        
        if not api_result['success']:
            # API call failed
            error_msg = api_result.get('error', 'Unknown error')
            self.output.print_section("SUMMARY API ERROR")
            self.output.print(f"Error: {error_msg}")
            # Continue without summary
            return True
        
        # API call succeeded
        response = api_result['response']
        
        # Extract summary text
        summary_text = ""
        for content in response.content:
            if content.type == "text":
                summary_text = content.text
                break
        
        # Update last summary
        self.last_summary = summary_text
        
        # Store in knowledge base
        self.kb.update("player_progress", "last_summary", summary_text)
        
        # Reset conversation and turn count
        self.turn_count = 0
        self.message_history = []
        
        if summary_text:
            self.message_history.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"Previous progress summary: {self.last_summary}"
                    }
                ]
            })
        
        self.output.print_section("SUMMARY", summary_text)
        self.output.print("Context reset with summary")
        
        return True
    
    def check_for_summarization(self):
        """Check if we need to summarize and reset conversation."""
        self.turn_count += 1
        
        # If it's time for a summary and we're not already waiting for one
        if self.turn_count >= self.max_turns_before_summary and not self.waiting_for_summary:
            self.summary_due = True
    
    def run(self, num_steps=None):
        """Run the agent for specified steps, keeping the game running continuously."""
        step_count = 0
        frames_run = 0
        last_frame_time = time.time()
        running = True
        
        try:
            self.output.print_section("STARTING POKÉMON AGENT", f"Model: {self.model}, Temperature: {self.temperature}")
            
            # Request the first action
            self.request_next_action()
            
            while running:
                # Calculate time since last frame
                now = time.time()
                elapsed = now - last_frame_time
                
                # If it's time for a new frame
                if elapsed >= self.frame_time:
                    # Run the frame
                    self.emulator.pyboy.tick()
                    frames_run += 1
                    last_frame_time = now
                
                # First, check if a summary is due and we're not already waiting for one
                if self.summary_due and not self.waiting_for_summary and not self.waiting_for_action:
                    # Request a summary asynchronously
                    self.request_summarization()
                
                # Then, check if a summary response is available
                if self.waiting_for_summary:
                    if self.check_for_summary_response():
                        # If we got a summary response, request the next action
                        if not self.waiting_for_action:
                            self.request_next_action()
                
                # Check for action response if we're waiting for one
                if self.waiting_for_action:
                    success, action, result = self.check_for_action_response()
                    
                    # If we got an action response
                    if success:
                        # Count this as a complete step
                        step_count += 1
                        self.output.print_section(f"STEP {step_count} SUMMARY", f"Action: {action}\nResult: {result}")
                        
                        # Check if we've reached the maximum number of steps
                        if num_steps is not None and step_count >= num_steps:
                            running = False
                            break
                        
                        # Show stats every 10 steps
                        if step_count % 10 == 0:
                            stats_text = "Action distribution:\n"
                            stats = self.tools_manager.get_stats()
                            total = sum(stats.values()) or 1
                            for act, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                                stats_text += f"  {act}: {count} ({count/total*100:.1f}%)\n"
                            self.output.print_section(f"STATISTICS AFTER {step_count} STEPS", stats_text)
                        
                        # Save state periodically
                        if step_count % 50 == 0:
                            save_path = self.emulator.save_state(f"step_{step_count}.state")
                            self.output.print(f"Saved game state to {save_path}")
                        
                        # Check for summarization
                        self.check_for_summarization()
                        
                        # If a summary is due, request it
                        if self.summary_due and not self.waiting_for_summary:
                            self.request_summarization()
                        else:
                            # Otherwise, request the next action
                            self.request_next_action()
                
                # If we're not waiting for anything, start a new action request
                if not self.waiting_for_action and not self.waiting_for_summary:
                    # First check if a summary is due
                    if self.summary_due:
                        self.request_summarization()
                    else:
                        # Otherwise, request the next action
                        self.request_next_action()
                
                # Sleep a tiny bit to avoid hogging the CPU
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            self.output.print_section("INTERRUPTED", "Stopped by user")
        except Exception as e:
            self.output.print_section("ERROR", f"Error running agent: {e}")
            traceback_str = traceback.format_exc()
            self.output.print(traceback_str)
        finally:
            # Save final state
            self.emulator.save_state("final.state")
            
            # Save action stats
            with open("action_stats.json", 'w') as f:
                json.dump(self.tools_manager.get_stats(), f, indent=2)
            
            self.output.print_section("RUN COMPLETE", f"Total steps: {step_count}")
            self.output.print(f"Total frames: {frames_run}")
            
            # Clean up
            self.close()
    
    def close(self):
        """Clean up resources."""
        self.claude_client.stop()
        self.emulator.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a Pokémon AI agent using Claude asynchronously')
    parser.add_argument('rom_path', help='Path to the Pokémon ROM file')
    parser.add_argument('--api-key', help='API key (will use ANTHROPIC_API_KEY env var if not provided)')
    parser.add_argument('--model', default='claude-3-7-sonnet-20250219', help='Model to use')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no window)')
    parser.add_argument('--steps', type=int, help='Number of steps to run (infinite if not specified)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for the LLM')
    parser.add_argument('--speed', type=int, default=1, help='Game speed multiplier')
    parser.add_argument('--no-sound', action='store_true', help='Disable game sound')
    parser.add_argument('--log-to-file', action='store_true', help='Log output to file')
    parser.add_argument('--log-file', help='Path to log file (default: pokemon_agent.log)')
    parser.add_argument('--fps', type=int, default=60, help='Target frames per second')
    parser.add_argument('--summary-interval', type=int, default=10, help='Number of turns between summaries (default: 10)')
    
    args = parser.parse_args()
    
    # Create and run the agent
    agent = PokemonAgent(
        rom_path=args.rom_path,
        model_name=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        headless=args.headless,
        speed=args.speed,
        sound=not args.no_sound,
        output_to_file=args.log_to_file,
        log_file=args.log_file,
        summary_interval=args.summary_interval
    )
    
    try:
        agent.start()
        agent.run(num_steps=args.steps)
    finally:
        agent.close()