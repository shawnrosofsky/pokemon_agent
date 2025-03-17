#!/usr/bin/env python3
"""
Pokemon Agent with Threading - Main module for the Pokemon AI agent system.
This version runs the emulator in a background thread so it continues
running while Claude is thinking about its next action.
"""

import os
import time
import json
import traceback
import threading
import queue
import anthropic
from typing import Dict, List, Any, Optional, Tuple

# Import our modules
# Make sure these modules are in the same directory or properly installed
# from pokemon_emulator import GameEmulator
# from pokemon_knowledge import KnowledgeBase
# from pokemon_tools import PokemonTools


class OutputManager:
    """Manages output of Claude's thoughts and agent actions."""
    
    def __init__(self, output_to_file=False, log_file=None):
        self.output_to_file = output_to_file
        self.log_file = log_file or "pokemon_agent.log"
        
        # Create log file if needed
        if self.output_to_file:
            with open(self.log_file, 'w') as f:
                f.write(f"=== Pokémon Agent Log - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    def print(self, text, end="\n"):
        """Print text to console and optionally to file."""
        print(text, end=end)
        
        if self.output_to_file:
            with open(self.log_file, 'a') as f:
                f.write(text)
                if end:
                    f.write(end)
    
    def print_section(self, title, content=None):
        """Print a formatted section with title."""
        separator = "="*50
        self.print(f"\n{separator}")
        self.print(f"{title}")
        self.print(f"{separator}")
        
        if content:
            self.print(content)
            self.print(separator)


class EmulatorThread(threading.Thread):
    """Thread to run the emulator continuously."""
    
    def __init__(self, emulator, tick_rate=60):
        """Initialize the emulator thread."""
        super().__init__(daemon=True)  # Use daemon thread so it exits when main thread exits
        self.emulator = emulator
        self.tick_rate = tick_rate
        self.running = True
        self.paused = False
        self.action_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.lock = threading.Lock()
    
    def run(self):
        """Main thread loop to run the emulator continuously."""
        while self.running:
            # Process any actions in the queue
            try:
                while not self.action_queue.empty():
                    action = self.action_queue.get_nowait()
                    if action["type"] == "button":
                        button = action["button"]
                        hold_frames = action.get("hold_frames", 10)
                        result = self.emulator.press_button(button, hold_frames)
                        self.result_queue.put({"action": f"button_{button}", "result": result})
                    elif action["type"] == "wait":
                        num_frames = action.get("num_frames", 30)
                        result = self.emulator.wait_frames(num_frames)
                        self.result_queue.put({"action": "wait", "result": result})
                    elif action["type"] == "save_state":
                        filename = action.get("filename", "auto_save.state")
                        result = self.emulator.save_state(filename)
                        self.result_queue.put({"action": "save_state", "result": result})
                    elif action["type"] == "load_state":
                        filename = action.get("filename")
                        result = self.emulator.load_state(filename)
                        self.result_queue.put({"action": "load_state", "result": result})
                    
                    # Mark this action as completed
                    self.action_queue.task_done()
            except Exception as e:
                print(f"Error in emulator thread: {e}")
            
            # If not paused, tick the emulator
            if not self.paused:
                with self.lock:
                    self.emulator.pyboy.tick()
            
            # Sleep to maintain frame rate
            time.sleep(1 / self.tick_rate)
    
    def press_button(self, button, hold_frames=10):
        """Queue a button press action."""
        self.action_queue.put({
            "type": "button",
            "button": button,
            "hold_frames": hold_frames
        })
    
    def wait_frames(self, num_frames=30):
        """Queue a wait action."""
        self.action_queue.put({
            "type": "wait",
            "num_frames": num_frames
        })
    
    def save_state(self, filename="auto_save.state"):
        """Queue a save state action."""
        self.action_queue.put({
            "type": "save_state",
            "filename": filename
        })
    
    def load_state(self, filename):
        """Queue a load state action."""
        self.action_queue.put({
            "type": "load_state",
            "filename": filename
        })
    
    def get_game_state(self):
        """Get the current game state (thread-safe)."""
        with self.lock:
            return self.emulator.get_game_state()
    
    def get_screen_base64(self):
        """Get the current screen as base64 (thread-safe)."""
        with self.lock:
            return self.emulator.get_screen_base64()
    
    def pause(self):
        """Pause the emulator."""
        self.paused = True
    
    def resume(self):
        """Resume the emulator."""
        self.paused = False
    
    def stop(self):
        """Stop the emulator thread."""
        self.running = False
        # Wait for pending actions to complete
        self.action_queue.join()


class PokemonAgent:
    """LLM Agent for playing Pokémon using Anthropic Claude with threading."""
    
    def __init__(self, rom_path, model_name="claude-3-7-sonnet-20250219", 
                 api_key=None, temperature=0.7, headless=False, speed=1,
                 sound=True, output_to_file=False, log_file=None,
                 emulator_tick_rate=60):
        """Initialize the agent."""
        # Setup API
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set ANTHROPIC_API_KEY env var or pass directly.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model_name
        self.temperature = temperature
        
        # Setup components
        self.emulator = GameEmulator(rom_path, headless=headless, speed=speed, sound=sound)
        self.emulator_thread = EmulatorThread(self.emulator, tick_rate=emulator_tick_rate)
        self.kb = KnowledgeBase()
        self.tools_manager = PokemonTools(self.emulator, self.kb)
        
        # Setup output manager
        self.output = OutputManager(output_to_file=output_to_file, log_file=log_file)
        
        # Conversation management
        self.message_history = []
        self.turn_count = 0
        self.max_turns_before_summary = 10
        self.last_summary = ""
        
        # Last error tracking
        self.last_error = None
        
        # Thread control
        self.emulator_thread_started = False
    
    def start(self):
        """Start the game and emulator thread."""
        # Start the game
        self.emulator.start_game(skip_intro=True)
        
        # Start the emulator thread
        if not self.emulator_thread_started:
            self.emulator_thread.start()
            self.emulator_thread_started = True
    
    def get_action(self) -> Tuple[str, str]:
        """
        Get current game state, analyze with Claude, and execute the action.
        The emulator continues to run in the background during this process.
        
        Returns:
            Tuple of (action, result)
        """
        try:
            # Get game state
            game_state = self.emulator_thread.get_game_state()
            screen_base64 = self.emulator_thread.get_screen_base64()
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

            # Process only the last few messages to keep context manageable
            # and ensure we're not including invalid message structures
            pruned_history = []
            tool_call_message_indices = []
            
            # Find all messages with tool_use content
            for i, msg in enumerate(self.message_history):
                if msg.get("role") == "assistant":
                    for content in msg.get("content", []):
                        if isinstance(content, dict) and content.get("type") == "tool_use":
                            tool_call_message_indices.append(i)
                            break
            
            # Build valid conversation history
            valid_messages = []
            skip_next = False
            for i, msg in enumerate(self.message_history[-10:]):  # Only consider last 10 messages
                if skip_next:
                    skip_next = False
                    continue
                    
                if i in tool_call_message_indices:
                    # If we're about to include a tool_use message, ensure we also include the tool_result
                    if i + 1 < len(self.message_history) and self.message_history[i+1].get("role") == "user":
                        valid_messages.append(msg)
                        valid_messages.append(self.message_history[i+1])
                        skip_next = True
                    else:
                        # Skip this tool_use message if there's no corresponding tool_result
                        continue
                else:
                    valid_messages.append(msg)
            
            # Add current message
            current_message = {
                "role": "user",
                "content": message_content
            }
            
            # Use pruned history plus current message for API call
            messages = valid_messages + [current_message]
            
            # Get tool definitions
            tools = self.tools_manager.define_tools()
            
            # Make API call (non-streaming)
            self.output.print_section("CALLING CLAUDE")
            self.output.print("Waiting for Claude's response (game continues running)...")
            
            try:
                # Make the API call while the emulator continues running in the background
                response = self.client.messages.create(
                    model=self.model,
                    system=system_prompt,
                    messages=messages,
                    max_tokens=1000,
                    temperature=self.temperature,
                    tools=tools
                )
                
                # Store message for history
                self.message_history.append(current_message)
                
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
                    
                    # Execute the tool through the emulator thread
                    self.output.print_section("EXECUTING ACTION")
                    self.output.print(f"Executing {tool_name} with parameters: {tool_params}")
                    
                    # Map tool actions to emulator thread actions
                    if tool_name == "press_button":
                        button = tool_params.get("button")
                        hold_frames = tool_params.get("hold_frames", 10)
                        self.emulator_thread.press_button(button, hold_frames)
                        action = button
                        result_type = f"button_{button}"
                    elif tool_name == "wait_frames":
                        num_frames = tool_params.get("num_frames", 30)
                        self.emulator_thread.wait_frames(num_frames)
                        action = "wait"
                        result_type = "wait"
                    else:
                        # For other tools, execute them directly
                        result = self.tools_manager.execute_tool(tool_name, tool_params)
                        action = tool_name
                        
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
                        
                        return action, result
                    
                    # Wait for the action to complete and get the result from the queue
                    timeout = 5.0  # Maximum time to wait for the action
                    start_time = time.time()
                    result = None
                    
                    while time.time() - start_time < timeout:
                        try:
                            if not self.emulator_thread.result_queue.empty():
                                action_result = self.emulator_thread.result_queue.get_nowait()
                                if action_result["action"] == result_type:
                                    result = action_result["result"]
                                    break
                        except queue.Empty:
                            pass
                        time.sleep(0.1)
                    
                    if result is None:
                        result = f"Action {action} submitted but no result received within timeout"
                    
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
                    
                    return action, result
                else:
                    # No tool was used, fall back to wait
                    self.output.print_section("NO TOOL USED")
                    self.output.print("Claude didn't use a tool. Falling back to wait...")
                    self.emulator_thread.wait_frames(30)
                    action = "wait"
                    
                    # Wait for the result from the queue
                    timeout = 3.0
                    start_time = time.time()
                    result = None
                    
                    while time.time() - start_time < timeout:
                        try:
                            if not self.emulator_thread.result_queue.empty():
                                action_result = self.emulator_thread.result_queue.get_nowait()
                                if action_result["action"] == "wait":
                                    result = action_result["result"]
                                    break
                        except queue.Empty:
                            pass
                        time.sleep(0.1)
                    
                    if result is None:
                        result = "Wait action submitted but no result received within timeout"
                    
                    self.kb.add_action("wait", "No tool used")
                    return "wait", result
                    
            except Exception as e:
                self.output.print_section("API ERROR")
                self.output.print(f"Error: {str(e)}")
                traceback_str = traceback.format_exc()
                self.output.print(traceback_str)
                self.last_error = str(e)
                
                # Fallback to wait
                self.emulator_thread.wait_frames(30)
                action = "wait"
                
                # Wait briefly for the result
                time.sleep(1.0)
                result = "Error fallback: " + str(e)
                if not self.emulator_thread.result_queue.empty():
                    try:
                        action_result = self.emulator_thread.result_queue.get_nowait()
                        if action_result["action"] == "wait":
                            result = action_result["result"]
                    except:
                        pass
                
                self.kb.add_action("wait", result)
                return "wait", result
                
        except Exception as e:
            self.output.print_section("GENERAL ERROR")
            self.output.print(f"Error: {str(e)}")
            traceback_str = traceback.format_exc()
            self.output.print(traceback_str)
            self.last_error = str(e)
            
            # Fallback to wait
            self.emulator_thread.wait_frames(30)
            time.sleep(1.0)
            result = "Error fallback: " + str(e)
            self.kb.add_action("wait", result)
            return "wait", result
    
    def check_for_summarization(self):
        """Check if we need to summarize and reset conversation."""
        self.turn_count += 1
        
        if self.turn_count >= self.max_turns_before_summary:
            try:
                self.output.print_section("GENERATING SUMMARY", f"After {self.turn_count} turns")
                
                # Get game state for additional context
                game_state = self.emulator_thread.get_game_state()
                state_text = self.emulator.format_game_state(game_state)
                
                # Request a summary from Claude
                summary_response = self.client.messages.create(
                    model=self.model,
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
                    temperature=0.7
                )
                
                # Extract summary text
                summary_text = ""
                for content in summary_response.content:
                    if content.type == "text":
                        summary_text = content.text
                        break
                
                # Update last summary
                self.last_summary = summary_text
                
                # Store in knowledge base
                self.kb.update("player_progress", "last_summary", summary_text)
                
                # Reset conversation
                self.turn_count = 0
                self.message_history = [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Previous progress summary: {self.last_summary}"
                            }
                        ]
                    }
                ]
                
                self.output.print_section("SUMMARY", summary_text)
                self.output.print("Context reset with summary")
                
            except Exception as e:
                self.output.print_section("SUMMARIZATION ERROR", str(e))
                # Continue without resetting if summarization fails
                self.turn_count = self.max_turns_before_summary - 5  # Try again in 5 turns
    
    def run(self, num_steps=None):
        """Run the agent for specified steps."""
        step_count = 0
        
        try:
            self.output.print_section("STARTING POKÉMON AGENT", f"Model: {self.model}, Temperature: {self.temperature}")
            
            while num_steps is None or step_count < num_steps:
                # Get action from Claude
                action, result = self.get_action()
                
                # Increment counter
                step_count += 1
                
                # Print progress
                self.output.print_section(f"STEP {step_count} SUMMARY", f"Action: {action}\nResult: {result}")
                
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
                    self.emulator_thread.save_state(f"step_{step_count}.state")
                    self.output.print(f"Saved game state at step {step_count}")
                
                # Check for summarization
                self.check_for_summarization()
                
                # Small delay between decision cycles (not between game frames)
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.output.print_section("INTERRUPTED", "Stopped by user")
        except Exception as e:
            self.output.print_section("ERROR", f"Error running agent: {e}")
            traceback_str = traceback.format_exc()
            self.output.print(traceback_str)
        finally:
            # Save final state
            self.emulator_thread.save_state("final.state")
            
            # Save action stats
            with open("action_stats.json", 'w') as f:
                json.dump(self.tools_manager.get_stats(), f, indent=2)
            
            self.output.print_section("RUN COMPLETE", f"Total steps: {step_count}")
            
            # Clean up
            self.close()
    
    def close(self):
        """Clean up resources."""
        if self.emulator_thread_started:
            self.emulator_thread.stop()
        self.emulator.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a Pokémon AI agent using Claude (Threaded Version)')
    parser.add_argument('rom_path', help='Path to the Pokémon ROM file')
    parser.add_argument('--api-key', help='API key (will use ANTHROPIC_API_KEY env var if not provided)')
    parser.add_argument('--model', default='claude-3-7-sonnet-20250219', help='Model to use')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no window)')
    parser.add_argument('--steps', type=int, help='Number of steps to run (infinite if not specified)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for the LLM')
    parser.add_argument('--speed', type=int, default=1, help='Game speed multiplier')
    parser.add_argument('--no-sound', action='store_true', help='Disable game sound')
    parser.add_argument('--log-to-file', action='store_true', help='Log output to file')
    parser.add_argument('--log-file', help='Path to log file (default: pokemon_agent.log)')
    parser.add_argument('--emulator-tick-rate', type=int, default=60, help='Emulator frames per second')
    
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
        emulator_tick_rate=args.emulator_tick_rate
    )
    
    try:
        agent.start()
        agent.run(num_steps=args.steps)
    finally:
        agent.close()