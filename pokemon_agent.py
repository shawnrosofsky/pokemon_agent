import os
import time
import json
import base64
from io import BytesIO
from PIL import Image
import anthropic
from typing import Dict, List, Any, Optional
import traceback

from pyboy import PyBoy
from pyboy.utils import WindowEvent

# Configure output settings
class OutputManager:
    """Manages output of Claude's thoughts and agent actions."""
    
    def __init__(self, output_to_file=False, log_file=None):
        self.output_to_file = output_to_file
        self.log_file = log_file or "pokemon_agent.log"
        
        # Create log file if needed
        if self.output_to_file:
            with open(self.log_file, 'w') as f:
                f.write(f"=== Pokémon Agent Log - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    def print(self, text, end="\n", flush=True):
        """Print text to console and optionally to file."""
        print(text, end=end, flush=flush)
        
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

class KnowledgeBase:
    """Simple knowledge base to store game information and history."""
    
    def __init__(self, save_path="knowledge_base.json"):
        self.save_path = save_path
        self.data = {
            "game_state": {},
            "player_progress": {},
            "strategies": {},
            "map_data": {},
            "action_history": []
        }
        self.load()
    
    def load(self):
        """Load knowledge base from disk if it exists."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"Error loading knowledge base: {e}")
    
    def save(self):
        """Save knowledge base to disk."""
        with open(self.save_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def update(self, section, key, value):
        """Update a section of the knowledge base."""
        if section not in self.data:
            self.data[section] = {}
        self.data[section][key] = value
        self.save()
    
    def get(self, section, key=None):
        """Get data from the knowledge base."""
        if section not in self.data:
            return None
        if key is None:
            return self.data[section]
        return self.data[section].get(key)
    
    def add_action(self, action, result):
        """Add an action to the history."""
        self.data["action_history"].append({
            "action": action,
            "result": result,
            "timestamp": time.time()
        })
        # Limit history size
        if len(self.data["action_history"]) > 100:
            self.data["action_history"] = self.data["action_history"][-100:]
        self.save()
    
    def get_recent_actions(self, count=5):
        """Get recent actions as formatted text."""
        recent = self.data["action_history"][-count:] if self.data["action_history"] else []
        return "\n".join([f"Action: {a['action']}, Result: {a['result']}" for a in recent])


class GameEmulator:
    """Interface to the Pokemon game emulator (PyBoy)."""
    
    # Game Boy button mappings
    BUTTONS = {
        'up': WindowEvent.PRESS_ARROW_UP,
        'down': WindowEvent.PRESS_ARROW_DOWN,
        'left': WindowEvent.PRESS_ARROW_LEFT,
        'right': WindowEvent.PRESS_ARROW_RIGHT,
        'a': WindowEvent.PRESS_BUTTON_A,
        'b': WindowEvent.PRESS_BUTTON_B,
        'start': WindowEvent.PRESS_BUTTON_START,
        'select': WindowEvent.PRESS_BUTTON_SELECT
    }
    
    RELEASE_BUTTONS = {
        'up': WindowEvent.RELEASE_ARROW_UP,
        'down': WindowEvent.RELEASE_ARROW_DOWN,
        'left': WindowEvent.RELEASE_ARROW_LEFT,
        'right': WindowEvent.RELEASE_ARROW_RIGHT,
        'a': WindowEvent.RELEASE_BUTTON_A,
        'b': WindowEvent.RELEASE_BUTTON_B,
        'start': WindowEvent.RELEASE_BUTTON_START,
        'select': WindowEvent.RELEASE_BUTTON_SELECT
    }
    
    # Memory addresses for Pokémon Red/Blue
    MEMORY_MAP = {
        'player_x': 0xD362,
        'player_y': 0xD361,
        'map_id': 0xD35E,
        'player_direction': 0xD367,
        'battle_type': 0xD057,
        'pokemon_count': 0xD163,
        'first_pokemon_hp': 0xD16C,
        'first_pokemon_max_hp': 0xD18D,
        'text_progress': 0xC6AC,
        'badges': 0xD356,
        'money': 0xD347,
    }
    
    def __init__(self, rom_path, headless=False, speed=1, sound=False):
        """Initialize the game emulator."""
        self.rom_path = rom_path
        window = "null" if headless else "SDL2"
        
        # Initialize PyBoy with sound options
        self.pyboy = PyBoy(
            rom_path, 
            window=window,
            sound=sound, 
            sound_emulated=sound
        )
        
        self.pyboy.set_emulation_speed(speed)
        
        # Get screen accessor
        self.screen = self.pyboy.screen
        
        self.frame_count = 0
        self.save_dir = "saves"
        os.makedirs(self.save_dir, exist_ok=True)
    
    def start_game(self, skip_intro=True):
        """Start the game and optionally skip intro."""
        # Boot the game
        for _ in range(100):
            self.pyboy.tick()
            self.frame_count += 1
        
        if skip_intro:
            # Skip intro by pressing start a few times
            for _ in range(3):
                self.press_button('start', hold_frames=5)
                time.sleep(0.1)
        
        # Wait for game to settle
        for _ in range(60):
            self.pyboy.tick()
            self.frame_count += 1
    
    def press_button(self, button, hold_frames=10):
        """Press a button for specified frames."""
        if button not in self.BUTTONS:
            return f"Error: Unknown button '{button}'"
        
        # Press button
        self.pyboy.send_input(self.BUTTONS[button])
        
        # Hold for frames
        for _ in range(hold_frames):
            self.pyboy.tick()
            self.frame_count += 1
        
        # Release button
        self.pyboy.send_input(self.RELEASE_BUTTONS[button])
        self.pyboy.tick()
        self.frame_count += 1
        
        return f"Pressed {button} for {hold_frames} frames"
    
    def wait_frames(self, num_frames=30):
        """Wait for specified frames."""
        for _ in range(num_frames):
            self.pyboy.tick()
            self.frame_count += 1
        
        return f"Waited for {num_frames} frames"
    
    def get_memory_value(self, address):
        """Get value from memory address."""
        if isinstance(address, str):
            if address in self.MEMORY_MAP:
                address = self.MEMORY_MAP[address]
            else:
                return None
        
        return self.pyboy.memory[address]
    
    def get_2byte_value(self, address):
        """Get 2-byte value (little endian)."""
        if isinstance(address, str):
            if address in self.MEMORY_MAP:
                address = self.MEMORY_MAP[address]
            else:
                return None
        
        low_byte = self.get_memory_value(address)
        high_byte = self.get_memory_value(address + 1)
        return (high_byte << 8) | low_byte
    
    def get_screen_base64(self):
        """Get current screen as base64 string."""
        screen_array = self.screen.ndarray
        pil_image = Image.fromarray(screen_array)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def get_game_state(self):
        """Get comprehensive game state."""
        # Basic info
        x = self.get_memory_value('player_x')
        y = self.get_memory_value('player_y')
        map_id = self.get_memory_value('map_id')
        direction = self.get_memory_value('player_direction')
        in_battle = self.get_memory_value('battle_type') != 0
        text_active = self.get_memory_value('text_progress') != 0
        
        # Pokémon info
        pokemon_count = self.get_memory_value('pokemon_count')
        first_hp = self.get_2byte_value('first_pokemon_hp') if pokemon_count > 0 else 0
        first_max_hp = self.get_2byte_value('first_pokemon_max_hp') if pokemon_count > 0 else 0
        
        # Progress indicators
        badges = self.get_memory_value('badges')
        badge_count = bin(badges).count('1') if badges is not None else 0
        
        return {
            'position': (x, y),
            'map_id': map_id,
            'direction': direction,
            'in_battle': in_battle,
            'text_active': text_active,
            'pokemon_count': pokemon_count,
            'first_pokemon_hp': first_hp,
            'first_pokemon_max_hp': first_max_hp,
            'badges': badge_count,
            'frame': self.frame_count
        }
    
    def format_game_state(self, state):
        """Format game state as readable text."""
        text = "CURRENT GAME STATE:\n"
        text += f"Position: ({state['position'][0]}, {state['position'][1]}) on Map {state['map_id']}\n"
        text += f"Direction: {state['direction']}, Badges: {state['badges']}\n"
        
        if state['in_battle']:
            text += "Currently in battle\n"
        
        text += f"Pokémon Count: {state['pokemon_count']}\n"
        if state['pokemon_count'] > 0:
            text += f"First Pokémon HP: {state['first_pokemon_hp']}/{state['first_pokemon_max_hp']}\n"
        
        if state['text_active']:
            text += "Text is currently being displayed\n"
        
        return text
    
    def save_state(self, filename=None):
        """Save game state to file."""
        if filename is None:
            filename = f"{self.save_dir}/state_frame_{self.frame_count}.state"
        else:
            filename = f"{self.save_dir}/{filename}"
        
        self.pyboy.save_state(filename)
        return filename
    
    def close(self):
        """Clean up resources."""
        self.pyboy.stop()


class PokemonAgent:
    """LLM Agent for playing Pokémon using Anthropic Claude."""
    
    def __init__(self, rom_path, model_name="claude-3-7-sonnet-20250219", 
                 api_key=None, temperature=0.7, headless=False, speed=1,
                 sound=True, output_to_file=False, log_file=None):
        """Initialize the agent."""
        # Setup API
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set ANTHROPIC_API_KEY env var or pass directly.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model_name
        self.temperature = temperature
        
        # Setup game and knowledge
        self.emulator = GameEmulator(rom_path, headless=headless, speed=speed, sound=sound)
        self.kb = KnowledgeBase()
        
        # Setup output manager
        self.output = OutputManager(output_to_file=output_to_file, log_file=log_file)
        
        # Conversation management
        self.message_history = []
        self.turn_count = 0
        self.max_turns_before_summary = 10
        self.last_summary = ""
        
        # Stats
        self.action_stats = {}
        self.last_error = None
        
        # Define tools
        self.tools = self._define_tools()
    
    def _define_tools(self):
        """Define the tools Claude can use."""
        return [
            {
                "name": "press_button",
                "description": "Press a button on the Game Boy",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "button": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right", "a", "b", "start", "select"],
                            "description": "The button to press"
                        },
                        "hold_frames": {
                            "type": "integer",
                            "description": "Number of frames to hold the button",
                            "default": 10
                        }
                    },
                    "required": ["button"]
                }
            },
            {
                "name": "wait_frames",
                "description": "Wait for a specified number of frames without taking any action",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "num_frames": {
                            "type": "integer",
                            "description": "Number of frames to wait",
                            "default": 30
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "update_knowledge",
                "description": "Update the knowledge base with new information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "enum": ["game_state", "player_progress", "strategies", "map_data"],
                            "description": "Section to update"
                        },
                        "key": {
                            "type": "string",
                            "description": "Key within the section"
                        },
                        "value": {
                            "type": "string",
                            "description": "Value to store"
                        }
                    },
                    "required": ["section", "key", "value"]
                }
            }
        ]
    
    def start(self):
        """Start the game."""
        self.emulator.start_game(skip_intro=True)
    
    def execute_tool(self, tool_name, params):
        """Execute a tool based on name and parameters."""
        try:
            if tool_name == "press_button":
                button = params.get("button", "")
                hold_frames = params.get("hold_frames", 10)
                
                # Track action stats
                if button in self.action_stats:
                    self.action_stats[button] += 1
                else:
                    self.action_stats[button] = 1
                
                return self.emulator.press_button(button, hold_frames)
                
            elif tool_name == "wait_frames":
                num_frames = params.get("num_frames", 30)
                
                # Track stats
                if "wait" in self.action_stats:
                    self.action_stats["wait"] += 1
                else:
                    self.action_stats["wait"] = 1
                
                return self.emulator.wait_frames(num_frames)
                
            elif tool_name == "update_knowledge":
                section = params.get("section", "")
                key = params.get("key", "")
                value = params.get("value", "")
                
                # Track stats
                if "update_kb" in self.action_stats:
                    self.action_stats["update_kb"] += 1
                else:
                    self.action_stats["update_kb"] = 1
                
                self.kb.update(section, key, value)
                return f"Updated knowledge base: {section}.{key} = {value}"
                
            else:
                error_msg = f"Unknown tool: {tool_name}. Using wait fallback."
                self.last_error = error_msg
                return self.emulator.wait_frames(30)
                
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            self.last_error = error_msg
            print(error_msg)
            return self.emulator.wait_frames(30)
    
    def get_action(self):
        """
        Get current game state, analyze with Claude, and execute the action in a single step.
        This function combines analysis and action execution for efficiency.
        Uses streaming to show Claude's thought process in real-time.
        """
        try:
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

            # Clean and validate message history
            valid_messages = []
            if self.message_history:
                # Check for proper message sequence
                valid = True
                tool_use_positions = []
                
                # Find all tool_use positions
                for i, msg in enumerate(self.message_history):
                    if msg.get("role") == "assistant" and any(
                        isinstance(c, dict) and c.get("type") == "tool_use" 
                        for c in msg.get("content", []) if isinstance(c, dict)):
                        tool_use_positions.append(i)
                
                # Verify each tool_use has a corresponding tool_result
                for pos in tool_use_positions:
                    if pos + 1 >= len(self.message_history) or self.message_history[pos + 1].get("role") != "user":
                        valid = False
                        break
                    user_msg = self.message_history[pos + 1]
                    has_tool_result = any(
                        isinstance(c, dict) and c.get("type") == "tool_result" 
                        for c in user_msg.get("content", []) if isinstance(c, dict))
                    if not has_tool_result:
                        valid = False
                        break
                
                if valid:
                    # Use last few exchanges if valid
                    valid_messages = self.message_history[-4:] if len(self.message_history) > 4 else self.message_history
                else:
                    # Reset history if invalid
                    self.output.print("Invalid message history pattern detected, resetting conversation")
                    self.message_history = []
                    
                    # Insert summary if available
                    if self.last_summary:
                        valid_messages = [
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
            
            # Add current message
            messages = valid_messages + [
                {
                    "role": "user",
                    "content": message_content
                }
            ]
            
            # Call Claude with streaming enabled
            self.output.print_section("CLAUDE'S THOUGHT PROCESS")
            
            stream = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=messages,
                max_tokens=1000,
                temperature=self.temperature,
                tools=self.tools,
                stream=True
            )
            
            # Process the stream
            accumulated_content = []
            tool_use_blocks = []
            current_text = ""
            current_block_type = None
            current_tool_use = None
            
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text":
                        # For text blocks
                        current_text += event.delta.text
                        self.output.print(event.delta.text, end="", flush=True)
                    elif event.delta.type == "tool_use":
                        # For tool use blocks
                        if event.delta.name:
                            self.output.print(f"\n[Using tool: {event.delta.name}]", end="", flush=True)
                            if current_tool_use is None:
                                current_tool_use = {"type": "tool_use", "name": event.delta.name}
                            else:
                                current_tool_use["name"] = event.delta.name
                        if event.delta.input:
                            self.output.print(f" with parameters: {event.delta.input}", end="", flush=True)
                            if current_tool_use is None:
                                current_tool_use = {"type": "tool_use", "input": event.delta.input}
                            else:
                                current_tool_use["input"] = event.delta.input
                        if event.delta.id:
                            if current_tool_use is None:
                                current_tool_use = {"type": "tool_use", "id": event.delta.id}
                            else:
                                current_tool_use["id"] = event.delta.id
                
                elif event.type == "message_delta":
                    # Message complete
                    pass
                
                elif event.type == "content_block_start":
                    # Start of a new content block
                    current_block_type = event.content_block.type
                    if current_block_type == "tool_use":
                        self.output.print(f"\n[Starting to use a tool]", end="", flush=True)
                        current_tool_use = {"type": "tool_use"}
                        # Only add if it has an ID
                        if hasattr(event.content_block, 'id') and event.content_block.id:
                            current_tool_use["id"] = event.content_block.id
                
                elif event.type == "content_block_stop":
                    # End of a content block
                    if current_block_type == "text" and current_text:
                        accumulated_content.append({
                            "type": "text",
                            "text": current_text
                        })
                        current_text = ""
                    elif current_block_type == "tool_use" and current_tool_use:
                        accumulated_content.append(current_tool_use)
                        current_tool_use = None
                    
                    current_block_type = None
            
            self.output.print("")  # End the streaming output with a newline
            
            # Add to conversation history
            self.message_history.append({
                "role": "user",
                "content": message_content
            })
            self.message_history.append({
                "role": "assistant", 
                "content": accumulated_content
            })
            
            # Process any tool calls
            tool_call = None
            for content in accumulated_content:
                if content.get("type") == "tool_use":
                    tool_call = content
                    break
            
            if tool_call:
                # Execute the tool
                tool_name = tool_call.get("name")
                tool_params = tool_call.get("input", {})
                tool_id = tool_call.get("id")
                
                self.output.print_section("ACTION EXECUTION")
                self.output.print(f"Executing {tool_name} with params: {tool_params}")
                result = self.execute_tool(tool_name, tool_params)
                self.output.print(f"Result: {result}")
                
                # Track action taken
                action = tool_name
                if tool_name == "press_button" and "button" in tool_params:
                    action = tool_params["button"]
                
                # Add to knowledge base
                self.kb.add_action(action, result)
                
                # Add tool result to conversation history
                user_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result
                        }
                    ]
                }
                self.message_history.append(user_msg)
                
                return action, result
            else:
                # No tool call, default to waiting
                self.output.print_section("ACTION EXECUTION")
                self.output.print("No tool call detected, using wait fallback")
                result = self.emulator.wait_frames(30)
                self.kb.add_action("wait", result)
                
                # Add a standard response to keep conversation flowing
                self.message_history.append({
                    "role": "user",
                    "content": [{"type": "text", "text": "I couldn't determine a specific action to take. Please help."}]
                })
                
                return "wait", result
            
        except Exception as e:
            error_msg = f"Error getting action: {e}"
            self.output.print_section("ERROR")
            self.output.print(error_msg)
            traceback_str = traceback.format_exc()
            self.output.print(traceback_str)
            self.last_error = str(e)
            
            # Fallback to wait
            try:
                result = self.emulator.wait_frames(30)
                self.kb.add_action("wait", "Error fallback: " + str(e))
                return "wait", result
            except:
                return "error", "Failed to execute wait fallback"
    
    def check_for_summarization(self):
        """Check if we need to summarize and reset conversation."""
        self.turn_count += 1
        
        if self.turn_count >= self.max_turns_before_summary:
            try:
                self.output.print_section("GENERATING SUMMARY", f"After {self.turn_count} turns")
                
                # Get game state for additional context
                game_state = self.emulator.get_game_state()
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
                # Get action in a single step
                action, result = self.get_action()
                
                # Increment counter
                step_count += 1
                
                # Print progress
                self.output.print_section(f"STEP {step_count} SUMMARY", f"Action: {action}\nResult: {result}")
                
                # Show stats every 10 steps
                if step_count % 10 == 0:
                    stats_text = "Action distribution:\n"
                    total = sum(self.action_stats.values()) or 1
                    for act, count in sorted(self.action_stats.items(), key=lambda x: x[1], reverse=True):
                        stats_text += f"  {act}: {count} ({count/total*100:.1f}%)\n"
                    self.output.print_section(f"STATISTICS AFTER {step_count} STEPS", stats_text)
                
                # Save state periodically
                if step_count % 50 == 0:
                    save_path = self.emulator.save_state(f"step_{step_count}.state")
                    self.output.print(f"Saved game state to {save_path}")
                
                # Check for summarization
                self.check_for_summarization()
                
                # Small delay
                time.sleep(0.01)
                
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
                json.dump(self.action_stats, f, indent=2)
            
            self.output.print_section("RUN COMPLETE", f"Total steps: {step_count}")
            
            # Clean up
            self.emulator.close()
    
    def close(self):
        """Clean up resources."""
        self.emulator.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a Pokémon AI agent using Claude')
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
        log_file=args.log_file
    )
    
    try:
        agent.start()
        agent.run(num_steps=args.steps)
    finally:
        agent.close()