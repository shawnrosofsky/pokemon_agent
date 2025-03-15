import os
import time
import json
import base64
import traceback
from typing import Dict, List, Any, Optional, Union
import uuid
from io import BytesIO
from PIL import Image
import anthropic
from anthropic.types import MessageParam, ToolUseBlock

# Import PyBoy for game interaction
from pyboy import PyBoy
from pyboy.utils import WindowEvent

class KnowledgeBase:
    """
    Knowledge base for storing and retrieving game-related information.
    Maintains game state, player progress, and strategic information.
    """
    
    def __init__(self, save_path="knowledge_base.json"):
        self.save_path = save_path
        self.data = {
            "game_state": {},
            "player_progress": {},
            "strategies": {},
            "map_data": {},
            "previous_states": [],
            "action_history": []
        }
        self.load()
    
    def save(self):
        """Save the knowledge base to disk."""
        with open(self.save_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def load(self):
        """Load the knowledge base from disk if it exists."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"Error loading knowledge base: {e}")
    
    def update(self, section, key, value):
        """Update a specific section and key in the knowledge base."""
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
        """Add an action and its result to the history."""
        self.data["action_history"].append({
            "action": action,
            "result": result,
            "timestamp": time.time()
        })
        # Limit history size
        if len(self.data["action_history"]) > 100:
            self.data["action_history"] = self.data["action_history"][-100:]
        self.save()
    
    def add_game_state(self, state):
        """Add a game state snapshot to history."""
        self.data["previous_states"].append({
            "state": state,
            "timestamp": time.time()
        })
        # Limit history size
        if len(self.data["previous_states"]) > 10:
            self.data["previous_states"] = self.data["previous_states"][-10:]
        self.save()
    
    def summarize_recent_history(self, max_items=5):
        """Summarize recent action history for context management."""
        recent_actions = self.data["action_history"][-max_items:] if self.data["action_history"] else []
        summary = []
        
        for item in recent_actions:
            summary.append(f"Action: {item['action']}, Result: {item['result']}")
        
        return "\n".join(summary)


class GameEmulator:
    """
    Interface for the Pokémon game emulator (PyBoy).
    Handles game controls and state observation.
    """
    
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
        # Player and Map
        'player_x': 0xD362,            # X position on map
        'player_y': 0xD361,            # Y position on map
        'map_id': 0xD35E,              # Current map ID
        'player_direction': 0xD367,    # Direction (0: down, 4: up, 8: left, 0xC: right)
        
        # Battle system
        'battle_type': 0xD057,         # Non-zero when in battle
        'enemy_pokemon_hp': 0xCFE7,    # Enemy Pokémon HP (2 bytes, little endian)
        'enemy_pokemon_level': 0xCFE8, # Enemy Pokémon level
        'enemy_pokemon_species': 0xCFDE, # Enemy Pokémon species ID
        
        # Player Pokémon
        'pokemon_count': 0xD163,       # Number of Pokémon in party
        'first_pokemon_hp': 0xD16C,    # First Pokémon's current HP (2 bytes)
        'first_pokemon_max_hp': 0xD18D, # First Pokémon's max HP (2 bytes)
        'first_pokemon_level': 0xD18C, # First Pokémon's level
        
        # Menu and text
        'text_progress': 0xC6AC,       # Text box progress (non-zero when text is displaying)
        
        # Badges and progress
        'badges': 0xD356,              # Badges obtained (each bit = one badge)
        'money': 0xD347,               # Money (3 bytes, BCD format)
    }
    
    def __init__(self, rom_path, headless=False, speed=1, sound=False):
        """Initialize the game emulator with the ROM."""
        self.rom_path = rom_path
        window = "null" if headless else "SDL2"
        self.pyboy = PyBoy(rom_path, window=window, sound=sound, sound_emulated=sound)
        self.pyboy.set_emulation_speed(speed)
        
        # Setup screen manager
        self.screen_manager = self.pyboy.screen
        
        # Game state tracking
        self.frame_count = 0
        self.save_dir = "saves"
        os.makedirs(self.save_dir, exist_ok=True)
    
    def start_game(self, skip_intro=True):
        """Start the game and optionally skip intro sequence."""
        # Let the game boot
        for _ in range(100):
            self.pyboy.tick()
            self.frame_count += 1
        
        if skip_intro:
            # Skip intro sequence
            for _ in range(3):
                print(self.press_button('start', hold_frames=5))
                time.sleep(0.1)
        
        # Wait for game to settle
        for _ in range(60):
            self.pyboy.tick()
            self.frame_count += 1
    
    def press_button(self, button, hold_frames=10):
        """Press a button for the specified number of frames."""
        if button not in self.BUTTONS:
            print(f"Warning: Unknown button '{button}'")
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
    
    def get_memory_value(self, address):
        """Get value from a memory address."""
        if isinstance(address, str):
            if address in self.MEMORY_MAP:
                address = self.MEMORY_MAP[address]
            else:
                print(f"Warning: Unknown memory address alias '{address}'")
                return None
        
        return self.pyboy.memory[address]
    
    def get_2byte_value(self, address):
        """Get a 2-byte value (little endian)."""
        if isinstance(address, str):
            if address in self.MEMORY_MAP:
                address = self.MEMORY_MAP[address]
            else:
                return None
        
        low_byte = self.get_memory_value(address)
        high_byte = self.get_memory_value(address + 1)
        return (high_byte << 8) | low_byte
    
    def get_screen(self):
        """Get current screen as numpy array."""
        return self.screen_manager.ndarray
    
    def get_screen_as_pil(self):
        """Get current screen as PIL image."""
        screen_array = self.get_screen()
        return Image.fromarray(screen_array)
    
    def get_screen_base64(self):
        """Get current screen as base64 string."""
        pil_image = self.get_screen_as_pil()
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def get_game_state(self):
        """Get comprehensive game state."""
        # Player position and map
        player_x = self.get_memory_value('player_x')
        player_y = self.get_memory_value('player_y')
        current_map = self.get_memory_value('map_id')
        player_direction = self.get_memory_value('player_direction')
        
        # Battle information
        in_battle = self.get_memory_value('battle_type') != 0
        
        # Pokémon party
        pokemon_count = self.get_memory_value('pokemon_count')
        
        # First Pokémon info
        first_pokemon = {}
        if pokemon_count > 0:
            first_pokemon = {
                'hp': self.get_2byte_value('first_pokemon_hp'),
                'max_hp': self.get_2byte_value('first_pokemon_max_hp'),
                'level': self.get_memory_value('first_pokemon_level')
            }
        
        # Enemy Pokémon if in battle
        enemy_pokemon = {}
        if in_battle:
            enemy_pokemon = {
                'hp': self.get_2byte_value('enemy_pokemon_hp'),
                'level': self.get_memory_value('enemy_pokemon_level'),
                'species': self.get_memory_value('enemy_pokemon_species')
            }
        
        # Text status
        text_active = self.get_memory_value('text_progress') != 0
        
        # Progress indicators
        badges = self.get_memory_value('badges')
        badge_count = bin(badges).count('1') if badges is not None else 0
        
        # Format money from BCD
        money_bytes = [
            self.get_memory_value(self.MEMORY_MAP['money']),
            self.get_memory_value(self.MEMORY_MAP['money'] + 1),
            self.get_memory_value(self.MEMORY_MAP['money'] + 2)
        ]
        
        money = 0
        for byte in money_bytes:
            if byte is not None:
                money = money * 100 + ((byte >> 4) * 10 + (byte & 0x0F))
        
        return {
            'frame': self.frame_count,
            'player': {
                'position': (player_x, player_y),
                'map': current_map,
                'direction': player_direction,
                'badges': badge_count,
                'money': money
            },
            'battle': {
                'active': in_battle,
                'enemy_pokemon': enemy_pokemon if in_battle else None
            },
            'party': {
                'count': pokemon_count,
                'first_pokemon': first_pokemon if pokemon_count > 0 else None
            },
            'ui': {
                'text_active': text_active
            }
        }
    
    def format_game_state(self, state):
        """Format game state as readable text."""
        text = "CURRENT GAME STATE:\n"
        
        # Player info
        player = state['player']
        text += f"Player Position: ({player['position'][0]}, {player['position'][1]}) on Map {player['map']}\n"
        text += f"Direction: {player['direction']}, Money: ${player['money']}, Badges: {player['badges']}\n"
        
        # Battle info
        if state['battle']['active']:
            enemy = state['battle']['enemy_pokemon']
            text += "\nIN BATTLE:\n"
            text += f"Enemy Pokémon: Species #{enemy['species']}, Level {enemy['level']}, HP: {enemy['hp']}\n"
        
        # Party info
        party = state['party']
        text += f"\nPokémon Party: {party['count']} Pokémon\n"
        if party['first_pokemon']:
            first = party['first_pokemon']
            text += f"First Pokémon: Level {first['level']}, HP: {first['hp']}/{first['max_hp']}\n"
        
        # UI state
        ui = state['ui']
        if ui['text_active']:
            text += "\nText is currently being displayed.\n"
        
        return text
    
    def save_state(self, filename=None):
        """Save game state to file."""
        if filename is None:
            filename = f"{self.save_dir}/state_frame_{self.frame_count}.state"
        else:
            filename = f"{self.save_dir}/{filename}"
        
        with open(filename, "wb") as f:
            self.pyboy.save_state(f)

        return filename
    
    def load_state(self, filename):
        """Load game state from file."""
        filepath = f"{self.save_dir}/{filename}" if not filename.startswith(self.save_dir) else filename
        
        try:
            self.pyboy.load_state(filepath)
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False
    
    def wait_frames(self, num_frames):
        """Wait for specified number of frames."""
        for _ in range(num_frames):
            self.pyboy.tick()
            self.frame_count += 1
        
        return f"Waited for {num_frames} frames"
    
    def close(self):
        """Clean up resources."""
        self.pyboy.stop()


class PokemonAnthropicAgent:
    """
    Pokémon agent implementation using direct Anthropic API.
    """
    
    def __init__(self, rom_path, model_name="claude-3-7-sonnet-20250219", 
                 api_key=None, temperature=1.0,
                 max_tokens=10000,
                 thinking=False, thinking_budget=4000,
                 headless=False, speed=1, sound=False):
        """Initialize the agent."""
        # Set up API key
        self.api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set ANTHROPIC_API_KEY env var or pass directly.")
        
        # Set up model
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Initialize game emulator
        self.emulator = GameEmulator(rom_path, headless=headless, speed=speed, sound=sound)
        
        # Initialize knowledge base
        self.knowledge_base = KnowledgeBase()
        
        # Config for message handling
        self.conversation_turn_count = 0
        self.last_summary = ""
        self.max_turns_before_summary = 10
        self.message_history = []
        
        # Define the tools
        self.tools = self._define_tools()
        
        # Statistics
        self.action_stats = {}
        self.start_time = None
        self.last_error = None
    
    def _define_tools(self):
        """Define tools for Claude to use."""
        return [
            {
                "name": "press_button",
                "description": "Press a button on the Game Boy for a specified number of frames",
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
                "description": "Wait for a specified number of frames without taking any action (safe fallback)",
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
                "name": "navigate_to",
                "description": "Navigate to a specific location in the game",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "destination": {
                            "type": "string",
                            "description": "The destination location to navigate to"
                        }
                    },
                    "required": ["destination"]
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
                            "description": "Section to update (game_state, player_progress, strategies, map_data)"
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
    
    def _execute_tool(self, tool_name, tool_params):
        """Execute a tool based on name and parameters."""
        try:
            if tool_name == "press_button":
                button = tool_params.get("button", "")
                hold_frames = tool_params.get("hold_frames", 10)
                
                # Check if button is valid
                if button not in self.emulator.BUTTONS:
                    error_msg = f"Invalid button: '{button}'. Using wait fallback."
                    self.last_error = error_msg
                    return self.emulator.wait_frames(30)
                
                # Press the button
                result = self.emulator.press_button(button, hold_frames)
                
                # Update action stats
                if button in self.action_stats:
                    self.action_stats[button] += 1
                else:
                    self.action_stats[button] = 1
                
                return result
                
            elif tool_name == "wait_frames":
                num_frames = tool_params.get("num_frames", 30)
                result = self.emulator.wait_frames(num_frames)
                
                # Update action stats
                if "wait" in self.action_stats:
                    self.action_stats["wait"] += 1
                else:
                    self.action_stats["wait"] = 1
                
                return result
                
            elif tool_name == "navigate_to":
                destination = tool_params.get("destination", "")
                
                # Get current position
                state = self.emulator.get_game_state()
                current_map = state['player']['map']
                
                # Check if we have path information in knowledge base
                paths = self.knowledge_base.get("map_data", "paths") or {}
                key = f"{current_map}_{destination}"
                
                if key in paths:
                    # Execute saved path
                    path = paths[key]
                    result = f"Navigating from map {current_map} to {destination} using stored path.\n"
                    
                    for step in path[:5]:  # Execute first few steps
                        result += self.emulator.press_button(step) + "\n"
                    
                    # Update action stats
                    if "navigate" in self.action_stats:
                        self.action_stats["navigate"] += 1
                    else:
                        self.action_stats["navigate"] = 1
                    
                    return result
                else:
                    return f"No stored path found for {destination}. Please use press_button for manual navigation."
                
            elif tool_name == "update_knowledge":
                section = tool_params.get("section", "")
                key = tool_params.get("key", "")
                value = tool_params.get("value", "")
                
                self.knowledge_base.update(section, key, value)
                
                # Update action stats
                if "update_kb" in self.action_stats:
                    self.action_stats["update_kb"] += 1
                else:
                    self.action_stats["update_kb"] = 1
                
                return f"Updated knowledge base: {section}.{key} = {value}"
            
            else:
                error_msg = f"Unknown tool: {tool_name}. Using wait fallback."
                self.last_error = error_msg
                return self.emulator.wait_frames(30)
                
        except Exception as e:
            # Handle tool execution errors
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            traceback_str = traceback.format_exc()
            print(f"{error_msg}\n{traceback_str}")
            
            # Store the error
            self.last_error = error_msg
            
            # Fall back to waiting
            try:
                return self.emulator.wait_frames(30)
            except Exception as e2:
                return f"Both tool execution and wait fallback failed: {str(e2)}"
    
    def analyze_game_state(self):
        """Analyze the current game state and get an action from Claude."""
        try:
            # Get the current game state
            game_state = self.emulator.get_game_state()
            screen_base64 = self.emulator.get_screen_base64()
            state_text = self.emulator.format_game_state(game_state)
            kb_summary = self.knowledge_base.summarize_recent_history()
            
            # Create a system message for Claude
            system_prompt = """
You are an expert Pokémon player. You analyze the game state and choose the best action to make progress.

Focus on:
1. Understanding the current situation (location, battles, UI)
2. Selecting appropriate actions to advance in the game
3. Using tools to control the Game Boy

If you're unsure what to do, or if there was an error with the previous action, use wait_frames
instead of pressing buttons randomly. Waiting is safer than pressing buttons when uncertain.
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
Please analyze the current game state:

{state_text}

Recent actions:
{kb_summary}

{f"Note about previous action: {self.last_error}" if self.last_error else ""}

Based on the screen and game state information, what action should I take next?
"""
                }
            ]

            # Prepare the message for Claude
            messages: List[MessageParam] = []
            
            # Add message history if we have it
            if len(self.message_history) > 0:
                messages.extend(self.message_history[-5:])  # Last 5 exchanges
            
            # Add the current message
            messages.append({
                "role": "user",
                "content": message_content
            })
            
            # Call Claude
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                tools=self.tools
            )
            
            # Add the response to message history
            self.message_history.append({
                "role": "user",
                "content": message_content
            })
            self.message_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Keep message history at reasonable size
            if len(self.message_history) > 20:
                self.message_history = self.message_history[-20:]
            
            return response
            
        except Exception as e:
            print(f"Error analyzing game state: {e}")
            traceback_str = traceback.format_exc()
            print(traceback_str)
            return None
    
    def handle_tool_calls(self, response):
        """Handle tool calls from Claude's response."""
        # Process tool usage if any
        tool_result = None
        action_taken = "wait"  # Default fallback
        
        try:
            # Check for tool usage in the response
            for idx, content_block in enumerate(response.content):
                if content_block.type == "tool_use":
                    tool_use: ToolUseBlock = content_block
                    tool_name = tool_use.name
                    tool_params = tool_use.input
                    
                    print(f"Tool call: {tool_name} with params: {tool_params}")
                    
                    # Execute the tool
                    tool_result = self._execute_tool(tool_name, tool_params)
                    action_taken = tool_name
                    
                    if tool_name == "press_button":
                        action_taken = tool_params.get("button", "unknown")
                    
                    # Update knowledge base with this action
                    self.knowledge_base.add_action(action_taken, tool_result)
                    
                    # Update conversation turn counter
                    self.conversation_turn_count += 1
                    
                    # Send tool results back to Claude
                    tool_response = self.client.messages.create(
                        model=self.model_name,
                        messages=self.message_history + [
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "tool_use",
                                        "id": tool_use.id,
                                        "name": tool_use.name,
                                        "input": tool_use.input
                                    }
                                ]
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_use.id,
                                        "content": tool_result
                                    }
                                ]
                            }
                        ],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        tools=self.tools
                    )
                    
                    # Update message history with tool use and result
                    self.message_history.append({
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": tool_use.id,
                                "name": tool_use.name,
                                "input": tool_use.input
                            }
                        ]
                    })
                    self.message_history.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": tool_result
                            }
                        ]
                    })
                    
                    return action_taken, tool_result
                
            # If no tool was used, default to waiting
            if tool_result is None:
                print("No tool call detected, using wait fallback")
                tool_result = self.emulator.wait_frames(30)
                action_taken = "wait"
                
                # Update knowledge base
                self.knowledge_base.add_action(action_taken, tool_result)
                
                # Update conversation turn counter
                self.conversation_turn_count += 1
                
            return action_taken, tool_result
            
        except Exception as e:
            print(f"Error handling tool calls: {e}")
            traceback_str = traceback.format_exc()
            print(traceback_str)
            
            # Fall back to waiting
            try:
                result = self.emulator.wait_frames(30)
                self.knowledge_base.add_action("wait", result)
                return "wait", result
            except Exception as e2:
                print(f"Even wait fallback failed: {e2}")
                return "error", "Both tool handling and wait fallback failed"
    
    def check_summarization(self):
        """Check if we need to summarize conversation to manage context."""
        if self.conversation_turn_count >= self.max_turns_before_summary:
            try:
                print(f"Generating summary after {self.conversation_turn_count} turns")
                
                # Get recent action history
                history_text = "\n".join([
                    f"Action: {item['action']}, Result: {item['result']}" 
                    for item in self.knowledge_base.data["action_history"][-20:]
                ])
                
                # Request a summary from Claude
                summary_response = self.client.messages.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""
Please summarize the recent game progress:

{history_text}

Previous summary (if any):
{self.last_summary}

Focus on:
1. Current location and objective
2. Recent battles and encounters
3. Party status
4. Important items obtained
5. Progress towards game goals
"""
                        }
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                
                # Extract the summary
                summary_text = ""
                for content in summary_response.content:
                    if content.type == "text":
                        summary_text = content.text
                        break
                
                # Update last summary
                self.last_summary = summary_text
                
                # Reset conversation turn counter
                self.conversation_turn_count = 0
                
                # Clear message history but keep the summary as context
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
                
                print("Successfully generated summary and reset context")
                
            except Exception as e:
                print(f"Error during summarization: {e}")
                # Continue without summarizing if it fails
                pass
    
    def start(self, skip_intro=True):
        """Start the game."""
        self.emulator.start_game(skip_intro=skip_intro)
        self.start_time = time.time()
        
    def run(self, num_steps=None):
        """Run the agent for a specified number of steps."""
        if self.start_time is None:
            self.start()
        
        step_count = 0
        
        try:
            print("Starting Pokémon Anthropic Agent...")
            
            # Main agent loop
            while num_steps is None or step_count < num_steps:
                # Get game analysis and action decision from Claude
                response = self.analyze_game_state()
                
                if response:
                    # Handle any tool calls
                    action, result = self.handle_tool_calls(response)
                    
                    # Increment step counter
                    step_count += 1
                    
                    # Print progress
                    print(f"Step {step_count}: {action} -> {result}")
                    
                    # Print action stats every 10 steps
                    if step_count % 10 == 0:
                        print("\nAction distribution:")
                        total = sum(self.action_stats.values())
                        for act, count in sorted(self.action_stats.items(), key=lambda x: x[1], reverse=True):
                            print(f"  {act}: {count} ({count/total*100:.1f}%)")
                    
                    # Save game state periodically
                    if step_count % 50 == 0:
                        save_path = self.emulator.save_state(f"step_{step_count}.state")
                        print(f"Saved game state to {save_path}")
                    
                    # Check if we need to summarize
                    self.check_summarization()
                
                else:
                    # If analysis failed, wait and try again
                    print("Game state analysis failed, waiting...")
                    self.emulator.wait_frames(30)
                
                # Small delay to prevent maxing CPU
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Error running agent: {e}")
            traceback_str = traceback.format_exc()
            print(f"Traceback:\n{traceback_str}")
        finally:
            # Save final state
            self.emulator.save_state("final.state")
            
            # Save action statistics
            with open("action_stats.json", 'w') as f:
                json.dump(self.action_stats, f, indent=2)
            
            # Print summary
            elapsed_time = time.time() - self.start_time
            print(f"\nRun complete! Elapsed time: {elapsed_time:.2f}s")
            print(f"Frame count: {self.emulator.frame_count}")
            
            # Clean up
            self.emulator.close()
    
    def close(self):
        """Clean up resources."""
        self.emulator.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a Pokémon AI agent using direct Anthropic API')
    parser.add_argument('rom_path', help='Path to the Pokémon ROM file')
    parser.add_argument('--api-key', help='API key (will use ANTHROPIC_API_KEY env var if not provided)')
    parser.add_argument('--model', default='claude-3-7-sonnet-20250219', help='Model to use')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no window)')
    parser.add_argument('--steps', type=int, help='Number of steps to run (infinite if not specified)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for the LLM')
    parser.add_argument('--speed', type=int, default=1, help='Game speed multiplier')
    parser.add_argument('--max-tokens', type=int, default=10000, help='Maximum tokens for LLM')
    parser.add_argument('--no-sound', action='store_true', help='Disable sound')
    parser.add_argument('--thinking', action='store_true', help='Enable thinking mode')
    
    args = parser.parse_args()
    
    # Create and run the agent
    sound = not args.no_sound
    agent = PokemonAnthropicAgent(
        rom_path=args.rom_path,
        model_name=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        headless=args.headless,
        speed=args.speed,
        sound=sound
    )
    
    try:
        agent.start()
        agent.run(num_steps=args.steps)
    finally:
        agent.close()