import os
import time
import base64
import numpy as np
from io import BytesIO
from abc import ABC, abstractmethod
from PIL import Image
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import json

class PokemonGameInterface:
    """
    Interface class for Pokémon Red/Blue that handles interaction with the game.
    Provides methods for accessing game state and sending inputs.
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
        'player_movement': 0xD528,     # Movement state

        # Battle system
        'battle_type': 0xD057,         # Non-zero when in battle
        'battle_menu_cursor': 0xCC28,  # Position of cursor in battle menu
        'battle_turn': 0xCCD3,         # Whose turn it is in battle
        'enemy_pokemon_hp': 0xCFE7,    # Enemy Pokémon HP (2 bytes, little endian)
        'enemy_pokemon_level': 0xCFE8, # Enemy Pokémon level
        'enemy_pokemon_species': 0xCFDE, # Enemy Pokémon species ID
        'enemy_pokemon_status': 0xCFF1, # Enemy Pokémon status condition
        
        # Player Pokémon
        'pokemon_count': 0xD163,       # Number of Pokémon in party
        'first_pokemon_hp': 0xD16C,    # First Pokémon's current HP (2 bytes)
        'first_pokemon_max_hp': 0xD18D, # First Pokémon's max HP (2 bytes)
        'first_pokemon_level': 0xD18C, # First Pokémon's level
        'first_pokemon_status': 0xD16F, # First Pokémon's status condition
        'first_pokemon_species': 0xD164, # First Pokémon's species ID
        
        # Menu and text
        'text_progress': 0xC6AC,       # Text box progress (non-zero when text is displaying)
        'menu_open': 0xD730,           # Menu state (various flags)
        'dialog_type': 0xD355,         # Type of dialog box showing
        
        # Items
        'item_count': 0xD31D,          # Number of items in bag
        'money': 0xD347,               # Money (3 bytes, BCD format)
        
        # Badges and progress
        'badges': 0xD356,              # Badges obtained (each bit = one badge)
        'events': 0xD747,              # Event flags (large area, multiple addresses)
        
        # Various game states
        'game_mode': 0xD057,           # Current game mode
        'joypad_state': 0xC0F0,        # Current joypad state
    }
    
    def __init__(self, rom_path, headless=False, speed=1):
        """
        Initialize the game interface.
        
        Args:
            rom_path (str): Path to the Pokémon ROM file
            headless (bool): Whether to run in headless mode (no window)
            speed (int): Emulation speed (1 = normal)
        """
        self.rom_path = rom_path
        window_type = "null" if headless else "SDL2"
        self.pyboy = PyBoy(rom_path, window_type=window_type)
        self.pyboy.set_emulation_speed(speed)
        
        # Setup bot support manager
        self.screen_manager = self.pyboy.botsupport_manager().screen()
        
        # Game state tracking
        self.frame_count = 0
        self.history = []
        self.save_dir = "agent_data"
        
        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
    
    def start_game(self, skip_intro=True):
        """
        Start the game and optionally skip the intro sequence.
        
        Args:
            skip_intro (bool): Whether to skip the intro sequence
        """
        # Let the game boot up
        for _ in range(100):
            self.pyboy.tick()
        
        if skip_intro:
            # Skip intro by pressing start a few times
            for _ in range(3):
                self.press_button('start', hold_frames=5)
                time.sleep(0.1)
        
        # Wait a bit more for the game to settle
        for _ in range(60):
            self.pyboy.tick()
    
    def press_button(self, button, hold_frames=10):
        """
        Press a button for the specified number of frames.
        
        Args:
            button (str): Button to press ('up', 'down', 'left', 'right', 'a', 'b', 'start', 'select')
            hold_frames (int): Number of frames to hold the button
        """
        if button not in self.BUTTONS:
            print(f"Warning: Unknown button '{button}'")
            return
        
        # Press the button
        self.pyboy.send_input(self.BUTTONS[button])
        
        # Hold for specified frames
        for _ in range(hold_frames):
            self.pyboy.tick()
            self.frame_count += 1
        
        # Release the button
        self.pyboy.send_input(self.RELEASE_BUTTONS[button])
        self.pyboy.tick()
        self.frame_count += 1
    
    def get_memory_value(self, address):
        """
        Get the value at a specific memory address.
        
        Args:
            address (int): Memory address to read
            
        Returns:
            int: Value at the memory address
        """
        if isinstance(address, str):
            if address in self.MEMORY_MAP:
                address = self.MEMORY_MAP[address]
            else:
                print(f"Warning: Unknown memory address alias '{address}'")
                return None
        
        return self.pyboy.get_memory_value(address)
    
    def get_memory_values(self, addresses):
        """
        Get values from multiple memory addresses.
        
        Args:
            addresses (list): List of memory addresses or their aliases
            
        Returns:
            dict: Dictionary mapping addresses or aliases to their values
        """
        result = {}
        for addr in addresses:
            if isinstance(addr, str):
                if addr in self.MEMORY_MAP:
                    result[addr] = self.get_memory_value(self.MEMORY_MAP[addr])
                else:
                    print(f"Warning: Unknown memory address alias '{addr}'")
                    result[addr] = None
            else:
                result[hex(addr)] = self.get_memory_value(addr)
        
        return result
    
    def get_2byte_value(self, address):
        """
        Get a 2-byte value from memory (little endian format).
        
        Args:
            address (int or str): Memory address or alias for the first byte
            
        Returns:
            int: 2-byte value
        """
        if isinstance(address, str):
            if address in self.MEMORY_MAP:
                address = self.MEMORY_MAP[address]
            else:
                print(f"Warning: Unknown memory address alias '{address}'")
                return None
        
        low_byte = self.get_memory_value(address)
        high_byte = self.get_memory_value(address + 1)
        return (high_byte << 8) | low_byte
    
    def get_screen(self):
        """
        Get the current screen as a numpy array.
        
        Returns:
            numpy.ndarray: RGB screen image as numpy array
        """
        return self.screen_manager.screen_ndarray()
    
    def get_screen_as_pil(self):
        """
        Get the current screen as a PIL Image.
        
        Returns:
            PIL.Image: Screen as PIL Image
        """
        screen_array = self.get_screen()
        return Image.fromarray(screen_array)
    
    def get_screen_base64(self):
        """
        Get the current screen as a base64 encoded string.
        
        Returns:
            str: Base64 encoded screen image
        """
        pil_image = self.get_screen_as_pil()
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def get_player_position(self):
        """
        Get the player's position on the map.
        
        Returns:
            tuple: (x, y) coordinates of the player
        """
        x = self.get_memory_value('player_x')
        y = self.get_memory_value('player_y')
        return (x, y)
    
    def get_game_state(self):
        """
        Get a comprehensive view of the current game state.
        
        Returns:
            dict: Dictionary containing various game state information
        """
        # Get player position and map
        player_pos = self.get_player_position()
        current_map = self.get_memory_value('map_id')
        player_direction = self.get_memory_value('player_direction')
        
        # Battle information
        in_battle = self.get_memory_value('battle_type') != 0
        
        # Get Pokémon party information
        pokemon_count = self.get_memory_value('pokemon_count')
        
        # First Pokémon info if available
        first_pokemon = {}
        if pokemon_count > 0:
            first_pokemon = {
                'hp': self.get_2byte_value('first_pokemon_hp'),
                'max_hp': self.get_2byte_value('first_pokemon_max_hp'),
                'level': self.get_memory_value('first_pokemon_level'),
                'species': self.get_memory_value('first_pokemon_species'),
                'status': self.get_memory_value('first_pokemon_status')
            }
        
        # Enemy Pokémon info if in battle
        enemy_pokemon = {}
        if in_battle:
            enemy_pokemon = {
                'hp': self.get_2byte_value('enemy_pokemon_hp'),
                'level': self.get_memory_value('enemy_pokemon_level'),
                'species': self.get_memory_value('enemy_pokemon_species'),
                'status': self.get_memory_value('enemy_pokemon_status')
            }
        
        # Text/dialog info
        text_active = self.get_memory_value('text_progress') != 0
        
        # Badges and money
        badges = self.get_memory_value('badges')
        money_bytes = [
            self.get_memory_value(self.MEMORY_MAP['money']),
            self.get_memory_value(self.MEMORY_MAP['money'] + 1),
            self.get_memory_value(self.MEMORY_MAP['money'] + 2)
        ]
        
        # Convert BCD bytes to integer (each nibble is a decimal digit)
        money = 0
        for byte in money_bytes:
            money = money * 100 + ((byte >> 4) * 10 + (byte & 0x0F))
        
        return {
            'frame': self.frame_count,
            'player': {
                'position': player_pos,
                'map': current_map,
                'direction': player_direction,
                'money': money,
                'badges': bin(badges)[2:].zfill(8)  # Convert to binary string
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
                'text_active': text_active,
                'menu_state': self.get_memory_value('menu_open')
            }
        }
    
    def save_state(self, filename=None):
        """
        Save the current game state.
        
        Args:
            filename (str, optional): Filename to save to. If None, generates one based on frame number.
        
        Returns:
            str: Path to the saved state file
        """
        if filename is None:
            filename = f"{self.save_dir}/state_frame_{self.frame_count}.state"
        else:
            filename = f"{self.save_dir}/{filename}"
        
        self.pyboy.save_state(filename)
        return filename
    
    def load_state(self, filename):
        """
        Load a saved game state.
        
        Args:
            filename (str): Filename to load
            
        Returns:
            bool: Whether the load was successful
        """
        filepath = f"{self.save_dir}/{filename}" if not filename.startswith(self.save_dir) else filename
        
        try:
            self.pyboy.load_state(filepath)
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False
    
    def record_state(self, action=None):
        """
        Record the current state and the last action for history.
        
        Args:
            action (str, optional): The last action performed
        """
        state = self.get_game_state()
        screen_base64 = self.get_screen_base64()
        
        record = {
            'frame': self.frame_count,
            'state': state,
            'action': action,
            'timestamp': time.time(),
            'screen': screen_base64  # Include the screen for visual context
        }
        
        self.history.append(record)
        
        # Optionally, periodically save history to disk
        if len(self.history) % 100 == 0:
            self.save_history()
    
    def save_history(self, filename=None):
        """
        Save the action/state history to a file.
        
        Args:
            filename (str, optional): Filename to save to
            
        Returns:
            str: Path to the saved history file
        """
        if filename is None:
            filename = f"{self.save_dir}/history_{int(time.time())}.json"
        else:
            filename = f"{self.save_dir}/{filename}"
        
        with open(filename, 'w') as f:
            json.dump(self.history, f)
        
        return filename
    
    def wait_frames(self, num_frames):
        """
        Wait for a specified number of frames.
        
        Args:
            num_frames (int): Number of frames to wait
        """
        for _ in range(num_frames):
            self.pyboy.tick()
            self.frame_count += 1
    
    def is_in_battle(self):
        """
        Check if the player is currently in a battle.
        
        Returns:
            bool: True if in battle, False otherwise
        """
        return self.get_memory_value('battle_type') != 0
    
    def is_text_active(self):
        """
        Check if text is currently being displayed.
        
        Returns:
            bool: True if text is active, False otherwise
        """
        return self.get_memory_value('text_progress') != 0
    
    def close(self):
        """Clean up resources."""
        self.save_history()
        self.pyboy.stop()


class PokemonLLMAgentBase(ABC):
    """
    Abstract base class for Pokémon LLM agents.
    Provides a framework for building different LLM agent implementations.
    """
    
    def __init__(self, game_interface, observation_interval=30):
        """
        Initialize the LLM agent.
        
        Args:
            game_interface (PokemonGameInterface): Interface to the Pokémon game
            observation_interval (int): Number of frames between observations
        """
        self.game = game_interface
        self.observation_interval = observation_interval
        self.last_observation_frame = 0
        self.action_history = []
        self.context_window = []
        self.max_context_entries = 10  # Maximum number of entries in the context window
    
    @abstractmethod
    def get_llm_response(self, prompt, system_prompt=None):
        """
        Get a response from the LLM.
        Must be implemented by subclasses with specific LLM implementations.
        
        Args:
            prompt (str): The prompt to send to the LLM
            system_prompt (str, optional): Optional system prompt
            
        Returns:
            str: Response from the LLM
        """
        pass
    
    def format_game_state(self, state):
        """
        Format the game state into a text representation for the LLM.
        
        Args:
            state (dict): Game state dictionary
            
        Returns:
            str: Formatted game state text
        """
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
            text += f"First Pokémon: Species #{first['species']}, Level {first['level']}, HP: {first['hp']}/{first['max_hp']}\n"
        
        # UI state
        ui = state['ui']
        if ui['text_active']:
            text += "\nText is currently being displayed.\n"
        
        if ui['menu_state'] > 0:
            text += "A menu is currently open.\n"
        
        return text
    
    def create_observation_prompt(self):
        """
        Create a prompt based on the current game state for the LLM.
        
        Returns:
            str: Prompt describing the current game state
        """
        # Get the current state
        state = self.game.get_game_state()
        
        # Get the screen as base64
        screen_base64 = self.game.get_screen_base64()
        
        # Format the game state
        state_text = self.format_game_state(state)
        
        # Create a context section with recent history
        context = "\nRECENT CONTEXT:\n"
        for i, entry in enumerate(self.context_window[-5:]):  # Show last 5 entries
            context += f"{i+1}. {entry}\n"
        
        # Assemble the prompt
        prompt = f"""
OBSERVATION:
Frame: {self.game.frame_count}

{state_text}

{context}

The screen is provided as a base64 encoded PNG image.

Available Actions:
- press_up: Move up
- press_down: Move down
- press_left: Move left
- press_right: Move right
- press_a: Interact/Select/Confirm
- press_b: Cancel/Back
- press_start: Open menu
- press_select: Secondary menu
- wait: Wait for a few frames

What action should I take next? Please respond with ONE action from the list above.
"""
        return prompt, screen_base64
    
    def update_context(self, action, observation):
        """
        Update the context window with a new action and observation.
        
        Args:
            action (str): The action that was taken
            observation (str): The observation after taking the action
        """
        entry = f"Action: {action}, Result: {observation}"
        
        self.context_window.append(entry)
        
        # Keep context window within size limit
        if len(self.context_window) > self.max_context_entries:
            self.context_window.pop(0)
    
    def execute_action(self, action):
        """
        Execute an action in the game.
        
        Args:
            action (str): Action to execute
            
        Returns:
            str: Description of the result
        """
        # Parse the action
        parts = action.lower().split('_')
        
        if len(parts) < 2:
            return "Invalid action format. Use 'press_X' or 'wait'."
        
        command = parts[0]
        
        if command == "press":
            button = parts[1]
            if button in self.game.BUTTONS:
                self.game.press_button(button)
                return f"Pressed {button} button."
            else:
                return f"Unknown button: {button}"
        
        elif command == "wait":
            self.game.wait_frames(30)  # Wait for half a second at 60 FPS
            return "Waited for 30 frames."
        
        else:
            return f"Unknown command: {command}"
    
    def observe_and_act(self):
        """
        Perform a single observation-action cycle.
        
        Returns:
            tuple: (action, observation, response_time)
        """
        # Only make a new observation if enough frames have passed
        if self.game.frame_count - self.last_observation_frame >= self.observation_interval:
            self.last_observation_frame = self.game.frame_count
            
            # Get the prompt and screen
            prompt, screen = self.create_observation_prompt()
            
            # Get response from LLM
            start_time = time.time()
            response = self.get_llm_response(prompt, screen)
            response_time = time.time() - start_time
            
            # Parse the response to get the action
            # This may need to be customized based on the LLM's output format
            action = self._parse_llm_response(response)
            
            # Execute the action
            observation = self.execute_action(action)
            
            # Update context
            self.update_context(action, observation)
            
            # Record the state
            self.game.record_state(action)
            
            return action, observation, response_time
        
        # If not enough frames have passed, just tick the game forward
        self.game.wait_frames(1)
        return None, None, 0
    
    def _parse_llm_response(self, response):
        """
        Parse the LLM's response to extract the action.
        
        Args:
            response (str): LLM response
            
        Returns:
            str: Extracted action
        """
        # This is a simple parser that looks for known action patterns
        # Could be enhanced based on the specific format of your LLM responses
        
        # Look for "press_X" or "wait" in the response
        possible_actions = ["press_up", "press_down", "press_left", "press_right",
                           "press_a", "press_b", "press_start", "press_select", "wait"]
        
        for action in possible_actions:
            if action.lower() in response.lower():
                return action
        
        # Default to waiting if no valid action found
        print("Warning: Could not parse a valid action from LLM response. Defaulting to wait.")
        print(f"Response was: {response}")
        return "wait"
    
    @abstractmethod
    def run(self, num_steps=None):
        """
        Run the agent for a specified number of steps.
        Must be implemented by subclasses.
        
        Args:
            num_steps (int, optional): Number of steps to run, or None for indefinite
        """
        pass
