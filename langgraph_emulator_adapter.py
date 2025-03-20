"""
LangGraph Emulator Adapter - Enhanced adapter for PyBoy Game Boy emulator
with asynchronous capabilities for LangGraph integration.
"""

import os
import time
import base64
import asyncio
from io import BytesIO
from typing import Dict, Any, Tuple, Optional, List

from PIL import Image
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent

# Import our location mapping for context
from pokemon_locations import get_location_name, get_pokemon_name, get_move_name, get_type_name


class GameEmulatorAdapter:
    """Enhanced adapter for the PyBoy Game Boy emulator with async support."""
    
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
    
    # Memory addresses for Pokémon Red/Blue (same as original)
    MEMORY_MAP = {
        # Player and Map 
        'player_x': 0xD362,            # X position on map
        'player_y': 0xD361,            # Y position on map
        'map_id': 0xD35E,              # Current map ID
        'player_direction': 0xD367,    # Direction (0: down, 4: up, 8: left, 0xC: right)
        'player_movement': 0xD528,     # Movement state
        'bike_riding': 0xD730,         # Bit 5 of this byte is set when riding bicycle

        # Battle system
        'battle_type': 0xD057,         # Non-zero when in battle
        'battle_menu_cursor': 0xCC28,  # Position of cursor in battle menu
        'battle_turn': 0xCCD3,         # Whose turn it is in battle
        'enemy_pokemon_hp': 0xCFE7,    # Enemy Pokémon HP (2 bytes, little endian)
        'enemy_pokemon_level': 0xCFE8, # Enemy Pokémon level
        'enemy_pokemon_species': 0xCFE5,  # Enemy Pokémon species ID
        'enemy_pokemon_status': 0xCFE9,    # Enemy Pokémon status condition

        # Player Pokémon party
        'pokemon_count': 0xD163,       # Number of Pokémon in party
        'party_species': 0xD164,       # 6 bytes: species IDs
        'party_hp': 0xD016,            # 12 bytes: current HP (2 bytes each)
        'party_status': 0xD178,        # 6 bytes: status conditions
        'party_level': 0xD17E,         # 6 bytes: levels
        'party_max_hp': 0xD024,        # 12 bytes: max HP
        'party_attack': 0xD026,        # 12 bytes: attack stat
        'party_defense': 0xD028,       # 12 bytes: defense stat
        'party_speed': 0xD02A,         # 12 bytes: speed stat
        'party_special': 0xD02C,       # 12 bytes: special stat
        'party_types': 0xD170,         # 6 bytes: types
        'party_moves': 0xD01C,         # 24 bytes: moves
        'party_pp': 0xD02D,            # 24 bytes: PP for moves

        # Menu and text
        'text_progress': 0xFF8C,       # Text box progress (non-zero when text is displaying)
        'text_id': 0xF8C8,             # ID of the text being displayed
        'menu_open': 0xFFF5,           # Menu state (0 = no menu open)
        'dialog_type': 0xD355,         # Type of dialog box showing

        # Items
        'item_count': 0xD31D,          # Number of items in bag
        'money': 0xD347,               # Money (3 bytes, BCD format)

        # Badges and progress
        'badges': 0xD356,              # Badges obtained (each bit = one badge)
        'events': 0xD747,              # Event flags (large area, multiple addresses)

        # Various game states
        'game_mode': 0xD057,           # Current game mode (shared with battle flag)
        'joypad_state': 0xFF8B,        # Current joypad state
    }
    
    def __init__(self, rom_path, headless=False, speed=1, sound=True):
        """Initialize the game emulator adapter."""
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
        
        # Asyncio lock for thread safety
        self._lock = asyncio.Lock()
    
    def start_game(self, skip_intro=True):
        """Start the game and optionally skip intro."""
        # Boot the game
        for _ in range(500):
            self.pyboy.tick()
            self.frame_count += 1
        
        if skip_intro:
            # Skip intro by pressing start a few times
            for _ in range(3):
                self.press_button('start', hold_frames=5)
                self.pyboy.tick()
                self.frame_count += 1
        
        # Wait for game to settle
        for _ in range(60):
            self.pyboy.tick()
            self.frame_count += 1
    
    async def async_start_game(self, skip_intro=True):
        """Start the game asynchronously."""
        async with self._lock:
            self.start_game(skip_intro=skip_intro)
    
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
    
    async def async_press_button(self, button, hold_frames=10):
        """Press a button asynchronously."""
        async with self._lock:
            return self.press_button(button, hold_frames)
    
    def wait_frames(self, num_frames=30):
        """Wait for specified frames."""
        for _ in range(num_frames):
            self.pyboy.tick()
            self.frame_count += 1
        
        return f"Waited for {num_frames} frames"
    
    async def async_wait_frames(self, num_frames=30):
        """Wait for specified frames asynchronously."""
        async with self._lock:
            return self.wait_frames(num_frames)
    
    def tick_single_frame(self, render=True):
        """Process a single frame."""
        result = self.pyboy.tick(1, render)
        self.frame_count += 1
        return result
    
    async def async_tick_single_frame(self, render=True):
        """Process a single frame asynchronously."""
        async with self._lock:
            return self.tick_single_frame(render)
    
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
    
    def is_bike_riding(self):
        """Check if the player is currently riding a bicycle."""
        value = self.get_memory_value('bike_riding')
        if value is None:
            return False
        return (value & 0x20) != 0  # Check bit 5
    
    def get_screen_base64(self):
        """Get current screen as base64 string."""
        screen_array = self.screen.ndarray
        pil_image = Image.fromarray(screen_array)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    async def async_get_screen_base64(self):
        """Get current screen as base64 string asynchronously."""
        async with self._lock:
            return self.get_screen_base64()
    
    def get_screen_pil(self):
        """Get current screen as PIL Image."""
        screen_array = self.screen.ndarray
        return Image.fromarray(screen_array)
    
    def get_party_pokemon(self):
        """Get detailed information about all Pokémon in the party."""
        pokemon_count = self.get_memory_value('pokemon_count') or 0
        if pokemon_count == 0:
            return []
        
        party = []
        for i in range(min(pokemon_count, 6)):  # Up to 6 Pokémon
            # Base addresses for this Pokémon
            species_addr = self.MEMORY_MAP['party_species'] + i
            status_addr = self.MEMORY_MAP['party_status'] + i
            level_addr = self.MEMORY_MAP['party_level'] + i
            
            hp_addr = self.MEMORY_MAP['party_hp'] + (i * 2)
            max_hp_addr = self.MEMORY_MAP['party_max_hp'] + (i * 2)
            
            attack_addr = self.MEMORY_MAP['party_attack'] + (i * 2)
            defense_addr = self.MEMORY_MAP['party_defense'] + (i * 2)
            speed_addr = self.MEMORY_MAP['party_speed'] + (i * 2)
            special_addr = self.MEMORY_MAP['party_special'] + (i * 2)
            
            types_addr = self.MEMORY_MAP['party_types'] + (i * 2)
            moves_addr = self.MEMORY_MAP['party_moves'] + (i * 4)
            pp_addr = self.MEMORY_MAP['party_pp'] + (i * 4)
            
            # Get values
            species_id = self.get_memory_value(species_addr)
            level = self.get_memory_value(level_addr)
            status = self.get_memory_value(status_addr)
            
            hp = self.get_2byte_value(hp_addr)
            max_hp = self.get_2byte_value(max_hp_addr)
            
            attack = self.get_2byte_value(attack_addr)
            defense = self.get_2byte_value(defense_addr)
            speed = self.get_2byte_value(speed_addr)
            special = self.get_2byte_value(special_addr)
            
            type1 = self.get_memory_value(types_addr)
            type2 = self.get_memory_value(types_addr + 1)
            
            # Get moves and PP
            moves = []
            for j in range(4):
                move_id = self.get_memory_value(moves_addr + j)
                pp = self.get_memory_value(pp_addr + j)
                
                if move_id and move_id > 0:
                    moves.append({
                        "id": move_id,
                        "name": get_move_name(move_id),
                        "pp": pp
                    })
            
            # Create Pokémon object
            pokemon = {
                "index": i + 1,
                "species_id": species_id,
                "name": get_pokemon_name(species_id),
                "level": level,
                "status": status,
                "hp": hp,
                "max_hp": max_hp,
                "attack": attack,
                "defense": defense,
                "speed": speed,
                "special": special,
                "types": [
                    get_type_name(type1) if type1 is not None else "Unknown",
                    get_type_name(type2) if type2 is not None else "Unknown"
                ],
                "moves": moves
            }
            
            party.append(pokemon)
        
        return party
    
    def get_game_state(self):
        """Get comprehensive game state."""
        # Basic info
        x = self.get_memory_value('player_x')
        y = self.get_memory_value('player_y')
        map_id = self.get_memory_value('map_id')
        direction = self.get_memory_value('player_direction')
        in_battle = self.get_memory_value('battle_type') != 0
        text_active = self.get_memory_value('text_progress') != 0
        bike_riding = self.is_bike_riding()
        
        # Pokémon info
        pokemon_count = self.get_memory_value('pokemon_count') or 0
        party_pokemon = self.get_party_pokemon()
        
        # First Pokémon quick access (for backward compatibility)
        first_hp = party_pokemon[0]['hp'] if pokemon_count > 0 and party_pokemon else 0
        first_max_hp = party_pokemon[0]['max_hp'] if pokemon_count > 0 and party_pokemon else 0
        
        # Enemy Pokémon in battle
        enemy_pokemon = None
        if in_battle:
            enemy_species = self.get_memory_value('enemy_pokemon_species')
            enemy_pokemon = {
                "species_id": enemy_species,
                "name": get_pokemon_name(enemy_species),
                "level": self.get_memory_value('enemy_pokemon_level'),
                "hp": self.get_2byte_value('enemy_pokemon_hp'),
                "status": self.get_memory_value('enemy_pokemon_status')
            }
        
        # Progress indicators
        badges = self.get_memory_value('badges')
        badge_count = bin(badges).count('1') if badges is not None else 0
        
        # Format money from BCD (Binary-Coded Decimal)
        money_bytes = [
            self.get_memory_value(self.MEMORY_MAP['money']),
            self.get_memory_value(self.MEMORY_MAP['money'] + 1),
            self.get_memory_value(self.MEMORY_MAP['money'] + 2)
        ]
        
        money = 0
        for byte in money_bytes:
            if byte is not None:
                money = money * 100 + ((byte >> 4) * 10 + (byte & 0x0F))
        
        # Build the complete state object
        return {
            'position': (x, y),
            'map_id': map_id,
            'location': get_location_name(map_id),
            'direction': direction,
            'in_battle': in_battle,
            'text_active': text_active,
            'bike_riding': bike_riding,
            'pokemon_count': pokemon_count,
            'first_pokemon_hp': first_hp,
            'first_pokemon_max_hp': first_max_hp,
            'party_pokemon': party_pokemon,
            'enemy_pokemon': enemy_pokemon,
            'badges': badge_count,
            'money': money,
            'frame': self.frame_count
        }
    
    async def async_get_game_state(self):
        """Get game state asynchronously."""
        async with self._lock:
            return self.get_game_state()
    
    def format_game_state(self, state):
        """Format game state as readable text."""
        text = "CURRENT GAME STATE:\n"
        text += f"Location: {state['location']} (Map ID: {state['map_id']})\n"
        text += f"Position: ({state['position'][0]}, {state['position'][1]})\n"
        text += f"Direction: {state['direction']}, Badges: {state['badges']}, Money: ${state['money']}\n"
        
        if state['bike_riding']:
            text += "Currently riding bicycle\n"
            
        if state['in_battle']:
            text += "\nBATTLE STATUS:\n"
            if state['enemy_pokemon']:
                enemy = state['enemy_pokemon']
                text += f"Fighting {enemy['name']} (Lv. {enemy['level']}) - HP: {enemy['hp']}\n"
        
        text += f"\nPOKÉMON PARTY ({state['pokemon_count']}):\n"
        
        for pokemon in state['party_pokemon']:
            text += f"{pokemon['name']} (Lv. {pokemon['level']}) - HP: {pokemon['hp']}/{pokemon['max_hp']}\n"
            text += f"  Types: {'/'.join(pokemon['types'])}\n"
            text += f"  Moves: {', '.join([f'{m['name']} (PP: {m['pp']})' for m in pokemon['moves']])}\n"
        
        if state['text_active']:
            text += "\nText is currently being displayed\n"
        
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
    
    async def async_save_state(self, filename=None):
        """Save game state asynchronously."""
        async with self._lock:
            return self.save_state(filename)
    
    def load_state(self, filename):
        """Load game state from file."""
        if not filename.startswith(self.save_dir):
            filename = f"{self.save_dir}/{filename}"
            
        try:
            with open(filename, "rb") as f:
                self.pyboy.load_state(f)
            return f"Successfully loaded state from {filename}"
        except Exception as e:
            return f"Failed to load state: {str(e)}"
    
    async def async_load_state(self, filename):
        """Load game state asynchronously."""
        async with self._lock:
            return self.load_state(filename)
    
    async def run_game_loop(self, fps=60, stop_event=None):
        """
        Run the game in a continuous loop asynchronously.
        
        Args:
            fps: Target frames per second
            stop_event: Optional asyncio.Event to signal stopping
        """
        frame_time = 1.0 / fps
        last_frame_time = time.time()
        
        while True:
            # Check if we should stop
            if stop_event and stop_event.is_set():
                break
                
            # Calculate time since last frame
            now = time.time()
            elapsed = now - last_frame_time
            
            # If it's time for a new frame
            if elapsed >= frame_time:
                # Process frame without blocking
                async with self._lock:
                    self.pyboy.tick(1, True)
                    self.frame_count += 1
                last_frame_time = now
            
            # Yield to other tasks
            await asyncio.sleep(0.001)
    
    def close(self):
        """Clean up resources."""
        self.pyboy.stop()

# Simple test code
if __name__ == "__main__":
    import argparse
    import traceback
    
    parser = argparse.ArgumentParser(description='Test the Pokémon Game Boy emulator adapter')
    parser.add_argument('rom_path', help='Path to the Pokémon ROM file')
    
    args = parser.parse_args()
    
    # Create and test the emulator adapter
    adapter = GameEmulatorAdapter(args.rom_path, headless=False, speed=1, sound=False)
    
    try:
        print("Starting game...")
        adapter.start_game(skip_intro=True)
        
        print("Performing test actions...")
        print(adapter.press_button('a', hold_frames=10))
        print(adapter.wait_frames(30))
        
        # Get and display game state
        print("\nGame State Information:")
        state = adapter.get_game_state()
        print(adapter.format_game_state(state))
        
        # Save a state
        save_path = adapter.save_state("test_state.state")
        print(f"\nSaved state to: {save_path}")
        
        # Async test
        async def test_async():
            print("\nTesting async functions...")
            await adapter.async_press_button('up', hold_frames=10)
            await adapter.async_wait_frames(30)
            state = await adapter.async_get_game_state()
            print("Updated position:", state['position'])
            
            # Test continuous game loop
            stop_event = asyncio.Event()
            
            # Create task for game loop
            loop_task = asyncio.create_task(adapter.run_game_loop(fps=60, stop_event=stop_event))
            
            # Let it run for a few seconds
            await asyncio.sleep(3)
            
            # Stop the loop
            stop_event.set()
            await loop_task
            
            print("Async tests completed")
        
        # Run async tests
        asyncio.run(test_async())
        
        print("\nAdapter tests completed successfully!")
        
    except Exception as e:
        print(f"Error during adapter test: {e}")
        traceback.print_exc()
    finally:
        # Clean up
        print("\nClosing emulator...")
        adapter.close()
        print("Emulator closed.")