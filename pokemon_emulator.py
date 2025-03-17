"""
Pokemon Emulator Module - Interface for the PyBoy Game Boy emulator
to play Pokemon Red/Blue.
"""

import os
import time
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Any, Tuple, Optional, List

from pyboy import PyBoy
from pyboy.utils import WindowEvent

# Import our location mapping
from pokemon_locations import get_location_name, get_pokemon_name, get_move_name, get_type_name


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
        'enemy_pokemon_species': 0xCFDE,  # Enemy Pokémon species ID
        'enemy_pokemon_status': 0xCFF1,    # Enemy Pokémon status condition

        # Player Pokémon party
        'pokemon_count': 0xD163,       # Number of Pokémon in party
        'party_species': 0xD164,       # 6 bytes: species IDs
        'party_hp': 0xD16C,            # 12 bytes: current HP (2 bytes each)
        'party_status': 0xD178,        # 6 bytes: status conditions (corrected; was 0xD16F)
        'party_level': 0xD17E,         # 6 bytes: levels (corrected; was 0xD18C)
        'party_max_hp': 0xD184,        # 12 bytes: max HP (corrected; was 0xD18D)
        'party_attack': 0xD190,        # 12 bytes: attack stat (corrected; was 0xD197)
        'party_defense': 0xD19A,       # 12 bytes: defense stat (corrected; was 0xD1A1)
        'party_speed': 0xD1A4,         # 12 bytes: speed stat (corrected; was 0xD1AB)
        'party_special': 0xD1AE,       # 12 bytes: special stat (corrected; was 0xD1B5)
        'party_types': 0xD1B8,         # 6 bytes: types (corrected; was 0xD1BF)
        'party_moves': 0xD1BE,         # 24 bytes: moves (corrected; was 0xD1D7)
        'party_pp': 0xD1E8,            # 24 bytes: PP for moves (corrected; was 0xD1F1)

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
    
    def close(self):
        """Clean up resources."""
        self.pyboy.stop()


def test_emulator(rom_path):
    """
    Test function to verify that the emulator is working correctly.
    
    Args:
        rom_path: Path to the Pokémon ROM file
    """
    print("\n==== Pokémon Emulator Test ====")
    print(f"ROM Path: {rom_path}")
    print("Initializing emulator...")
    emulator = GameEmulator(rom_path, headless=False, speed=1, sound=False)
    
    try:
        print("\nStarting game...")
        emulator.start_game(skip_intro=True)
        
        print("\nPerforming test actions...")
        # Press some buttons
        print(emulator.press_button('a', hold_frames=10))
        print(emulator.wait_frames(30))
        print(emulator.press_button('up', hold_frames=15))
        print(emulator.press_button('down', hold_frames=15))
        
        # Get and display game state
        print("\nGame State Information:")
        state = emulator.get_game_state()
        print(emulator.format_game_state(state))
        
        # Test memory reading
        x_pos = emulator.get_memory_value('player_x')
        y_pos = emulator.get_memory_value('player_y')
        map_id = emulator.get_memory_value('map_id')
        print(f"\nMemory Reading Test:")
        print(f"  Player Position: ({x_pos}, {y_pos}) on Map {map_id}")
        
        # Save a state
        save_path = emulator.save_state("test_state.state")
        print(f"\nSaved state to: {save_path}")
        
        print("\nEmulator test completed successfully!")
        
    except Exception as e:
        print(f"Error during emulator test: {e}")
        traceback_str = traceback.format_exc()
        print(traceback_str)
    finally:
        # Clean up
        print("\nClosing emulator...")
        emulator.close()
        print("Emulator closed.")


if __name__ == "__main__":
    # Run test if this file is executed directly
    import argparse
    import traceback
    
    parser = argparse.ArgumentParser(description='Test the Pokémon Game Boy emulator')
    parser.add_argument('rom_path', help='Path to the Pokémon ROM file')
    
    args = parser.parse_args()
    
    test_emulator(args.rom_path)