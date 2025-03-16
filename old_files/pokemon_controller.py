import sys
import time
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

class PokemonController:
    """
    A controller class for interacting with Pokémon Red through PyBoy.
    """
    # Game Boy buttons
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
    
    # Important memory addresses for Pokémon Red/Blue
    MEMORY_ADDRESSES = {
        'player_x': 0xD362,         # Player X position on the map
        'player_y': 0xD361,         # Player Y position on the map
        'map_id': 0xD35E,           # Current map ID
        'player_direction': 0xD367, # Direction player is facing (0: down, 4: up, 8: left, 0xC: right)
        'battle_type': 0xD057,      # Non-zero when in battle
        'menu_open': 0xD730,        # Menu state (e.g., 0xD730+0xD734 indicates menu open)
        'pokemon_count': 0xD163,    # Number of Pokémon in party
        'first_pokemon_hp': 0xD16C, # First Pokémon's current HP (2 bytes)
        # 'bike_state': 0xD355,       # Bike state (0: off, 1: on)
    }
    
    def __init__(self, rom_path, sound=False):
        """Initialize the Pokémon controller with the ROM path."""
        self.rom_path = rom_path
        self.pyboy = PyBoy(rom_path, window="null", sound_emulated=sound)  # Use null window for script control
        self.pyboy.set_emulation_speed(1)
        self.screen_buffer = None
        print("PyBoy initialized successfully")
        
    def start(self):
        """Start the game."""
        print("Starting game...")
        # Advance a few frames to let the game initialize
        for _ in range(60):  # Wait for ~1 second at 60 FPS
            self.pyboy.tick(1, True)
        
        # Press start to begin the game
        self.press_button('start')
        
    def press_button(self, button, hold_frames=20):
        """
        Press a button for a specified number of frames.
        
        Args:
            button (str): Button to press ('up', 'down', 'left', 'right', 'a', 'b', 'start', 'select')
            hold_frames (int): Number of frames to hold the button
        """
        if button not in self.BUTTONS:
            print(f"Unknown button: {button}")
            return
            
        # Press the button
        # self.pyboy.send_input(self.BUTTONS[button])
        self.pyboy.button_press(button)
        
        # Hold for specified frames
        for _ in range(hold_frames):
            self.pyboy.tick(1, True)
            
        # Release the button
        # self.pyboy.send_input(self.RELEASE_BUTTONS[button])
        self.pyboy.button_release(button)
        self.pyboy.tick(1, True)
        
    def get_screen(self):
        """Get the current screen as a numpy array."""
        return np.array(self.pyboy.screen.ndarray)
    
    def get_memory_value(self, address):
        """Get the value at a specific memory address."""
        return self.pyboy.memory[address]
    
    def get_player_position(self):
        """Get the player's position on the map."""
        x = self.get_memory_value(self.MEMORY_ADDRESSES['player_x'])
        y = self.get_memory_value(self.MEMORY_ADDRESSES['player_y'])
        return x, y
    
    def get_current_map(self):
        """Get the current map ID."""
        return self.get_memory_value(self.MEMORY_ADDRESSES['map_id'])
    
    def is_in_battle(self):
        """Check if the player is currently in a battle."""
        return self.get_memory_value(self.MEMORY_ADDRESSES['battle_type']) != 0
    
    def get_party_info(self):
        """Get basic information about the Pokémon party."""
        count = self.get_memory_value(self.MEMORY_ADDRESSES['pokemon_count'])
        # First Pokémon's HP is stored as a 2-byte value
        first_hp_high = self.get_memory_value(self.MEMORY_ADDRESSES['first_pokemon_hp'])
        first_hp_low = self.get_memory_value(self.MEMORY_ADDRESSES['first_pokemon_hp'] + 1)
        first_hp = (first_hp_high << 8) + first_hp_low
        
        return {
            'count': count,
            'first_pokemon_hp': first_hp
        }
    
    def run_sequence(self, button_sequence):
        """
        Run a sequence of button presses.
        
        Args:
            button_sequence (list): List of button names to press in sequence
        """
        for button in button_sequence:
            self.press_button(button)
            # Small delay between button presses
            for _ in range(5):
                self.pyboy.tick(1, True)
    
    def save_state(self, filename="save_state.state"):
        """Save the current game state."""
        self.pyboy.save_state(filename)
        print(f"Game state saved to {filename}")
    
    def load_state(self, filename="save_state.state"):
        """Load a saved game state."""
        self.pyboy.load_state(filename)
        print(f"Game state loaded from {filename}")
    
    def close(self):
        """Close the emulator."""
        self.pyboy.stop()
        print("Emulator closed")


def run_interactive_demo(rom_path, sound=False):
    """Run an interactive demo with visualization."""
    controller = PokemonController(rom_path, sound=sound)
    controller.start()
    
    # Setup the visualization
    # plt.ion()
    fig, ax = plt.subplots(figsize=(10, 9))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
    
    # Create buttons for controls
    btn_axs = {}
    buttons = ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select']
    
    # Create a 2x4 grid of buttons at the bottom
    for i, btn in enumerate(buttons):
        row = i // 4
        col = i % 4
        x = 0.2 + col * 0.15
        y = 0.15 - row * 0.1
        btn_axs[btn] = plt.axes([x, y, 0.1, 0.05])
    
    # Initial screen
    screen_plot = ax.imshow(np.zeros((144, 160, 3), dtype=np.uint8))
    status_text = ax.text(0.5, -0.1, "Status: Initializing...", 
                         transform=ax.transAxes, ha='center')
    
    # Button click handler
    def on_button_click(event):
        if event.inaxes in btn_axs.values():
            for btn, btn_ax in btn_axs.items():
                if event.inaxes == btn_ax:
                    controller.press_button(btn)
                    print(f'{controller.get_player_position() = }')
                    print(f'{controller.get_current_map() = }')
                    print(f'{controller.is_in_battle() = }')
                    print(f'{controller.get_party_info() = }')
                    # print(controller.pyboy.memory)
                    break
    
    # Connect the button event
    fig.canvas.mpl_connect('button_press_event', on_button_click)
    
    # Animation update function
    def update(_):
        # Update game state
        controller.pyboy.tick(1, True)
        
        # Update the screen
        screen = controller.get_screen()
        screen_plot.set_array(screen)
        
        # Get player info
        x, y = controller.get_player_position()
        map_id = controller.get_current_map()
        in_battle = "Yes" if controller.is_in_battle() else "No"
        
        # Update status
        status_text.set_text(f"Position: ({x}, {y}) | Map: {map_id} | In Battle: {in_battle}")
        
        # Create button text
        for btn in buttons:
            if btn not in btn_axs:
                continue
            ax = btn_axs[btn]
            if not hasattr(ax, 'button_text'):
                ax.button_text = ax.text(0.5, 0.5, btn, ha='center', va='center')
        
        return [screen_plot, status_text]
    
    # Run the animation
    # ani = FuncAnimation(fig, update, frames=None, blit=True, interval=50)
    ani = FuncAnimation(fig, update, frames=None, blit=True, interval=50)
    plt.show()
    
    # Cleanup when done
    controller.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pokemon_controller.py path/to/pokemon_red.gb")
        sys.exit(1)
    
    rom_path = sys.argv[1]
    run_interactive_demo(rom_path)
