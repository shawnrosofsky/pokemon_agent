# Setup for Pokémon Red Python Environment

# 1. Install the required packages
# First, create a virtual environment (optional but recommended)
# pip install virtualenv
# virtualenv pokemon_env
# source pokemon_env/bin/activate  # On Windows: pokemon_env\Scripts\activate

# Install PyBoy
# pip install pyboy

# 2. Basic script to run Pokémon Red
import sys
import time
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent

# Path to your Pokémon Red ROM (you need to provide this)
# ROM_PATH = "path/to/your/pokemon_red.gb"
ROM_PATH = "roms/Pokemon Red Version (Colorization)/Pokemon Red Version (Colorization).gb"

# Initialize PyBoy
# pyboy = PyBoy(ROM_PATH, game_wrapper=True)
pyboy = PyBoy(ROM_PATH)
pyboy.set_emulation_speed(1)  # 1 = normal speed

# Start the game
# pyboy.button_press(WindowEvent.PRESS_BUTTON_START)
pyboy.button('start')
time.sleep(0.1)  # Wait for the game to process input
pyboy.tick(1, True)

# Main game loop
try:
    while True:
        # Get screen data
        screen_image = np.array(pyboy.screen.ndarray)  # Numpy array of screen pixels
        
        # Access game RAM
        # For example, to get player's X position in the overworld
        # This is just an example, actual addresses will depend on specific game data
        player_x = pyboy.memory[0xD362]
        player_y = pyboy.memory[0xD361]
        
        print(f"Player position: ({player_x}, {player_y})")
        
        # Example: Send an input (press the A button)
        # pyboy.button_press(WindowEvent.PRESS_BUTTON_A)
        # time.sleep(0.1)
        # pyboy.tick(1, True)
        
        # Advance the emulator by one frame
        pyboy.tick(1, True)
        
        # Optional: sleep to slow down the loop
        time.sleep(0.01)
        
except KeyboardInterrupt:
    # Gracefully exit on Ctrl+C
    pyboy.stop()
    sys.exit(0)
