# Setup for Pokémon Red Python Environment

# 1. Install the required packages
# First, create a virtual environment (optional but recommended)
# pip install virtualenv
# virtualenv pokemon_env
# source pokemon_env/bin/activate  # On Windows: pokemon_env\Scripts\activate

# Install PyBoy
pip install pyboy

# 2. Basic script to run Pokémon Red
import sys
import time
from pyboy import PyBoy

# Path to your Pokémon Red ROM (you need to provide this)
ROM_PATH = "path/to/your/pokemon_red.gb"

# Initialize PyBoy
pyboy = PyBoy(ROM_PATH)
pyboy.set_emulation_speed(1)  # 1 = normal speed

# Start the game
pyboy.send_input(pyboy.button_press.BUTTON_START)
time.sleep(0.1)  # Wait for the game to process input
pyboy.tick()

# Main game loop
try:
    while True:
        # Get screen data
        screen = pyboy.botsupport_manager().screen()
        screen_image = screen.screen_ndarray()  # Numpy array of screen pixels
        
        # Access game RAM
        # For example, to get player's X position in the overworld
        # This is just an example, actual addresses will depend on specific game data
        player_x = pyboy.get_memory_value(0xD362)
        player_y = pyboy.get_memory_value(0xD361)
        
        print(f"Player position: ({player_x}, {player_y})")
        
        # Example: Send an input (press the A button)
        # pyboy.send_input(pyboy.button_press.BUTTON_A)
        # time.sleep(0.1)
        # pyboy.tick()
        
        # Advance the emulator by one frame
        pyboy.tick()
        
        # Optional: sleep to slow down the loop
        time.sleep(0.01)
        
except KeyboardInterrupt:
    # Gracefully exit on Ctrl+C
    pyboy.stop()
    sys.exit(0)
