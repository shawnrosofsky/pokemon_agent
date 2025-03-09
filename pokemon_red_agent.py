#!/usr/bin/env python3
"""
Basic example of using PyBoy 2.x.x to play Pokemon Red
"""
import time
import numpy as np
# from pyboy import PyBoy, WindowEvent
from pyboy import PyBoy
from pyboy.utils import WindowEvent

# Path to your Pokemon Red ROM file (you need to provide this legally)
# ROM_PATH = "path/to/your/pokemon_red.gb"
ROM_PATH = "roms/Pokemon Red Version (Colorization)/Pokemon Red Version (Colorization).gb"

def main():
    # Start PyBoy and load the ROM
    # pyboy = PyBoy(ROM_PATH, game_wrapper=True)
    pyboy = PyBoy(ROM_PATH)
    pyboy.set_emulation_speed(1)  # Set to 1 for normal speed
    
    # In PyBoy 2.x.x, booting animation happens automatically
    
    print("Starting Pokemon Red...")
    
    # Game loop
    try:
        running = True
        frame_count = 0
        
        while running:
            # Tick advances the game by one frame
            running = pyboy.tick()
            
            # Get the current game screen as a numpy array
            # game_screen = np.array(pyboy.screen.screen_ndarray())
            game_screen = np.array(pyboy.screen.ndarray)
            
            # Example: Press START button after 5 seconds (300 frames at 60fps)
            # if frame_count == 60 * 5:
            if frame_count == 60 * 10:
                # print(WindowEvent.PRESS_BUTTON_START)
                pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
                # pyboy.button_press('start')
                # pyboy.button('start')
                pyboy.tick()
                pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
                # pyboy.button_release('start')
                # print("Pressed START button")
            
            # Add your agent code here to:
            # 1. Process the game screen
            # 2. Decide on actions
            # 3. Send inputs to the game
            
            frame_count += 1
            # print("Frame count:", frame_count)
    except KeyboardInterrupt:
        print("Stopping emulation...")
    finally:
        # Stop the emulator
        pyboy.stop(save=True)

if __name__ == "__main__":
    main()