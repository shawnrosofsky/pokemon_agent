import sys
import time
import numpy as np
import cv2
from collections import deque
import random
from pyboy import PyBoy
from pyboy.utils import WindowEvent

class PokemonAIAgent:
    """
    A simple AI agent for Pokémon Red that can:
    1. Navigate the overworld
    2. Handle random encounters
    3. Make simple battle decisions
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
    
    # Important memory addresses
    MEMORY = {
        'player_x': 0xD362,
        'player_y': 0xD361,
        'map_id': 0xD35E,
        'battle_type': 0xD057,
        'menu_open': 0xD730,
        'battle_menu_cursor': 0xCC28,  # Position of cursor in battle menu
        'battle_turn': 0xCCD3,         # Whose turn it is in battle
        'text_progress': 0xC6AC,       # Text box progress indicator
    }
    
    def __init__(self, rom_path, headless=False, save_frames=False):
        """Initialize the AI agent with a ROM path."""
        window_type = "headless" if headless else "SDL2"
        self.pyboy = PyBoy(rom_path, window_type=window_type)
        self.pyboy.set_emulation_speed(0)  # Max speed
        
        self.last_position = None
        self.stuck_counter = 0
        self.last_moves = deque(maxlen=10)
        self.save_frames = save_frames
        self.frame_count = 0
        
        # For handling battles
        self.in_battle_last_frame = False
        self.battle_state = "none"
        
    def start(self):
        """Start the game and skip intro."""
        print("Booting game...")
        
        # Advance frames to start
        for _ in range(200):
            self.pyboy.tick()
        
        # Press start a few times to navigate through intro screens
        for _ in range(5):
            self.press_button('start')
            for _ in range(20):
                self.pyboy.tick()
        
        print("Game started!")
    
    def press_button(self, button, hold_frames=10):
        """Press a button for a specified number of frames."""
        if button not in self.BUTTONS:
            print(f"Unknown button: {button}")
            return
            
        # Press the button
        self.pyboy.send_input(self.BUTTONS[button])
        
        # Hold for specified frames
        for _ in range(hold_frames):
            self.pyboy.tick()
            self.capture_frame()
            
        # Release the button
        self.pyboy.send_input(self.RELEASE_BUTTONS[button])
        self.pyboy.tick()
        self.capture_frame()
    
    def capture_frame(self):
        """Capture the current frame if save_frames is enabled."""
        if self.save_frames:
            screen = self.get_screen()
            cv2.imwrite(f"frames/frame_{self.frame_count:06d}.png", screen[:, :, ::-1])
            self.frame_count += 1
    
    def get_screen(self):
        """Get the current screen as a numpy array."""
        screen = self.pyboy.botsupport_manager().screen()
        return screen.screen_ndarray()
    
    def get_memory_value(self, address):
        """Get the value at a memory address."""
        return self.pyboy.get_memory_value(address)
    
    def get_player_position(self):
        """Get the player's position as (x, y)."""
        x = self.get_memory_value(self.MEMORY['player_x'])
        y = self.get_memory_value(self.MEMORY['player_y'])
        return (x, y)
    
    def is_in_battle(self):
        """Check if the player is in a battle."""
        return self.get_memory_value(self.MEMORY['battle_type']) != 0
    
    def is_text_being_displayed(self):
        """Check if text is currently being displayed/processed."""
        return self.get_memory_value(self.MEMORY['text_progress']) != 0
    
    def handle_overworld(self):
        """Handle movement in the overworld."""
        current_pos = self.get_player_position()
        
        # Check if we're stuck in the same position
        if self.last_position == current_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_position = current_pos
        
        # If we're stuck, try different movement patterns
        if self.stuck_counter > 20:
            print(f"Stuck at position {current_pos}, trying random movement")
            # Try a random direction
            move = random.choice(['up', 'down', 'left', 'right'])
            self.press_button(move)
            
            # If very stuck, try a more complex sequence
            if self.stuck_counter > 50:
                print("Very stuck, trying to interact...")
                self.press_button('a')  # Try to interact with something
                
                # If nothing helps, try talking to menu
                if self.stuck_counter > 100:
                    print("Extremely stuck, trying menu...")
                    self.press_button('start')
                    time.sleep(0.5)
                    self.press_button('b')
                    self.stuck_counter = 0
        else:
            # Normal movement - avoid immediately reversing direction
            possible_moves = ['up', 'down', 'left', 'right']
            
            # Prefer not to go back the way we came
            if len(self.last_moves) > 0:
                last_move = self.last_moves[-1]
                opposite = {
                    'up': 'down',
                    'down': 'up',
                    'left': 'right',
                    'right': 'left'
                }
                if last_move in opposite:
                    # Lower probability of reversing
                    if random.random() > 0.2:
                        possible_moves.remove(opposite[last_move])
            
            move = random.choice(possible_moves)
            self.last_moves.append(move)
            self.press_button(move)
    
    def handle_battle(self):
        """Handle a Pokémon battle with simple heuristics."""
        # Detect transition into battle
        if not self.in_battle_last_frame:
            print("Battle started!")
            self.battle_state = "start"
            self.in_battle_last_frame = True
            return
        
        # Simple state machine for battle
        if self.battle_state == "start":
            # Wait for battle intro to finish, then usually select "FIGHT"
            self.press_button('a')
            if random.random() > 0.8:
                # Occasionally use an item or run
                self.press_button('right')
                self.battle_state = "menu"
            else:
                self.battle_state = "fight"
        
        elif self.battle_state == "fight":
            # Select the first move most of the time
            self.press_button('a')
            if random.random() > 0.7:
                # Sometimes select a different move
                self.press_button('down')
                
            self.battle_state = "executing"
        
        elif self.battle_state == "menu" or self.battle_state == "executing":
            # Just keep pressing A to progress through text
            if self.is_text_being_displayed():
                self.press_button('a')
            else:
                # If no text is being displayed, we might be back at a menu
                cursor_pos = self.get_memory_value(self.MEMORY['battle_menu_cursor'])
                if cursor_pos != 0:
                    self.battle_state = "start"
    
    def handle_text(self):
        """Handle text boxes by repeatedly pressing A."""
        # Keep pressing A to advance text
        self.press_button('a')
        time.sleep(0.1)  # Small delay
    
    def run(self, max_steps=1000):
        """Run the AI agent for a specified number of steps."""
        # Create directory for frames if needed
        if self.save_frames:
            import os
            os.makedirs("frames", exist_ok=True)
        
        try:
            for step in range(max_steps):
                # Check game state
                in_battle = self.is_in_battle()
                text_displayed = self.is_text_being_displayed()
                
                if step % 100 == 0:
                    print(f"Step {step}, Position: {self.get_player_position()}, Battle: {in_battle}")
                
                # Handle different game states
                if in_battle:
                    self.handle_battle()
                elif text_displayed:
                    self.handle_text()
                else:
                    self.handle_overworld()
                    self.in_battle_last_frame = False
                
                # Advance a few frames
                for _ in range(5):
                    self.pyboy.tick()
                    self.capture_frame()
                
                # Optional: Save game state periodically
                if step % 500 == 0 and step > 0:
                    self.pyboy.save_state(f"save_state_{step}.state")
                    
        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            # Clean up
            self.pyboy.stop()
            print("Agent stopped")
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pokemon_ai_agent.py path/to/pokemon_red.gb [--headless] [--save-frames]")
        sys.exit(1)
    
    rom_path = sys.argv[1]
    headless = "--headless" in sys.argv
    save_frames = "--save-frames" in sys.argv
    
    agent = PokemonAIAgent(rom_path, headless, save_frames)
    agent.start()
    agent.run(max_steps=5000)
