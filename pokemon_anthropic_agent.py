import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run a Pokémon AI agent using Anthropic API')
    parser.add_argument('rom_path', help='Path to the Pokémon ROM file')
    parser.add_argument('--api-key', help='Anthropic API key (will use ANTHROPIC_API_KEY env var if not provided)')
    parser.add_argument('--model', default='claude-3-opus-20240229', help='Anthropic model to use')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no window)')
    parser.add_argument('--steps', type=int, help='Number of steps to run (infinite if not specified)')
    parser.add_argument('--interval', type=int, default=30, help='Observation interval in frames')

    args = parser.parse_args()

    # Create the game interface
    game = PokemonGameInterface(args.rom_path, headless=args.headless, speed=1)
    
    try:
        # Start the game
        game.start_game(skip_intro=True)
        
        # Create and run the agent
        agent = PokemonAnthropicAgent(
            game_interface=game,
            model=args.model,
            api_key=args.api_key,
            observation_interval=args.interval
        )
        
        agent.run(num_steps=args.steps)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        game.close()

if __name__ == "__main__":
    main() os
import time
import json
from typing import Optional, Dict, Any
import anthropic
from anthropic.types.message import Image

from pokemon_llm_agent_base import PokemonLLMAgentBase, PokemonGameInterface

class PokemonAnthropicAgent(PokemonLLMAgentBase):
    """
    Pokémon agent implementation using Anthropic API.
    """
    
    def __init__(self, game_interface, model="claude-3-opus-20240229", api_key=None, 
                 max_tokens=1000, observation_interval=30):
        """
        Initialize the Anthropic-based agent.
        
        Args:
            game_interface (PokemonGameInterface): Interface to the Pokémon game
            model (str): Anthropic model to use
            api_key (str, optional): Anthropic API key, will use environment variable if not provided
            max_tokens (int): Maximum tokens to generate in the response
            observation_interval (int): Number of frames between observations
        """
        super().__init__(game_interface, observation_interval)
        
        # Set up Anthropic client
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Either pass it directly or set ANTHROPIC_API_KEY env variable.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        
        # System message to guide the LLM's behavior
        self.system_message = """
You are an intelligent agent playing Pokémon Red. Your goal is to make progress in the game by exploring, 
battling, and completing objectives. You need to choose actions based on the current game state.

Follow these guidelines:
1. Focus on making forward progress in the game
2. In battles, make strategic decisions based on your Pokémon's health and the opponent
3. When faced with obstacles, try different approaches
4. Navigate menus efficiently to achieve your objectives
5. Respond only with a single action from the available actions list

The available actions are:
- press_up: Move up
- press_down: Move down
- press_left: Move left
- press_right: Move right
- press_a: Interact/Select/Confirm
- press_b: Cancel/Back
- press_start: Open menu
- press_select: Secondary menu
- wait: Wait for a few frames

Do not explain your reasoning, just respond with the single action.
"""
    
    def get_llm_response(self, prompt, screen_base64):
        """
        Get a response from the Anthropic API.
        
        Args:
            prompt (str): The prompt to send to the API
            screen_base64 (str): Base64 encoded screen image
            
        Returns:
            str: Response from the API
        """
        try:
            # Create a message with both text and image
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_message,
                messages=[
                    {
                        "role": "user",
                        "content": [
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
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            # Extract and return the text response
            return message.content[0].text
            
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            # Return a default action if the API call fails
            return "wait"
    
    def run(self, num_steps=None):
        """
        Run the agent for a specified number of steps or indefinitely.
        
        Args:
            num_steps (int, optional): Number of steps to run, or None for indefinite
        """
        step_count = 0
        try:
            print("Starting Anthropic-based Pokémon agent...")
            
            # Main loop
            while num_steps is None or step_count < num_steps:
                action, observation, response_time = self.observe_and_act()
                
                if action:  # Only increment if we took an action
                    step_count += 1
                    print(f"Step {step_count}: {action} -> {observation} (response time: {response_time:.2f}s)")
                
                    # Save state periodically
                    if step_count % 50 == 0:
                        save_path = self.game.save_state(f"anthropic_step_{step_count}.state")
                        print(f"Saved game state to {save_path}")
                
                # Small delay to prevent maxing out CPU
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Error running agent: {e}")
        finally:
            # Save final state and history
            self.game.save_state("anthropic_final.state")
            history_path = self.game.save_history("anthropic_history.json")
            print(f"Saved final state and history to {history_path}")
            
            # Clean up
            self.game.close()


# Example usage
if __name__ == "__main__":
    import