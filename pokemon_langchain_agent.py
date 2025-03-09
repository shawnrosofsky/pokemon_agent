import os
import time
import json
from typing import Dict, List, Any, Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic.chat_models import ChatAnthropic

from pokemon_llm_agent_base import PokemonLLMAgentBase, PokemonGameInterface

class PokemonLangChainAgent(PokemonLLMAgentBase):
    """
    Pokémon agent implementation using LangChain.
    """
    
    def __init__(self, game_interface, model_name="claude-3-opus-20240229", 
                 api_key=None, temperature=0.2, observation_interval=30):
        """
        Initialize the LangChain-based agent.
        
        Args:
            game_interface (PokemonGameInterface): Interface to the Pokémon game
            model_name (str): Model name to use
            api_key (str, optional): API key, will use environment variable if not provided
            temperature (float): Temperature for the model
            observation_interval (int): Number of frames between observations
        """
        super().__init__(game_interface, observation_interval)
        
        # Set up API key
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Either pass it directly or set ANTHROPIC_API_KEY env variable.")
        
        # Set up model and chain
        self.model_name = model_name
        self.temperature = temperature
        
        # Create the LLM
        self.llm = ChatAnthropic(
            model_name=self.model_name,
            anthropic_api_key=self.api_key,
            temperature=self.temperature
        )
        
        # System message for consistent behavior
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

Your response should ONLY contain one of these actions and nothing else.
"""
        
        # Create message history
        self.message_history: List[Any] = [
            SystemMessage(content=self.system_message)
        ]
    
    def get_llm_response(self, prompt, screen_base64):
        """
        Get a response from the LLM via LangChain.
        
        Args:
            prompt (str): The prompt to send to the LLM
            screen_base64 (str): Base64 encoded screen image
            
        Returns:
            str: Response from the LLM
        """
        try:
            # Create human message with image
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screen_base64}"
                }
            }
            
            # Create a composite message with both text and image
            human_message = HumanMessage(
                content=[image_content, {"type": "text", "text": prompt}]
            )
            
            # Add message to history
            self.message_history.append(human_message)
            
            # Get response with memory
            with get_openai_callback() as cb:
                response = self.llm.invoke(self.message_history)
                
                # Print token usage for monitoring
                print(f"Tokens used: {cb.total_tokens}")
                
            # Add AI response to history
            self.message_history.append(response)
            
            # Limit history to manage context size
            if len(self.message_history) > 10:  # Keep system message + last 9 exchanges
                # Always keep the system message
                self.message_history = [self.message_history[0]] + self.message_history[-9:]
            
            return response.content
            
        except Exception as e:
            print(f"Error calling LLM via LangChain: {e}")
            # Return a default action if the API call fails
            return "wait"
    
    def run(self, num_steps=None):
        """
        Run the agent for a specified number of steps or indefinitely.
        
        Args:
            num_steps (int, optional): Number of steps to run, or None for indefinite
        """
        step_count = 0
        action_stats = {}  # Keep track of action frequency
        
        try:
            print("Starting LangChain-based Pokémon agent...")
            
            # Main loop
            while num_steps is None or step_count < num_steps:
                action, observation, response_time = self.observe_and_act()
                
                if action:  # Only increment if we took an action
                    step_count += 1
                    
                    # Update action statistics
                    action_stats[action] = action_stats.get(action, 0) + 1
                    
                    # Print progress and stats
                    print(f"Step {step_count}: {action} -> {observation} (response time: {response_time:.2f}s)")
                    
                    if step_count % 10 == 0:
                        # Print action distribution
                        print("Action distribution:")
                        total = sum(action_stats.values())
                        for act, count in sorted(action_stats.items(), key=lambda x: x[1], reverse=True):
                            print(f"  {act}: {count} ({count/total*100:.1f}%)")
                    
                    # Save state periodically
                    if step_count % 50 == 0:
                        save_path = self.game.save_state(f"langchain_step_{step_count}.state")
                        print(f"Saved game state to {save_path}")
                        
                        # Also save current stats
                        with open(f"{self.game.save_dir}/action_stats_{step_count}.json", 'w') as f:
                            json.dump(action_stats, f)
                
                # Small delay to prevent maxing out CPU
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Error running agent: {e}")
        finally:
            # Save final state and history
            self.game.save_state("langchain_final.state")
            history_path = self.game.save_history("langchain_history.json")
            print(f"Saved final state and history to {history_path}")
            
            # Save action statistics
            with open(f"{self.game.save_dir}/action_stats_final.json", 'w') as f:
                json.dump(action_stats, f)
            
            # Clean up
            self.game.close()


# Example usage
if __name__ == "__main__":
    import sys
    import argparse

    def main():
        parser = argparse.ArgumentParser(description='Run a Pokémon AI agent using LangChain')
        parser.add_argument('rom_path', help='Path to the Pokémon ROM file')
        parser.add_argument('--api-key', help='API key (will use ANTHROPIC_API_KEY env var if not provided)')
        parser.add_argument('--model', default='claude-3-opus-20240229', help='Model to use')
        parser.add_argument('--headless', action='store_true', help='Run in headless mode (no window)')
        parser.add_argument('--steps', type=int, help='Number of steps to run (infinite if not specified)')
        parser.add_argument('--interval', type=int, default=30, help='Observation interval in frames')
        parser.add_argument('--temperature', type=float, default=0.2, help='Model temperature')

        args = parser.parse_args()

        # Create the game interface
        game = PokemonGameInterface(args.rom_path, headless=args.headless, speed=1)
        
        try:
            # Start the game
            game.start_game(skip_intro=True)
            
            # Create and run the agent
            agent = PokemonLangChainAgent(
                game_interface=game,
                model_name=args.model,
                api_key=args.api_key,
                temperature=args.temperature,
                observation_interval=args.interval
            )
            
            agent.run(num_steps=args.steps)
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            game.close()

    if __name__ == "__main__":
        main()
