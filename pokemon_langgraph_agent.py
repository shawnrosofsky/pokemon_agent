import os
import time
import json
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Annotated
import uuid

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
import langgraph.checkpoint as checkpoint
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_anthropic import ChatAnthropic

# Local imports
from pokemon_llm_agent_base import PokemonLLMAgentBase, PokemonGameInterface

# Define our state type
class PokemonAgentState(TypedDict):
    """State for the Pokémon LangGraph agent."""
    # The current observation
    observation: str
    # The current game state as text
    game_state: str
    # The current screen as base64
    screen: str
    # The conversation history
    messages: List[Any]
    # The next action to take
    next_action: Optional[str]
    # Additional context
    context: Dict[str, Any]


class PokemonGameTool(BaseModel):
    """Tool for interacting with the Pokémon game."""
    action: str = Field(description="The action to take in the game")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the action in the game."""
        # This will be filled in by the actual agent implementation
        # We just define the interface here
        return {"result": f"Executed action: {self.action}"}


class PokemonLangGraphAgent(PokemonLLMAgentBase):
    """
    Pokémon agent implementation using LangGraph.
    """
    
    def __init__(self, game_interface, model_name="claude-3-7-sonnet-20250219", 
                 api_key=None, temperature=0.2, observation_interval=30,
                 checkpoint_dir="langgraph_checkpoints"):
        """
        Initialize the LangGraph-based agent.
        
        Args:
            game_interface (PokemonGameInterface): Interface to the Pokémon game
            model_name (str): Model name to use
            api_key (str, optional): API key, will use environment variable if not provided
            temperature (float): Temperature for the model
            observation_interval (int): Number of frames between observations
            checkpoint_dir (str): Directory to save checkpoints
        """
        super().__init__(game_interface, observation_interval)
        
        # Set up API key
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Either pass it directly or set ANTHROPIC_API_KEY env variable.")
        
        # Set up model and settings
        self.model_name = model_name
        self.temperature = temperature
        
        # Create the LLM
        self.llm = ChatAnthropic(
            model_name=self.model_name,
            anthropic_api_key=self.api_key,
            temperature=self.temperature
        )
        
        # Set up checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Set up LangGraph checkpointing
        checkpoint.configure(
            dir=self.checkpoint_dir
        )
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Statistics for tracking
        self.action_stats = {}
        
    def _build_graph(self):
        """Build the LangGraph workflow."""
        # Define the state graph
        graph_builder = StateGraph(PokemonAgentState)
        
        # Add the nodes to the graph
        graph_builder.add_node("analyze_game_state", self._analyze_game_state)
        graph_builder.add_node("decide_action", self._decide_action)
        graph_builder.add_node("execute_action", self._execute_action)
        
        # Set up the edges in the graph
        graph_builder.add_edge("analyze_game_state", "decide_action")
        graph_builder.add_edge("decide_action", "execute_action")
        graph_builder.add_edge("execute_action", END)
        
        # Set the entry point
        graph_builder.set_entry_point("analyze_game_state")
        
        # Compile the graph
        graph = graph_builder.compile()
        
        return graph
    
    def _analyze_game_state(self, state: PokemonAgentState) -> PokemonAgentState:
        """Analyze the current game state."""
        # Create the system message for analysis
        system_message = SystemMessage(content="""
You are an expert Pokémon game analyst. Your job is to analyze the current game state and provide a detailed assessment.
Focus on:
1. The player's current location and surroundings
2. Battle status if in a battle
3. Player's Pokémon team status
4. Any visible text or menus
5. Possible obstacles or objectives

Provide a clear and concise analysis that will help decide the next best action.
""")
        
        # Create the human message with the observation and screen
        human_message = HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{state['screen']}"
                }
            },
            {
                "type": "text",
                "text": f"""
Please analyze the current game state:
{state['observation']}

Game State Details:
{state['game_state']}

Based on what you see, what's happening in the game right now?
"""
            }
        ])
        
        # Get a response from the LLM
        messages = [system_message, human_message]
        response = self.llm.invoke(messages)
        
        # Update the state with the analysis
        state["messages"] = [system_message, human_message, response]
        state["context"]["analysis"] = response.content
        
        return state
    
    def _decide_action(self, state: PokemonAgentState) -> PokemonAgentState:
        """Decide on the next action to take."""
        # Create the system message for decision making
        system_message = SystemMessage(content="""
You are an expert Pokémon game player. Your job is to decide the next best action to take in the game.
Available actions:
- press_up: Move up
- press_down: Move down
- press_left: Move left
- press_right: Move right
- press_a: Interact/Select/Confirm
- press_b: Cancel/Back
- press_start: Open menu
- press_select: Secondary menu
- wait: Wait for a few frames

Respond with EXACTLY ONE of these actions, with no additional text or explanation.
""")
        
        # Create a human message with the analysis
        human_message = HumanMessage(content=f"""
Based on the following analysis of the game state, what action should I take next?

Analysis:
{state["context"]["analysis"]}

Choose ONE action from the available actions list.
""")
        
        # Get a response from the LLM
        messages = state["messages"] + [system_message, human_message]
        response = self.llm.invoke(messages)
        
        # Extract the action from the response
        action = self._parse_llm_response(response.content)
        
        # Update the state with the decision
        state["next_action"] = action
        state["messages"] = messages + [response]
        
        return state
    
    def _execute_action(self, state: PokemonAgentState) -> PokemonAgentState:
        """Execute the decided action in the game."""
        action = state["next_action"]
        
        # Execute the action in the game
        result = self.execute_action(action)
        
        # Update the action stats
        self.action_stats[action] = self.action_stats.get(action, 0) + 1
        
        # Create a tool message with the result
        tool_message = ToolMessage(content=result, name="game_action")
        
        # Update the state with the result
        state["messages"].append(tool_message)
        state["context"]["last_action"] = action
        state["context"]["last_result"] = result
        
        return state
    
    def get_llm_response(self, prompt, screen_base64):
        """
        Get a response from the LLM via LangGraph.
        
        Args:
            prompt (str): The prompt to send to the LLM
            screen_base64 (str): Base64 encoded screen image
            
        Returns:
            str: Response from the LLM (the next action to take)
        """
        try:
            # Get the game state
            state = self.game.get_game_state()
            
            # Format the game state
            state_text = self.format_game_state(state)
            
            # Create a new conversation thread ID
            thread_id = str(uuid.uuid4())
            
            # Create the initial state
            initial_state: PokemonAgentState = {
                "observation": prompt,
                "game_state": state_text,
                "screen": screen_base64,
                "messages": [],
                "next_action": None,
                "context": {
                    "frame": self.game.frame_count,
                    "thread_id": thread_id,
                }
            }
            
            # Run the graph
            final_state = self.graph.invoke(initial_state, {"configurable": {"thread_id": thread_id}})
            
            # Return the next action
            return final_state["next_action"]
            
        except Exception as e:
            print(f"Error in LangGraph workflow: {e}")
            # Return a default action if the workflow fails
            return "wait"
    
    def run(self, num_steps=None):
        """
        Run the agent for a specified number of steps or indefinitely.
        
        Args:
            num_steps (int, optional): Number of steps to run, or None for indefinite
        """
        step_count = 0
        
        try:
            print("Starting LangGraph-based Pokémon agent...")
            
            # Main loop
            while num_steps is None or step_count < num_steps:
                action, observation, response_time = self.observe_and_act()
                
                if action:  # Only increment if we took an action
                    step_count += 1
                    
                    # Print progress and stats
                    print(f"Step {step_count}: {action} -> {observation} (response time: {response_time:.2f}s)")
                    
                    if step_count % 10 == 0:
                        # Print action distribution
                        print("Action distribution:")
                        total = sum(self.action_stats.values())
                        for act, count in sorted(self.action_stats.items(), key=lambda x: x[1], reverse=True):
                            print(f"  {act}: {count} ({count/total*100:.1f}%)")
                    
                    # Save state periodically
                    if step_count % 50 == 0:
                        save_path = self.game.save_state(f"langgraph_step_{step_count}.state")
                        print(f"Saved game state to {save_path}")
                        
                        # Also save current stats
                        with open(f"{self.game.save_dir}/action_stats_{step_count}.json", 'w') as f:
                            json.dump(self.action_stats, f)
                
                # Small delay to prevent maxing out CPU
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Error running agent: {e}")
        finally:
            # Save final state and history
            self.game.save_state("langgraph_final.state")
            history_path = self.game.save_history("langgraph_history.json")
            print(f"Saved final state and history to {history_path}")
            
            # Save action statistics
            with open(f"{self.game.save_dir}/action_stats_final.json", 'w') as f:
                json.dump(self.action_stats, f)
            
            # Clean up
            self.game.close()


# Example usage
if __name__ == "__main__":
    import sys
    import argparse

    def main():
        parser = argparse.ArgumentParser(description='Run a Pokémon AI agent using LangGraph')
        parser.add_argument('rom_path', help='Path to the Pokémon ROM file')
        parser.add_argument('--api-key', help='API key (will use ANTHROPIC_API_KEY env var if not provided)')
        parser.add_argument('--model', default='claude-3-7-sonnet-20250219', help='Model to use')
        parser.add_argument('--headless', action='store_true', help='Run in headless mode (no window)')
        parser.add_argument('--steps', type=int, help='Number of steps to run (infinite if not specified)')
        parser.add_argument('--interval', type=int, default=30, help='Observation interval in frames')
        parser.add_argument('--temperature', type=float, default=0.2, help='Model temperature')
        parser.add_argument('--checkpoint-dir', default='langgraph_checkpoints', 
                          help='Directory to save checkpoints')

        args = parser.parse_args()

        # Create the game interface
        game = PokemonGameInterface(args.rom_path, headless=args.headless, speed=1)
        
        try:
            # Start the game
            game.start_game(skip_intro=True)
            
            # Create and run the agent
            agent = PokemonLangGraphAgent(
                game_interface=game,
                model_name=args.model,
                api_key=args.api_key,
                temperature=args.temperature,
                observation_interval=args.interval,
                checkpoint_dir=args.checkpoint_dir
            )
            
            agent.run(num_steps=args.steps)
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            game.close()

    if __name__ == "__main__":
        main()
