#!/usr/bin/env python3
"""
Main script to run Pokémon agents
"""

import os
import sys
import argparse
from pokemon_llm_agent_base import PokemonGameInterface
from pokemon_anthropic_agent import PokemonAnthropicAgent
from pokemon_langchain_agent import PokemonLangChainAgent
from pokemon_langgraph_agent import PokemonLangGraphAgent
from pokemon_langgraph_agent_advanced import PokemonLangGraphAgent as PokemonLangGraphAgentAdvanced
from pokemon_trainer import run_agent, setup_experiment_dir

def main():
    parser = argparse.ArgumentParser(description='Run Pokémon AI agents powered by LLMs')
    
    # Required arguments
    parser.add_argument('rom_path', help='Path to the Pokémon ROM file')
    
    # Agent configuration
    agent_group = parser.add_argument_group('Agent Configuration')
    agent_group.add_argument('--agent', choices=['anthropic', 'langchain', 'langgraph', 'langgraph_advanced'], default='anthropic',
                           help='Which type of agent to use (default: anthropic)')
    agent_group.add_argument('--api-key', help='API key (if not provided, uses ANTHROPIC_API_KEY env var)')
    agent_group.add_argument('--model', default='claude-3-7-sonnet-20250219',
                           help='Model name to use (default: claude-3-7-sonnet-20250219)')
    agent_group.add_argument('--temperature', type=float, default=0.2,
                           help='Temperature setting for the model (default: 0.2)')
    
    # Game configuration
    game_group = parser.add_argument_group('Game Configuration')
    game_group.add_argument('--headless', action='store_true',
                          help='Run in headless mode with no window (faster)')
    game_group.add_argument('--speed', type=int, default=1,
                          help='Game speed multiplier (default: 1)')
    game_group.add_argument('--skip-intro', action='store_true', default=True,
                          help='Skip the game intro sequence (default: True)')
    
    # Run configuration
    run_group = parser.add_argument_group('Run Configuration')
    run_group.add_argument('--steps', type=int,
                         help='Number of steps to run (if not specified, runs indefinitely)')
    run_group.add_argument('--observation-interval', type=int, default=30,
                         help='Frames between observations (default: 30)')
    run_group.add_argument('--exp-dir', default='experiments',
                         help='Base directory for experiments (default: experiments)')
    run_group.add_argument('--mode', choices=['interactive', 'experiment'], default='interactive',
                         help='Run mode: interactive or experiment (default: interactive)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check for API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: No API key provided. Either use --api-key or set the ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)
    
    if args.mode == 'experiment':
        # Run in experiment mode with metrics collection
        exp_dir = setup_experiment_dir(args.exp_dir)
        print(f"Running experiment, results will be saved to: {exp_dir}")
        
        summary = run_agent(
            agent_type=args.agent,
            rom_path=args.rom_path,
            api_key=api_key,
            model=args.model,
            steps=args.steps or 100,  # Default to 100 steps for experiments
            exp_dir=exp_dir,
            headless=args.headless
        )
        
        print("\nExperiment complete!")
        print(f"Results saved to: {exp_dir}")
        
    else:
        # Run in interactive mode
        print(f"Starting {args.agent} agent in interactive mode")
        
        # Create the game interface
        game = PokemonGameInterface(
            rom_path=args.rom_path,
            headless=args.headless,
            speed=args.speed
        )
        
        try:
            # Start the game
            game.start_game(skip_intro=args.skip_intro)
            
            # Create and run the appropriate agent
            if args.agent == 'anthropic':
                agent = PokemonAnthropicAgent(
                    game_interface=game,
                    model=args.model,
                    api_key=api_key,
                    observation_interval=args.observation_interval
                )
            elif args.agent == 'langchain':
                agent = PokemonLangChainAgent(
                    game_interface=game,
                    model_name=args.model,
                    api_key=api_key,
                    temperature=args.temperature,
                    observation_interval=args.observation_interval
                )
            elif args.agent == 'langgraph_advanced':
                # From existing pokemon_trainer.py
                agent = PokemonLangGraphAgentAdvanced(
                    game_interface=game,
                    model_name=args.model,
                    api_key=api_key,
                    observation_interval=args.observation_interval,
                    checkpoint_dir=os.path.join(exp_dir, 'checkpoints')
                )
            else:  # langgraph
                agent = PokemonLangGraphAgent(
                    game_interface=game,
                    model_name=args.model,
                    api_key=api_key,
                    temperature=args.temperature,
                    observation_interval=args.observation_interval,
                    checkpoint_dir=os.path.join(args.exp_dir, 'checkpoints')
                )
            
            # Run the agent
            agent.run(num_steps=args.steps)
            
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            game.close()

if __name__ == "__main__":
    main()
