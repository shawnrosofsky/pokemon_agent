#!/usr/bin/env python3
"""
Pokémon Trainer - A script to train and evaluate LLM agents playing Pokémon Red/Blue
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil

# Import the game interface and agent classes
from pokemon_llm_agent_base import PokemonGameInterface
from pokemon_anthropic_agent import PokemonAnthropicAgent
from pokemon_langchain_agent import PokemonLangChainAgent
from pokemon_langgraph_agent import PokemonLangGraphAgent

def setup_experiment_dir(base_dir="experiments"):
    """
    Set up a directory for experiment results.
    
    Args:
        base_dir (str): Base directory for experiments
        
    Returns:
        str: Path to the experiment directory
    """
    # Create a timestamp-based directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    
    # Create directories
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "states"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "screenshots"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "metrics"), exist_ok=True)
    
    return exp_dir

def record_metrics(metrics, exp_dir, step):
    """
    Record metrics to a file.
    
    Args:
        metrics (dict): Metrics to record
        exp_dir (str): Experiment directory
        step (int): Current step number
    """
    metrics_file = os.path.join(exp_dir, "metrics", f"metrics_{step}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

def plot_metrics(metrics_dir, output_file):
    """
    Plot metrics from JSON files.
    
    Args:
        metrics_dir (str): Directory containing metrics files
        output_file (str): Output file for the plot
    """
    # Collect all metrics files
    metrics_files = sorted([f for f in os.listdir(metrics_dir) if f.startswith("metrics_") and f.endswith(".json")])
    
    steps = []
    response_times = []
    progress_scores = []
    
    for file in metrics_files:
        with open(os.path.join(metrics_dir, file), 'r') as f:
            data = json.load(f)
            steps.append(data.get("step", 0))
            response_times.append(data.get("response_time", 0))
            progress_scores.append(data.get("progress_score", 0))
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Response time plot
    ax1.plot(steps, response_times, 'b-', label='Response Time (s)')
    ax1.set_ylabel('Response Time (s)')
    ax1.set_title('LLM Agent Performance Metrics')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Progress score plot
    ax2.plot(steps, progress_scores, 'g-', label='Progress Score')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Progress Score')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def create_timelapse_gif(screenshots_dir, output_file, fps=10):
    """
    Create a timelapse GIF from screenshots.
    
    Args:
        screenshots_dir (str): Directory containing screenshots
        output_file (str): Output file for the GIF
        fps (int): Frames per second for the GIF
    """
    # Collect all screenshots
    screenshots = sorted([f for f in os.listdir(screenshots_dir) if f.endswith(".png")])
    
    if not screenshots:
        print("No screenshots found to create timelapse.")
        return
    
    # Load images
    images = []
    for filename in screenshots:
        img = Image.open(os.path.join(screenshots_dir, filename))
        images.append(img)
    
    # Save as GIF
    # Save every 5th frame to keep file size reasonable
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1::5],  # Take every 5th frame
        optimize=False,
        duration=1000//fps,  # Duration in ms
        loop=0
    )

def calculate_progress_score(game_state):
    """
    Calculate a progress score based on the game state.
    Higher score indicates more progress in the game.
    
    Args:
        game_state (dict): The current game state
        
    Returns:
        float: A progress score
    """
    score = 0
    
    # Basic score based on badges
    if 'player' in game_state and 'badges' in game_state['player']:
        badges = game_state['player']['badges']
        badge_count = badges.count('1')
        score += badge_count * 50
    
    # Score based on Pokémon party
    if 'party' in game_state and 'count' in game_state['party']:
        pokemon_count = game_state['party']['count']
        score += pokemon_count * 10
        
        # Bonus for Pokémon levels
        if pokemon_count > 0 and 'first_pokemon' in game_state['party']:
            first_pokemon = game_state['party']['first_pokemon']
            if first_pokemon and 'level' in first_pokemon:
                score += first_pokemon['level'] * 2
    
    # Score based on money
    if 'player' in game_state and 'money' in game_state['player']:
        score += game_state['player']['money'] / 1000
    
    return score

def run_agent(agent_type, rom_path, api_key, model, steps, exp_dir, headless=False):
    """
    Run an agent and collect metrics.
    
    Args:
        agent_type (str): Type of agent to run ('anthropic' or 'langchain')
        rom_path (str): Path to the ROM file
        api_key (str): API key
        model (str): Model to use
        steps (int): Number of steps to run
        exp_dir (str): Experiment directory
        headless (bool): Whether to run in headless mode
        
    Returns:
        dict: Summary metrics
    """
    # Configure game interface to save to experiment directory
    game = PokemonGameInterface(rom_path, headless=headless, speed=1)
    game.save_dir = os.path.join(exp_dir, "states")
    
    # Start the game
    game.start_game(skip_intro=True)
    
    # Create the agent
    if agent_type.lower() == 'anthropic':
        agent = PokemonAnthropicAgent(
            game_interface=game,
            model=model,
            api_key=api_key,
            observation_interval=30
        )
    elif agent_type.lower() == 'langchain':
        agent = PokemonLangChainAgent(
            game_interface=game,
            model_name=model,
            api_key=api_key,
            observation_interval=30
        )
    elif agent_type.lower() == 'langgraph':
        agent = PokemonLangGraphAgent(
            game_interface=game,
            model_name=model,
            api_key=api_key,
            observation_interval=30,
            checkpoint_dir=os.path.join(exp_dir, 'checkpoints')
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    print(f"Running {agent_type} agent for {steps} steps...")
    
    # Run the agent and collect metrics
    step_count = 0
    total_response_time = 0
    action_stats = {}
    start_time = time.time()
    
    try:
        while step_count < steps:
            action, observation, response_time = agent.observe_and_act()
            
            if action:  # Only increment if we took an action
                step_count += 1
                total_response_time += response_time
                
                # Update action statistics
                action_stats[action] = action_stats.get(action, 0) + 1
                
                # Save screenshot
                screen = game.get_screen_as_pil()
                screen.save(os.path.join(exp_dir, "screenshots", f"step_{step_count:04d}.png"))
                
                # Calculate progress score
                game_state = game.get_game_state()
                progress_score = calculate_progress_score(game_state)
                
                # Record metrics
                metrics = {
                    "step": step_count,
                    "action": action,
                    "response_time": response_time,
                    "progress_score": progress_score,
                    "elapsed_time": time.time() - start_time,
                    "frame": game.frame_count,
                    "player_position": game_state['player']['position'],
                    "map_id": game_state['player']['map'],
                    "in_battle": game_state['battle']['active']
                }
                
                record_metrics(metrics, exp_dir, step_count)
                
                # Print progress
                print(f"Step {step_count}/{steps}: {action} -> {observation} (time: {response_time:.2f}s)")
                
                # Save game state every 10 steps
                if step_count % 10 == 0:
                    game.save_state(f"step_{step_count:04d}.state")
            
            # Small delay to prevent maxing out CPU
            time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error running agent: {e}")
    finally:
        # Save final state and history
        game.save_state("final.state")
        history_path = game.save_history("history.json")
        
        # Save action statistics
        with open(os.path.join(exp_dir, "metrics", "action_stats.json"), 'w') as f:
            json.dump(action_stats, f, indent=2)
        
        # Create summary
        elapsed_time = time.time() - start_time
        avg_response_time = total_response_time / max(1, step_count)
        
        summary = {
            "agent_type": agent_type,
            "model": model,
            "steps_completed": step_count,
            "total_elapsed_time": elapsed_time,
            "avg_response_time": avg_response_time,
            "action_distribution": action_stats
        }
        
        with open(os.path.join(exp_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create plots
        plot_metrics(
            os.path.join(exp_dir, "metrics"),
            os.path.join(exp_dir, "performance_plot.png")
        )
        
        # Create timelapse
        create_timelapse_gif(
            os.path.join(exp_dir, "screenshots"),
            os.path.join(exp_dir, "timelapse.gif")
        )
        
        # Clean up
        game.close()
        
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate LLM agents playing Pokémon')
    parser.add_argument('rom_path', help='Path to the Pokémon ROM file')
    parser.add_argument('--agent', choices=['anthropic', 'langchain', 'langgraph'], default='anthropic', 
                      help='Type of agent to use')
    parser.add_argument('--api-key', help='API key (will use ANTHROPIC_API_KEY env var if not provided)')
    parser.add_argument('--model', default='claude-3-7-sonnet-20250219', help='Model to use')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no window)')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to run')
    parser.add_argument('--exp-dir', default='experiments', help='Base directory for experiments')
    
    args = parser.parse_args()
    
    # Set up experiment directory
    exp_dir = setup_experiment_dir(args.exp_dir)
    print(f"Experiment results will be saved to: {exp_dir}")
    
    # Save experiment configuration
    config = vars(args)
    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run the agent
    summary = run_agent(
        agent_type=args.agent,
        rom_path=args.rom_path,
        api_key=args.api_key,
        model=args.model,
        steps=args.steps,
        exp_dir=exp_dir,
        headless=args.headless
    )
    
    print("\nExperiment complete!")
    print(f"Summary: {json.dumps(summary, indent=2)}")
    print(f"Results saved to: {exp_dir}")
