#!/usr/bin/env python3
"""
LangGraph Pokemon Agent Runner - Main script to run the LangGraph Pokemon Agent.
Now with LangChain integration for LLM calls.
"""

import os
import asyncio
import argparse
import logging
from datetime import datetime

# Import our modules
from langgraph_pokemon_agent import PokemonAgent
from langgraph_emulator_adapter import GameEmulatorAdapter
from langgraph_knowledge_base import KnowledgeBase
from langgraph_tools_adapter import PokemonToolsAdapter, AsyncToolExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pokemon_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PokemonAgent")


async def main():
    """Main entry point for the LangGraph Pokemon Agent."""
    parser = argparse.ArgumentParser(description='Run a Pokémon AI agent using LangGraph with LangChain')
    parser.add_argument('rom_path', help='Path to the Pokémon ROM file')
    parser.add_argument('--provider', default='claude', choices=['claude', 'openai', 'gemini', 'ollama'], 
                        help='LLM provider to use')
    parser.add_argument('--api-key', help='API key (will use corresponding env var if not provided)')
    parser.add_argument('--model', help='Model to use (will use appropriate default for provider if not specified)')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no window)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for the LLM')
    parser.add_argument('--speed', type=int, default=1, help='Game speed multiplier')
    parser.add_argument('--no-sound', action='store_true', help='Disable game sound')
    parser.add_argument('--log-to-file', action='store_true', help='Log output to file')
    parser.add_argument('--log-file', help='Path to log file (default: pokemon_agent.log)')
    parser.add_argument('--summary-interval', type=int, default=10, help='Number of turns between summaries')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--token-limit', type=int, default=10000, help='Token limit for each message')
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Initialize API key based on provider
    if not args.api_key:
        if args.provider == "claude":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            env_var_name = "ANTHROPIC_API_KEY"
        elif args.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            env_var_name = "OPENAI_API_KEY"
        elif args.provider == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY")
            env_var_name = "GOOGLE_API_KEY"
        else:  # ollama doesn't need API key
            api_key = None
            env_var_name = None
    else:
        api_key = args.api_key
        env_var_name = None
    
    # Check if API key is required and present
    if args.provider in ["claude", "openai", "gemini"] and not api_key:
        logger.error(f"{args.provider.capitalize()} API key required. Set {env_var_name} env var or pass --api-key.")
        return 1
    
    # Set default model based on provider if not specified
    if not args.model:
        if args.provider == "claude":
            model = "claude-3-7-sonnet-20250219"
        elif args.provider == "openai":
            model = "gpt-4o"
        elif args.provider == "gemini":
            model = "gemini-2.0-flash"
        elif args.provider == "ollama":
            model = "llama3.3"
        else:
            model = "claude-3-7-sonnet-20250219"  # Default fallback
    else:
        model = args.model
    
    # Custom log file if provided
    log_file = args.log_file or f"pokemon_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    try:
        logger.info(f"Initializing Pokemon agent with ROM: {args.rom_path}")
        logger.info(f"Using {args.provider.capitalize()} provider with model: {model}")
        
        # Initialize components
        logger.info("Creating emulator adapter")
        emulator = GameEmulatorAdapter(
            args.rom_path, 
            headless=args.headless, 
            speed=args.speed, 
            sound=not args.no_sound
        )
        
        logger.info("Creating knowledge base")
        knowledge_base = KnowledgeBase()
        
        logger.info("Creating tools adapter")
        tools_adapter = PokemonToolsAdapter(emulator, knowledge_base)
        
        logger.info("Creating tool executor")
        tool_executor = AsyncToolExecutor(tools_adapter)
        
        logger.info("Creating Pokemon agent")
        agent = PokemonAgent(
            rom_path=args.rom_path,
            emulator=emulator,  # Pass the emulator directly
            knowledge_base=knowledge_base,  # Pass the knowledge base
            tools_adapter=tools_adapter,  # Pass the tools adapter
            model_name=model,
            api_key=api_key,
            provider=args.provider,
            temperature=args.temperature,
            headless=args.headless,
            speed=args.speed,
            sound=not args.no_sound,
            output_to_file=args.log_to_file,
            log_file=log_file,
            summary_interval=args.summary_interval,
            token_limit=args.token_limit
        )
        
        # Start the game
        logger.info("Starting the game emulator")
        emulator.start_game(skip_intro=True)
        
        # Create stop event for the game loop
        stop_event = asyncio.Event()
        
        # Start game loop in background task
        logger.info("Starting game loop")
        game_loop_task = asyncio.create_task(
            emulator.run_game_loop(fps=60, stop_event=stop_event)
        )
        
        # Start the agent
        logger.info("Starting the agent")
        await agent.start()
        
        # Wait for game loop to complete (should never happen normally)
        await game_loop_task
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Stopping...")
    except Exception as e:
        logger.exception(f"Error running agent: {e}")
    finally:
        # Clean up
        logger.info("Stopping the agent and cleaning up resources")
        
        # Signal the game loop to stop
        if 'stop_event' in locals():
            stop_event.set()
        
        # Stop the agent if it was created
        if 'agent' in locals():
            agent.stop()
        
        # The agent no longer owns the emulator, so we need to close it here
        if 'emulator' in locals():
            emulator.close()
        
        logger.info("Agent stopped. Goodbye!")
    
    return 0


if __name__ == "__main__":
    # Run the main function
    exit_code = asyncio.run(main())
    exit(exit_code)