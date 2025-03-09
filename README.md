# Pokémon LLM Agent

A Python framework for building AI agents that can play Pokémon Red/Blue using Large Language Models (LLMs).

## Overview

This project enables LLM-powered agents to play Pokémon by:

1. Running the game through a Game Boy emulator (PyBoy)
2. Capturing the game screen and state at regular intervals
3. Sending this information to an LLM (Claude) via API calls
4. Having the LLM decide on the best action to take
5. Executing the chosen action in the game

The framework provides a flexible base for experimenting with different LLM implementations, including direct Anthropic API calls and LangChain-based approaches.

## Key Features

- **Game Interface**: Provides access to game screen, memory, and controls
- **LLM Agent Base Class**: Abstract base class for implementing different LLM agents
- **Multiple Implementations**: 
  - Anthropic API-based agent using raw API calls
  - LangChain-based agent for more complex workflows
  - LangGraph-based agent for stateful, multi-step reasoning
- **Experimentation Framework**: Run experiments, collect metrics, and visualize agent performance
- **Memory Mapping**: Predefined memory addresses for accessing critical game state information

## Requirements

- Python 3.8+
- PyBoy (Game Boy emulator with Python API)
- Anthropic API key for Claude
- Pokémon Red/Blue ROM file (must be legally obtained)
- Additional libraries: langchain, PIL, numpy, matplotlib (for visualization)

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install pyboy langchain langchain_anthropic langgraph anthropic numpy matplotlib pillow
```

3. Set up your API key:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

Run the agent using the direct Anthropic API:

```bash
python pokemon_main.py path/to/pokemon_red.gb --agent anthropic --steps 100
```

Run the agent using LangChain:

```bash
python pokemon_main.py path/to/pokemon_red.gb --agent langchain --steps 100
```

Run the agent using LangGraph:

```bash
python pokemon_main.py path/to/pokemon_red.gb --agent langgraph --steps 100
```

### Advanced Options

```bash
python pokemon_main.py path/to/pokemon_red.gb \
  --agent anthropic \
  --model claude-3-sonnet-20240229 \
  --headless \
  --speed 2 \
  --steps 500 \
  --observation-interval 20 \
  --mode experiment
```

### Running Experiments

To run a full experiment with metrics collection and visualization:

```bash
python pokemon_trainer.py path/to/pokemon_red.gb --agent anthropic --steps 200
```

This will:
1. Create an experiment directory
2. Run the agent for the specified number of steps
3. Collect metrics like response time and game progress
4. Save screenshots at each step
5. Generate performance plots
6. Create a timelapse GIF of gameplay

## Project Structure

- `pokemon_llm_agent_base.py`: Core classes for game interaction and agent base
- `pokemon_anthropic_agent.py`: Anthropic API implementation
- `pokemon_langchain_agent.py`: LangChain implementation
- `pokemon_langgraph_agent.py`: LangGraph implementation with multi-step reasoning
- `pokemon_trainer.py`: Framework for running experiments and collecting metrics
- `pokemon_main.py`: Command-line interface for running agents

## Game State Information

The agents can access detailed game state information, including:

- Player position and map ID
- Pokémon party information (count, HP, levels)
- Battle status (active, enemy Pokémon)
- UI state (text boxes, menus)
- Progress indicators (badges, money)

## Extending the Framework

### Creating a New Agent Implementation

You can create new agent implementations by subclassing `PokemonLLMAgentBase` and implementing the required methods:

```python
class MyCustomAgent(PokemonLLMAgentBase):
    def __init__(self, game_interface, ...):
        super().__init__(game_interface)
        # Your initialization code
    
    def get_llm_response(self, prompt, screen_base64):
        # Your LLM implementation
        pass
    
    def run(self, num_steps=None):
        # Your run implementation
        pass
```

### Adding Custom Memory Mappings

To add more memory addresses for game state tracking:

```python
# Add to MEMORY_MAP in PokemonGameInterface
self.MEMORY_MAP.update({
    'my_custom_address': 0xD123,
    'another_address': 0xC456,
})
```

## License

MIT License