# LangGraph Pokémon Agent with LangChain Integration

A fully asynchronous agent for playing Pokémon games using LangGraph for orchestration, LangChain for LLM integration, and PyBoy for emulation. This agent leverages multiple LLM providers (Claude, OpenAI, Gemini, Ollama) to make intelligent gameplay decisions while keeping all components running asynchronously.

## Features

- **Fully Asynchronous**: Game continues running while the agent thinks
- **Stateful Agent**: Maintains context and memory across interactions
- **Intelligent Decision Making**: Supports multiple LLM providers via LangChain
- **Advanced Memory Management**: Prevents context duplication with token counting, message filtering, and automatic summarization
- **Modular Architecture**: Clean separation between emulation, knowledge, and agent components
- **Multi-Provider Support**: Compatible with Claude, OpenAI, Gemini, and Ollama

## Architecture Overview

The agent is built with a modular architecture following the design shown in the visual guide:

- **LangGraph Agent**: Core orchestration using LangGraph StateGraph
- **Knowledge Base**: Enhanced knowledge store with cross-thread memory
- **Emulator Adapter**: PyBoy wrapper with asynchronous capabilities
- **Tools Adapter**: Tool definitions and execution for agent actions

## Components

### 1. LangGraph Pokémon Agent (`langgraph_pokemon_agent.py`)

The main agent implementation using LangGraph for orchestration. Features:
- Asynchronous graph execution
- Continuous game loop
- Context management
- Periodic summarization
- Integration with LangChain for multiple LLM providers (Claude, OpenAI, Gemini, Ollama)
- Provider-specific image handling for multimodal capabilities

### 2. Emulator Adapter (`langgraph_emulator_adapter.py`)

Enhanced PyBoy adapter with asynchronous capabilities:
- Button press and wait functionality
- Game state extraction and formatting
- Screen capture and encoding
- Memory access and game information retrieval

### 3. Knowledge Base (`langgraph_knowledge_base.py`)

Enhanced knowledge base with LangGraph's memory capabilities:
- Cross-thread memory for persistence
- Action history tracking
- Map and location information
- Game progress milestones
- Memory search capabilities

### 4. Tools Adapter (`langgraph_tools_adapter.py`)

Tool definitions and execution for agent actions:
- Button press and wait tools
- Knowledge base update tools
- Game information retrieval tools
- Asynchronous tool execution

### 5. Main Runner (`main.py`)

Entry point script that combines all components:
- Command-line argument parsing
- Component initialization and orchestration
- Error handling and cleanup

## Installation

### Prerequisites

- Python 3.9 or higher
- [PyBoy](https://github.com/Baekalfen/PyBoy) emulator
- LangGraph 0.3.18 or newer
- LangChain Core and provider-specific packages
- API key for your chosen LLM provider (Claude, OpenAI, or Gemini)

### Setup

1. Clone this repository
2. Install dependencies:
```bash
pip install langgraph langchain-core langchain-anthropic langchain-openai langchain-google-genai langchain-community pyboy pillow pydantic
```

3. Obtain a Pokémon ROM file (legally!)
4. Set up your API key for your chosen provider:
```bash
# For Claude
export ANTHROPIC_API_KEY=your_api_key_here

# For OpenAI
export OPENAI_API_KEY=your_api_key_here

# For Gemini
export GOOGLE_API_KEY=your_api_key_here

# Ollama doesn't require an API key
```

## Usage

Run the agent with:

```bash
python langgraph_main.py path/to/pokemon.gb [options]
```

### Command-line Options

- `rom_path`: Path to the Pokémon ROM file (required)
- `--provider`: LLM provider to use (choices: claude, openai, gemini, ollama; default: claude)
- `--api-key`: API key for the chosen provider (will use appropriate env var if not provided)
- `--model`: Model to use (defaults based on provider if not specified)
- `--headless`: Run in headless mode (no window)
- `--temperature`: Temperature for the LLM (default: 0.7)
- `--speed`: Game speed multiplier (default: 1)
- `--no-sound`: Disable game sound
- `--log-to-file`: Log output to file
- `--log-file`: Path to log file (default: pokemon_agent_[timestamp].log)
- `--summary-interval`: Number of turns between summaries (default: 10)
- `--debug`: Enable debug logging
- `--token-limit`: Token limit for each message (default: 10000)

## How It Works

1. The `langgraph_main.py` script initializes all components and starts the agent
2. The game emulator runs continuously in the background
3. The LangGraph agent:
   - Observes the game state
   - Makes decisions using the specified LLM provider (Claude, OpenAI, Gemini, or Ollama)
   - Adapts image handling based on the provider's capabilities
   - Executes actions via the tools adapter
   - Periodically summarizes progress to maintain context
   - Runs one complete decision cycle per loop iteration to avoid recursion limits
4. Knowledge is preserved in the knowledge base for future reference

## LangGraph Implementation

The agent leverages LangGraph's features for orchestration:

- **State Management**: Uses `StateGraph` to define agent workflow
- **Memory**: Utilizes `MemorySaver` for persistent state
- **Tool Nodes**: Implements tools as LangGraph nodes
- **Conditional Edges**: Dynamically routes workflow based on agent state

The graph structure follows a cycle:
1. Get current game state
2. Make agent decision (with summarization when needed)
3. Execute action
4. Update state
5. Repeat

## Extending the Agent

### Adding New Tools

Add new tools to `langgraph_tools_adapter.py`:

1. Create a Pydantic model for tool input
2. Implement the tool function in `PokemonToolsAdapter`
3. Add the tool to `build_tool_definitions()`
4. Update `register_with_graph()` to include the new tool

### Modifying Agent Behavior

Adjust the agent's behavior by modifying:

- System prompts in `agent_decision_node()` or `generate_summary_node()`
- Decision extraction in `execute_action_node()`
- Summary interval in agent initialization

## Memory Management

The agent implements several sophisticated strategies to prevent context window overflow and duplication:

### 1. Token Counting

The agent continuously monitors token usage in the context window, using:
```python
def count_tokens(text: str) -> int:
    return len(self.tokenizer.encode(text))
```

### 2. Context Management Node

A dedicated node in the LangGraph structure monitors context size:
```python
def manage_context_node(self, state: GameState) -> GameState:
    # Count tokens in messages
    total_tokens = self.count_message_tokens(messages)
    
    # Determine if context management is needed
    context_management_needed = total_tokens > (self.token_limit * 0.7)
    
    # Update state with this information
    return {...}
```

### 3. Message Filtering

When sending messages to the LLM, the agent filters the conversation history:
```python
def filter_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Keep only the most relevant messages
    filtered_messages = []
    
    # Keep system messages, then recent exchanges
    # ...
    
    return filtered_messages
```

### 4. Periodic Summarization

The agent periodically creates summaries of gameplay to replace detailed history:
```python
def generate_summary_node(self, state: GameState) -> GameState:
    # Create summary
    # ...
    
    # Clean up old messages
    new_messages = [
        {"role": "system", "content": f"Previous gameplay summary: {summary_text}"}
    ]
    new_messages.extend(messages_to_keep)
```

### 5. Short-term vs. Long-term Memory

- **Short-term Memory**: Handled via LangGraph's `MemorySaver`, maintaining recent context
- **Long-term Memory**: Managed in the `KnowledgeBase` for persistent game information

## License

This project is released under the MIT License.

## Credits

- [LangGraph](https://github.com/langchain-ai/langgraph) for the agent orchestration framework
- [LangChain](https://www.langchain.com/) for the LLM integration capabilities
- [PyBoy](https://github.com/Baekalfen/PyBoy) for the Game Boy emulator
- [Anthropic](https://www.anthropic.com/), [OpenAI](https://openai.com/), [Google](https://deepmind.google/technologies/gemini/), and [Ollama](https://ollama.com/) for their LLM APIs