# Pokémon LLM Agent

An AI agent that plays Pokémon Red/Blue using LLMs (Claude, GPT-4, Gemini, or local models via Ollama) with asynchronous API calls through direct APIs or LangChain.

## Features

- Support for multiple LLM providers:
  - Claude (Anthropic)
  - GPT-4/4o (OpenAI)
  - Gemini (Google)
  - Local models via Ollama
- Option to use LangChain as a unified framework
- Fully asynchronous processing (the game continues running during API calls)
- Automatic memory management and summarization
- Complete game state tracking
- Detailed action statistics
- Game state saving/loading

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. You will need a Pokémon Red/Blue ROM (not included)

## API Keys

Set up your API keys as environment variables:

```bash
export ANTHROPIC_API_KEY="your_anthropic_key_here"
export OPENAI_API_KEY="your_openai_key_here"
export GOOGLE_API_KEY="your_google_key_here"
```

Alternatively, you can provide them as command-line arguments.

## Usage

Basic usage with Claude:

```bash
python pokemon_agent_multi.py path/to/pokemon_red.gb
```

Using LangChain with OpenAI:

```bash
python pokemon_agent_multi.py path/to/pokemon_red.gb --provider openai --use-langchain
```

Using local models with Ollama:

```bash
python pokemon_agent_multi.py path/to/pokemon_red.gb --provider ollama --api-base "http://localhost:11434"
```

## Command-Line Options

- `--provider`: LLM provider to use (claude, openai, gemini, ollama)
- `--api-key`: API key (will use env var if not provided)
- `--model`: Specific model name
- `--headless`: Run without display
- `--steps`: Number of steps to run
- `--temperature`: Temperature setting (0.0-1.0)
- `--speed`: Game speed multiplier
- `--no-sound`: Disable game sound
- `--log-to-file`: Save output to file
- `--log-file`: Path to log file
- `--summary-interval`: Turns between summaries
- `--api-base`: API base URL (for Ollama)
- `--use-langchain`: Use LangChain framework

## Architecture

The agent uses an asynchronous architecture:

1. The game runs continuously
2. LLM API calls happen in the background
3. When an API response is received, the action is executed

This ensures smooth gameplay even when API calls take time.

## Memory Management

The agent manages its memory through:

1. Maintaining a limited conversation history
2. Periodic summarization to condense game progress
3. Knowledge base tracking for long-term memory

## LangChain Integration

The LangChain integration provides:

1. A unified interface for all models
2. Compatibility with the existing architecture
3. Access to LangChain's tools and capabilities

To enable LangChain, add the `--use-langchain` flag.

## License

MIT License - See LICENSE file for details.