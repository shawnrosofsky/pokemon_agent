"""
LLM Factory Module - Creates appropriate LLM client based on provider and configuration.
"""

import os
from typing import Dict, Any, Optional

# Import all client implementations
from claude_client import AsyncClaudeClient
from openai_client import AsyncOpenAIClient
from gemini_client import AsyncGeminiClient
from ollama_client import AsyncOllamaClient


def create_llm_client(provider="claude", config=None, output_manager=None):
    """
    Factory function to create the appropriate LLM client.
    
    Args:
        provider: The LLM provider to use ('claude', 'openai', 'gemini', 'ollama')
        config: Dictionary with configuration options
        output_manager: OutputManager instance for logging
        
    Returns:
        An AsyncLLMClient instance for the specified provider
    """
    if config is None:
        config = {}
    
    # Default configurations
    defaults = {
        "claude": {
            "model": "claude-3-7-sonnet-20250219",
            "temperature": 0.7,
            "api_key": os.environ.get("ANTHROPIC_API_KEY", "")
        },
        "openai": {
            "model": "gpt-4o",
            "temperature": 0.7,
            "api_key": os.environ.get("OPENAI_API_KEY", "")
        },
        "gemini": {
            "model": "gemini-2.0-flash",
            "temperature": 0.7,
            "api_key": os.environ.get("GOOGLE_API_KEY", "")
        },
        "ollama": {
            "model": "llama3.3_70b",
            "temperature": 0.7,
            "api_base": "http://localhost:11434"
        }
    }
    
    # Merge provided config with defaults
    provider = provider.lower()
    if provider in defaults:
        for key, value in defaults[provider].items():
            if key not in config:
                config[key] = value
    
    # Create the appropriate client
    if provider == "claude":
        return AsyncClaudeClient(
            api_key=config["api_key"],
            model=config["model"],
            temperature=config["temperature"],
            output_manager=output_manager
        )
    elif provider == "openai":
        return AsyncOpenAIClient(
            api_key=config["api_key"],
            model=config["model"],
            temperature=config["temperature"],
            output_manager=output_manager
        )
    elif provider == "gemini":
        return AsyncGeminiClient(
            api_key=config["api_key"],
            model=config["model"],
            temperature=config["temperature"],
            output_manager=output_manager
        )
    elif provider == "ollama":
        return AsyncOllamaClient(
            model=config["model"],
            temperature=config["temperature"],
            api_base=config["api_base"],
            output_manager=output_manager
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


# Example usage
if __name__ == "__main__":
    # Simple demonstration of creating different clients
    print("LLM Factory Test")
    
    # Create an Claude client (default)
    claude_client = create_llm_client("claude")
    print(f"Created Claude client using model: {claude_client.model}")
    
    # Create an OpenAI client
    openai_client = create_llm_client("openai", {"model": "gpt-4o-mini"})
    print(f"Created OpenAI client using model: {openai_client.model}")
    
    # Create a Gemini client
    gemini_client = create_llm_client("gemini")
    print(f"Created Gemini client using model: {gemini_client.model}")
    
    # Create an Ollama client
    ollama_client = create_llm_client("ollama", {"model": "llama3.3_70b"})
    print(f"Created Ollama client using model: {ollama_client.model}")