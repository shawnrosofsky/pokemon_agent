"""
LangGraph Pokémon Agent - A fully asynchronous agent for playing Pokémon
using LangGraph for orchestration and PyBoy for emulation.
With enhanced memory management to prevent context duplication.
Now using LangChain for LLM calls.
"""

import os
import asyncio
import base64
import time
import uuid
import tiktoken
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, TypedDict, Annotated, Literal
from typing_extensions import TypedDict, NotRequired

# Replace anthropic direct import with langchain imports
from PIL import Image
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph import MessagesState
from langchain_core.messages import RemoveMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate


# Langchain model integrations
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# Import our modules
from langgraph_emulator_adapter import GameEmulatorAdapter
from langgraph_knowledge_base import KnowledgeBase
from langgraph_tools_adapter import PokemonToolsAdapter


class GameState(TypedDict):
    """State representation for the Pokemon game agent."""
    # Game-related state
    screen_base64: str
    game_state_text: str
    recent_actions: str
    last_error: Optional[str]
    
    # Agent-related state
    messages: List[Dict[str, Any]]
    action_count: int
    summary_interval: int
    summary_due: bool
    last_summary: str
    
    # Memory management
    total_tokens: Optional[int]
    token_limit: Optional[int]
    context_management_needed: Optional[bool]


class OutputManager:
    """Manages output of agent actions and thoughts."""
    
    def __init__(self, output_to_file=False, log_file=None):
        self.output_to_file = output_to_file
        self.log_file = log_file or "pokemon_agent.log"
        
        # Create log file if needed
        if self.output_to_file:
            with open(self.log_file, 'w') as f:
                f.write(f"=== Pokémon Agent Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    def print(self, text, end="\n"):
        """Print text to console and optionally to file."""
        print(text, end=end)
            
        if self.output_to_file:
            with open(self.log_file, 'a') as f:
                f.write(text)
                if end:
                    f.write(end)
    
    def print_section(self, title, content=None):
        """Print a formatted section with title."""
        separator = "="*50
        self.print(f"\n{separator}")
        self.print(f"{title}")
        self.print(f"{separator}")
        
        if content:
            self.print(content)
            self.print(separator)


class PokemonAgent:
    """LLM Agent for playing Pokémon using LangGraph and PyBoy."""
    
    def __init__(
        self, 
        rom_path: str,
        emulator: GameEmulatorAdapter,  # Using the emulator passed from main
        knowledge_base: KnowledgeBase,  # Using the knowledge base passed from main
        tools_adapter: PokemonToolsAdapter,  # Using the tools adapter passed from main
        model_name: str = "claude-3-7-sonnet-20250219", 
        api_key: Optional[str] = None,
        provider: str = "claude",  # Provider can be "claude", "openai", "gemini", "ollama"
        temperature: float = 0.7, 
        headless: bool = False, 
        speed: int = 1,
        sound: bool = True, 
        output_to_file: bool = False, 
        log_file: Optional[str] = None, 
        summary_interval: int = 10,
        token_limit: int = 8000  # Default token limit for context window
    ):
        """Initialize the agent."""
        # Setup API
        self.api_key = api_key
        if not self.api_key:
            # Try to get API key from environment based on provider
            if provider == "claude":
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY")
            elif provider == "gemini":
                self.api_key = os.environ.get("GOOGLE_API_KEY")
                
        # For Claude, OpenAI, and Gemini, we need an API key
        if provider in ["claude", "openai", "gemini"] and not self.api_key:
            raise ValueError(f"{provider.capitalize()} API key required. Set appropriate environment variable or pass directly.")
        
        # Setup output manager
        self.output = OutputManager(output_to_file=output_to_file, log_file=log_file)
        
        # Use passed components instead of creating new ones
        self.rom_path = rom_path
        self.emulator = emulator
        self.kb = knowledge_base
        self.tools_manager = tools_adapter
        
        # Agent settings
        self.provider = provider.lower()
        self.model_name = model_name  # Store original model name
        self.temperature = temperature
        self.summary_interval = summary_interval
        self.token_limit = token_limit
        
        # Initialize tokenizer for token counting (approximation using tiktoken)
        # For most models, cl100k_base is a reasonable approximation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Create LangChain model
        self.llm = self._create_langchain_model()
        
        # Setup async event loop and state
        self.running = False
        self.paused = False
        self.target_fps = 60
        self.frame_time = 1.0 / self.target_fps
        
        # Initialize LangGraph components - will be set up in build_graph()
        self.memory = None
        self.graph = None
        self.thread_id = str(uuid.uuid4())
        self.action_count = 0
        
        # Create the graph structure
        self.build_graph()
    
    def _create_langchain_model(self) -> BaseChatModel:
        """Create the appropriate LangChain model based on the provider."""
        if self.provider == "claude":
            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                anthropic_api_key=self.api_key,
                max_tokens=4000,
            )
        
        elif self.provider == "openai":
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=self.api_key,
                max_tokens=4000,
            )
        
        elif self.provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=self.api_key,
                max_output_tokens=4000,
            )
        
        elif self.provider == "ollama":
            return ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
            )
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string using the tokenizer."""
        return len(self.tokenizer.encode(text))
    
    def count_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count the total tokens in a list of messages."""
        total = 0
        
        for message in messages:
            # Count tokens in message content (could be string or list of content blocks)
            content = message.get("content", "")
            
            if isinstance(content, str):
                total += self.count_tokens(content)
            elif isinstance(content, list):
                # For content blocks (text, image, etc.)
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total += self.count_tokens(block.get("text", ""))
                    # Note: We don't count tokens for images here
            
            # Add a small overhead for message structure
            total += 5
        
        return total
    
    def build_graph(self):
        """Build the LangGraph structure for the agent."""
        self.output.print_section("BUILDING LANGGRAPH AGENT")
        
        # Create a memory saver for persistence
        self.memory = MemorySaver()
        
        # Create the graph with our state type
        builder = StateGraph(GameState)
        
        # Add nodes to the graph
        builder.add_node("get_game_state", self.get_game_state_node)
        builder.add_node("manage_context", self.manage_context_node)
        builder.add_node("agent_decision", self.agent_decision_node)
        builder.add_node("execute_action", self.execute_action_node)
        builder.add_node("generate_summary", self.generate_summary_node)
        builder.add_node("update_state", self.update_state_node)
        
        # Create conditional edges
        builder.add_edge(START, "get_game_state")
        builder.add_edge("get_game_state", "manage_context")
        
        # Conditional edge from manage_context 
        builder.add_conditional_edges(
            "manage_context",
            self.check_if_context_management_needed,
            {
                "needs_summary": "generate_summary",
                "continue": "agent_decision",
            }
        )
        
        # Conditional edge from agent_decision
        builder.add_conditional_edges(
            "agent_decision",
            self.check_if_summary_needed,
            {
                "needs_summary": "generate_summary",
                "continue": "execute_action",
            }
        )
        
        builder.add_edge("generate_summary", "execute_action")
        builder.add_edge("execute_action", "update_state")
        builder.add_edge("update_state", "get_game_state")
        
        # Compile the graph with the checkpoint saver
        self.graph = builder.compile(checkpointer=self.memory)
        
        self.output.print("Graph built successfully!")
    
    def get_game_state_node(self, state: GameState) -> GameState:
        """Node to get the current game state from the emulator."""
        # Get game state
        game_state = self.emulator.get_game_state()
        screen_base64 = self.emulator.get_screen_base64()
        state_text = self.emulator.format_game_state(game_state)
        recent_actions = self.kb.get_recent_actions()
        
        # Update the state
        return {
            **state,
            "screen_base64": screen_base64,
            "game_state_text": state_text,
            "recent_actions": recent_actions
        }
    
    def filter_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter messages to keep the context window manageable.
        Keeps the most relevant messages for decision-making.
        """
        # If messages are few, no need to filter
        if len(messages) <= 5:
            return messages
        
        # Otherwise, implement a filtering strategy:
        # 1. Keep the first message if it's a system message
        # 2. Keep recent user messages with game state
        # 3. Keep recent assistant decisions
        
        filtered_messages = []
        
        # Keep the first message if it's a system message
        if messages and messages[0].get("role") == "system":
            filtered_messages.append(messages[0])
        
        # Add the last N messages
        keep_last_n = 4  # Keep last 4 message pairs (user+assistant)
        last_messages = messages[-min(len(messages), keep_last_n*2):]
        filtered_messages.extend(last_messages)
        
        # If token count is still too high, we'll need to implement more aggressive filtering
        token_count = self.count_message_tokens(filtered_messages)
        if token_count > self.token_limit * 0.8:
            # Only keep the most recent interactions
            filtered_messages = filtered_messages[-2:] if len(filtered_messages) > 2 else filtered_messages
        
        return filtered_messages
    
    def agent_decision_node(self, state: GameState) -> GameState:
        """Node to get the agent's decision on what action to take."""
        self.output.print_section("AGENT DECISION")
        
        # Import LangChain's structured output tools
        from langchain.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.output_parsers import StructuredOutputParser
        from langchain_core.pydantic_v1 import BaseModel, Field

        # Define a schema for the response
        class PokemonAction(BaseModel):
            """Action to take in the Pokemon game."""
            analysis: str = Field(description="Your analysis of the current game state and situation")
            action: str = Field(description="The button to press: up, down, left, right, a, b, start, select, or wait")
            hold_frames: int = Field(default=10, description="Number of frames to hold the button (1-30)")
                
            def __str__(self):
                return f"Analysis: {self.analysis}\nAction: {self.action}\nHold Frames: {self.hold_frames}"
        
        # Create output parser
        parser = JsonOutputParser(pydantic_object=PokemonAction)
        
        # Create system prompt template
        system_template = """
You are an expert Pokémon player. Analyze the game state and decide on the best action to take.

First, think step-by-step about:
1. Where the player is currently and what's visible on screen
2. What progress has been made in the game so far
3. What the immediate goal should be
4. Available actions and their potential outcomes

Then, decide on the best action to take. Be specific about which button to press and for how long.
Use one of these actions only: up, down, left, right, a, b, start, select, or wait if you need to wait for something to happen.

{format_instructions}
"""

        # Get the last summary if available
        last_summary = state.get("last_summary", "")
        if last_summary:
            system_template += f"\n\nRecent Game Progress Summary:\n{last_summary}"

        # Format instructions for the parser
        format_instructions = parser.get_format_instructions()
        
        # Create the system message with format instructions
        system_message = SystemMessage(content=system_template.format(format_instructions=format_instructions))
        
        # Prepare the image content for LangChain
        # Different providers handle image data differently
        image_str = state["screen_base64"]
        
        # Create the human message with game state info
        human_message_text = f"""
Current game state:
{state['game_state_text']}

Recent actions:
{state['recent_actions']}

{f"Note about previous action: {state['last_error']}" if state.get('last_error') else ""}

First, analyze what you see on screen and the current situation.
Then, decide on the best action to take. Make sure to format your response according to the JSON schema.
"""
        
        # Create or extend the messages list in the state
        messages = state.get("messages", [])
        
        # Convert previous messages to LangChain format
        langchain_messages = self._convert_messages_to_langchain(messages)
        
        # Add system message at the beginning if not already present
        if not langchain_messages or not isinstance(langchain_messages[0], SystemMessage):
            langchain_messages.insert(0, system_message)
        
        # Create appropriate human message with image based on provider
        if self.provider in ["claude", "openai", "gemini"]:
            # These providers support images in the messages
            human_content = []
            
            # Add image content
            if self.provider == "claude":
                # Claude format
                human_content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_str
                        }
                    }
                )
            else:
                # OpenAI and Gemini format
                human_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_str}"
                        }
                    }
                )
            
            # Add text content
            human_content.append(
                {
                    "type": "text",
                    "text": human_message_text
                }
            )
            
            human_message = HumanMessage(content=human_content)
        else:
            # Fallback for providers that don't support images directly
            # Just use text content (image will be omitted)
            human_message = HumanMessage(content=f"[Image shown to agent]\n\n{human_message_text}")
        
        # Add the current human message to the list
        langchain_messages.append(human_message)
        
        # Filter messages to prevent context overflow
        filtered_messages = self._filter_langchain_messages(langchain_messages)
        
        # Log the filtering
        if len(filtered_messages) < len(langchain_messages):
            self.output.print(f"Filtered messages: {len(langchain_messages)} → {len(filtered_messages)}")
        
        # Make the API call
        self.output.print(f"Calling {self.provider.capitalize()} model for action decision...")
        try:
            response = self.llm.invoke(filtered_messages)
            
            # Extract the decision text
            decision_text = response.content
            
            # Try to parse the structured output
            try:
                # Parse the structured output using the JsonOutputParser
                parsed_action = parser.parse(decision_text)
                
                # Add parsed action to state
                action_info = {
                    "action": parsed_action["action"].lower(),
                    "hold_frames": parsed_action["hold_frames"],
                    "analysis": parsed_action["analysis"]
                }
                
                self.output.print(f"Successfully parsed structured output: {parsed_action['action']} for {parsed_action['hold_frames']} frames")
                
                # Format for display
                formatted_decision = f"## Game State Analysis\n\n{parsed_action['analysis']}\n\n## Best Action\n\n**Action: {parsed_action['action']}**\n\nHold for {parsed_action['hold_frames']} frames"
                
                # Log the decision with formatted display
                self.output.print_section(f"{self.provider.upper()} DECISION", formatted_decision)
                
                # Add the AI response to the message history - using the formatted version for clarity
                ai_message = AIMessage(content=formatted_decision)
                
            except Exception as parse_error:
                # If parsing fails, use the raw response
                self.output.print(f"Failed to parse structured output: {str(parse_error)}")
                self.output.print("Using raw response instead")
                
                # Log the raw decision
                self.output.print_section(f"{self.provider.upper()} DECISION", decision_text)
                
                # Add the original response to the message history
                ai_message = AIMessage(content=decision_text)
                
                # Set default action info
                action_info = {
                    "action": "wait",  # Default to wait
                    "hold_frames": 10,
                    "analysis": decision_text
                }
                
                # Try to extract action from text as fallback
                import re
                action_match = re.search(r'Action:\s*(\w+)', decision_text, re.IGNORECASE)
                if action_match:
                    action = action_match.group(1).lower()
                    if action in ["up", "down", "left", "right", "a", "b", "start", "select", "wait"]:
                        action_info["action"] = action
                        self.output.print(f"Extracted action from text: {action}")
            
            # Add the message to the history
            langchain_messages.append(ai_message)
            
            # Convert back to the state format
            updated_messages = self._convert_langchain_to_state_messages(langchain_messages)
            
            # Update the state
            return {
                **state,
                "messages": updated_messages,
                "decision_text": decision_text,
                "action_count": state.get("action_count", 0) + 1,
                "action_info": action_info
            }
        except Exception as e:
            self.output.print_section("ERROR", f"Error calling LLM: {str(e)}")
            # Return state unchanged on error
            return {
                **state,
                "action_count": state.get("action_count", 0) + 1,
                "last_error": f"LLM error: {str(e)}",
                "action_info": {
                    "action": "wait",  # Default to wait on error
                    "hold_frames": 10,
                    "analysis": f"Error occurred: {str(e)}"
                }
            }
    
    def _convert_messages_to_langchain(self, state_messages: List[Dict[str, Any]]) -> List[Any]:
        """Convert messages from state format to LangChain format."""
        langchain_messages = []
        
        for msg in state_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                # Handle system messages
                if isinstance(content, str):
                    langchain_messages.append(SystemMessage(content=content))
                else:
                    # If content is a list or dict, convert to string
                    langchain_messages.append(SystemMessage(content=str(content)))
            
            elif role == "user":
                # Handle user messages
                if isinstance(content, str):
                    langchain_messages.append(HumanMessage(content=content))
                elif isinstance(content, list):
                    # Handle content list (may include images)
                    if self.provider in ["claude", "openai", "gemini"]:
                        # These providers support structured content with images
                        langchain_messages.append(HumanMessage(content=content))
                    else:
                        # For other providers, extract text only
                        text_content = ""
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_content += item.get("text", "")
                            elif isinstance(item, str):
                                text_content += item
                        langchain_messages.append(HumanMessage(content=text_content))
            
            elif role == "assistant":
                # Handle assistant messages
                if isinstance(content, str):
                    langchain_messages.append(AIMessage(content=content))
                elif isinstance(content, list):
                    # Handle content list
                    text_content = ""
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content += item.get("text", "")
                        elif isinstance(item, str):
                            text_content += item
                    langchain_messages.append(AIMessage(content=text_content))
        
        return langchain_messages
    
    def _convert_langchain_to_state_messages(self, langchain_messages: List[Any]) -> List[Dict[str, Any]]:
        """Convert messages from LangChain format to state format."""
        state_messages = []
        
        for msg in langchain_messages:
            if isinstance(msg, SystemMessage):
                state_messages.append({
                    "role": "system",
                    "content": msg.content
                })
            elif isinstance(msg, HumanMessage):
                # Handle complex content (like images)
                if isinstance(msg.content, list):
                    state_messages.append({
                        "role": "user",
                        "content": msg.content
                    })
                else:
                    state_messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": msg.content}]
                    })
            elif isinstance(msg, AIMessage):
                state_messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": msg.content}]
                })
        
        return state_messages
    
    def _filter_langchain_messages(self, messages: List[Any]) -> List[Any]:
        """Filter LangChain messages to keep context window manageable."""
        # If messages are few, no need to filter
        if len(messages) <= 5:
            return messages
        
        filtered_messages = []
        
        # Keep the first message if it's a system message
        if messages and isinstance(messages[0], SystemMessage):
            filtered_messages.append(messages[0])
        
        # Add the last N messages
        keep_last_n = 4  # Keep last 4 message pairs (user+assistant)
        last_messages = messages[-min(len(messages), keep_last_n*2):]
        filtered_messages.extend(last_messages)
        
        return filtered_messages
    
    def manage_context_node(self, state: GameState) -> GameState:
        """
        Node to manage context window size.
        This node analyzes the current state and determines if context pruning is needed.
        """
        messages = state.get("messages", [])
        
        # Skip if no messages yet
        if not messages:
            return {
                **state,
                "context_management_needed": False,
                "total_tokens": 0
            }
        
        # Count tokens in the messages
        total_tokens = self.count_message_tokens(messages)
        
        # Determine if context management is needed
        # Trigger if token count exceeds 70% of the limit
        context_management_needed = total_tokens > (self.token_limit * 0.7)
        
        # Log token usage
        if total_tokens > 0:
            self.output.print(f"Current token usage: {total_tokens} tokens " +
                             f"({(total_tokens/self.token_limit*100):.1f}% of limit)")
        
        # Update the state
        return {
            **state,
            "total_tokens": total_tokens,
            "token_limit": self.token_limit,
            "context_management_needed": context_management_needed
        }
    
    def check_if_context_management_needed(self, state: GameState) -> Literal["needs_summary", "continue"]:
        """Check if context management is needed based on token usage."""
        context_management_needed = state.get("context_management_needed", False)
        
        if context_management_needed:
            self.output.print_section("CONTEXT MANAGEMENT", 
                                     f"Token usage ({state.get('total_tokens', 0)}) approaching limit. Generating summary.")
            return "needs_summary"
        
        return "continue"
    
    def check_if_summary_needed(self, state: GameState) -> Literal["needs_summary", "continue"]:
        """Check if a summary is needed before continuing."""
        action_count = state.get("action_count", 0)
        
        if action_count % self.summary_interval == 0 and action_count > 0:
            self.output.print(f"Summary needed after {action_count} actions")
            return "needs_summary"
        
        return "continue"
    
    def generate_summary_node(self, state: GameState) -> GameState:
        """
        Generate a summary of recent progress and clean up conversation history.
        This node handles both summarization of game progress and context management.
        """
        self.output.print_section("GENERATING SUMMARY")
        
        # System prompt for summarization
        system_prompt = """
You are a Pokémon game expert. Provide a concise summary of the player's progress.
Focus on:
1. Current location and objective
2. Recent battles and encounters
3. Party status
4. Progress towards game goals

Your summary should be comprehensive but concise (200-300 words maximum).
"""
        
        # Create the message for summarization
        game_state = self.emulator.get_game_state()
        state_text = self.emulator.format_game_state(game_state)
        
        message_content = f"""
Please summarize the recent game progress:

Current game state:
{state_text}

Recent actions:
{self.kb.get_recent_actions(20)}

Previous summary (if any):
{state.get('last_summary', '')}

Summarize the current progress and status.
"""
        
        # Create LangChain messages for the summarization
        langchain_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message_content)
        ]
        
        # Make the API call
        self.output.print(f"Calling {self.provider.capitalize()} for summary...")
        try:
            response = self.llm.invoke(langchain_messages)
            
            # Extract the summary text
            summary_text = response.content
            
            self.output.print_section("SUMMARY", summary_text)
            
            # Store in knowledge base
            self.kb.update("player_progress", "last_summary", summary_text)
            
            # Get the current messages
            messages = state.get("messages", [])
            
            # If we have a substantial conversation history, perform cleanup
            if len(messages) > 6:
                self.output.print_section("CONTEXT CLEANUP", "Pruning old messages to prevent context overflow")
                
                # Convert to langchain format to process
                langchain_state_messages = self._convert_messages_to_langchain(messages)
                
                # Strategy: 
                # 1. Keep system messages
                # 2. Remove old message exchanges
                # 3. Add a summary message
                
                # Find messages to keep
                messages_to_keep = []
                
                # Keep system messages at the beginning
                i = 0
                while i < len(langchain_state_messages) and isinstance(langchain_state_messages[i], SystemMessage):
                    messages_to_keep.append(langchain_state_messages[i])
                    i += 1
                
                # Keep the most recent messages (last 2 exchanges = 4 messages)
                if len(langchain_state_messages) > i + 4:
                    messages_to_keep.extend(langchain_state_messages[-4:])
                else:
                    messages_to_keep.extend(langchain_state_messages[i:])
                
                # Add a system message with the summary
                new_langchain_messages = [
                    SystemMessage(content=f"Previous gameplay summary: {summary_text}")
                ]
                new_langchain_messages.extend(messages_to_keep)
                
                # Convert back to state format
                new_messages = self._convert_langchain_to_state_messages(new_langchain_messages)
                
                self.output.print(f"Reduced context: {len(messages)} messages → {len(new_messages)} messages")
                
                # Update the state with the reduced message set
                return {
                    **state,
                    "messages": new_messages,
                    "last_summary": summary_text,
                    "summary_due": False,
                    "context_management_needed": False
                }
            
            # If not much history, just update the summary
            return {
                **state,
                "last_summary": summary_text,
                "summary_due": False,
                "context_management_needed": False
            }
            
        except Exception as e:
            self.output.print_section("ERROR", f"Error generating summary: {str(e)}")
            # Return original state with error
            return {
                **state,
                "last_error": f"Summary generation error: {str(e)}",
                "summary_due": False,
                "context_management_needed": False
            }
    
    def execute_action_node(self, state: GameState) -> GameState:
        """Execute the action decided by the agent."""
        self.output.print_section("EXECUTING ACTION")
        
        # Get action info from the parsed structured output if available
        action_info = state.get("action_info", {})
        
        if action_info:
            # Use the structured parsed action
            action = action_info.get("action", "wait").lower()
            hold_frames = action_info.get("hold_frames", 10)
            self.output.print(f"Using structured action: {action} for {hold_frames} frames")
        else:
            # Fallback to text parsing if no structured output is available
            decision_text = state.get("decision_text", "")
            self.output.print("No structured action found, falling back to text parsing")
            
            # Simple action extraction from text
            action = None
            hold_frames = 10  # Default
            
            # Look for button keywords in the decision
            for btn in ["up", "down", "left", "right", "a", "b", "start", "select", "wait"]:
                if btn in decision_text.lower():
                    action = btn
                    
                    # Try to extract hold_frames if specified
                    import re
                    hold_patterns = [
                        r'hold.*?(\d+).*?frames',
                        r'press.*?(\d+).*?frames',
                        r'for.*?(\d+).*?frames'
                    ]
                    
                    for pattern in hold_patterns:
                        match = re.search(pattern, decision_text.lower())
                        if match:
                            try:
                                hold_frames = int(match.group(1))
                                break
                            except ValueError:
                                pass
                    
                    break
            
            # Default to wait if no action found
            if not action:
                action = "wait"
                self.output.print("No clear action found in decision, defaulting to wait")
        
        # Validate the action is one of the allowed buttons
        valid_buttons = ["up", "down", "left", "right", "a", "b", "start", "select", "wait"]
        if action not in valid_buttons:
            self.output.print(f"Invalid action '{action}', defaulting to wait")
            action = "wait"
        
        # Validate hold_frames is within reasonable range
        if not isinstance(hold_frames, int) or hold_frames < 1:
            hold_frames = 10
        elif hold_frames > 30:
            hold_frames = 30
        
        # Execute the action
        result = ""
        error = None
        
        try:
            if action == "wait":
                result = self.emulator.wait_frames(hold_frames)
                self.kb.add_action("wait", f"Waited for {hold_frames} frames")
            else:
                # Use press_button from GameEmulatorAdapter
                result = self.emulator.press_button(action, hold_frames)
                self.kb.add_action(action, f"Pressed {action} for {hold_frames} frames")
        except Exception as e:
            error = str(e)
            result = f"Error: {error}"
            self.output.print(f"Error executing action: {error}")
            
            # Default to wait in case of error
            self.emulator.wait_frames(10)
            self.kb.add_action("wait", f"Error fallback: {error}")
        
        self.output.print(f"Executed {action} for {hold_frames} frames")
        self.output.print(f"Result: {result}")
        
        # Update the state with the result
        return {
            **state,
            "last_action": action,
            "last_result": result,
            "last_error": error
        }
    
    def update_state_node(self, state: GameState) -> GameState:
        """Update the state after executing an action."""
        action_count = state.get("action_count", 0)
        
        # Save state periodically
        if action_count % 50 == 0 and action_count > 0:
            save_path = self.emulator.save_state(f"step_{action_count}.state")
            self.output.print(f"Saved game state to {save_path}")
        
        # Show stats periodically
        if action_count % 10 == 0 and action_count > 0:
            stats_text = "Action distribution:\n"
            stats = self.tools_manager.get_stats()
            total = sum(stats.values()) or 1
            for act, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                stats_text += f"  {act}: {count} ({count/total*100:.1f}%)\n"
            self.output.print_section(f"STATISTICS AFTER {action_count} STEPS", stats_text)
        
        # Return updated state
        return state
    
    async def run_agent_loop(self):
        """
        Run the agent decision loop asynchronously.
        Modified to ensure we don't trigger recursion limits by
        having the agent execute only one full decision cycle per loop iteration.
        """
        # Initialize state
        initial_state = {
            "screen_base64": "",
            "game_state_text": "",
            "recent_actions": "",
            "last_error": None,
            "messages": [],
            "action_count": 0,
            "summary_interval": self.summary_interval,
            "summary_due": False,
            "last_summary": ""
        }
        
        # Track the current state
        current_state = initial_state
        
        # Main agent loop
        while self.running and not self.paused:
            try:
                # Run one complete graph execution (non-streaming)
                # This ensures we don't trigger recursion by running just one agent cycle per loop iteration
                config = {"configurable": {"thread_id": self.thread_id}}
                
                self.output.print("Starting new agent decision cycle")
                
                # Use invoke instead of astream to run one complete cycle
                try:
                    # Execute the graph once with the current state
                    final_state = await self.graph.ainvoke(current_state, config)
                    
                    # Update the current state for the next cycle
                    current_state = final_state
                    
                    # Log completion of the cycle
                    self.output.print(f"Completed agent cycle {current_state.get('action_count', 0)}")
                    
                except Exception as e:
                    self.output.print_section("GRAPH ERROR", f"Error executing graph: {str(e)}")
                    # If there's an error, wait longer before retrying
                    await asyncio.sleep(2)
                
                # Wait between agent cycles to allow the game to process and render
                # This is important to prevent the agent from making decisions too quickly
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.output.print_section("ERROR", f"Error in agent loop: {e}")
                await asyncio.sleep(1)  # Avoid rapid error loops
    
    async def start(self):
        """Start the agent asynchronously."""
        self.output.print_section(
            "STARTING POKÉMON AGENT", 
            f"Model: {self.model_name}, Temperature: {self.temperature}"
        )
        
        # Set flags
        self.running = True
        self.paused = False
        
        # Run only the agent loop (game loop is managed by main.py)
        agent_loop_task = asyncio.create_task(self.run_agent_loop())
        
        # Wait for the agent loop to complete
        await agent_loop_task
    
    def stop(self):
        """Stop the agent."""
        self.running = False
        self.emulator.save_state("final.state")
        
        # Final stats
        self.output.print_section("FINAL STATISTICS")
        stats = self.tools_manager.get_stats()
        total = sum(stats.values()) or 1
        for act, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            self.output.print(f"  {act}: {count} ({count/total*100:.1f}%)")
        
        self.output.print_section("AGENT STOPPED", "Goodbye!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a Pokémon AI agent using LangGraph')
    parser.add_argument('rom_path', help='Path to the Pokémon ROM file')
    parser.add_argument('--api-key', help='API key (will use ANTHROPIC_API_KEY env var if not provided)')
    parser.add_argument('--model', default='claude-3-7-sonnet-20250219', help='Model to use')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no window)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for the LLM')
    parser.add_argument('--speed', type=int, default=1, help='Game speed multiplier')
    parser.add_argument('--no-sound', action='store_true', help='Disable game sound')
    parser.add_argument('--log-to-file', action='store_true', help='Log output to file')
    parser.add_argument('--log-file', help='Path to log file (default: pokemon_agent.log)')
    parser.add_argument('--summary-interval', type=int, default=10, help='Number of turns between summaries')
    
    args = parser.parse_args()
    
    print("Please use langgraph_main.py instead of running this file directly.")
    print("Direct execution is no longer supported due to component dependencies.")