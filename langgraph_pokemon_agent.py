"""
LangGraph Pokémon Agent - A fully asynchronous agent for playing Pokémon
using LangGraph for orchestration and PyBoy for emulation.
With enhanced memory management to prevent context duplication.
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

import anthropic
from PIL import Image
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph import MessagesState
from langchain_core.messages import RemoveMessage, SystemMessage

# Import our modules
from pokemon_emulator import GameEmulator
from pokemon_knowledge import KnowledgeBase
from pokemon_tools import PokemonTools


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
        model_name: str = "claude-3-7-sonnet-20250219", 
        api_key: Optional[str] = None, 
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
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set ANTHROPIC_API_KEY env var or pass directly.")
        
        # Setup output manager
        self.output = OutputManager(output_to_file=output_to_file, log_file=log_file)
        
        # Setup components
        self.rom_path = rom_path
        self.emulator = GameEmulator(rom_path, headless=headless, speed=speed, sound=sound)
        self.kb = KnowledgeBase()
        self.tools_manager = PokemonTools(self.emulator, self.kb)
        
        # Agent settings
        self.model = model_name
        self.temperature = temperature
        self.summary_interval = summary_interval
        self.token_limit = token_limit
        
        # Initialize tokenizer for Claude (approximation using tiktoken)
        # For Claude, we use 'cl100k_base' as a reasonable approximation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Create Claude client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
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
        
        # System prompt for the agent
        system_prompt = """
You are an expert Pokémon player. Analyze the game state and decide on the best action to take.

First, think step-by-step about:
1. Where the player is currently and what's visible on screen
2. What progress has been made in the game so far
3. What the immediate goal should be
4. Available actions and their potential outcomes

Then, decide on the best action to take. Be specific about which button to press and for how long.
Use one of: up, down, left, right, a, b, start, select, or wait if you need to wait for something to happen.
"""

        # Get the last summary if available
        last_summary = state.get("last_summary", "")
        if last_summary:
            system_prompt += f"\n\nRecent Game Progress Summary:\n{last_summary}"

        # Prepare the message content
        message_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": state["screen_base64"]
                }
            },
            {
                "type": "text",
                "text": f"""
Current game state:
{state['game_state_text']}

Recent actions:
{state['recent_actions']}

{f"Note about previous action: {state['last_error']}" if state.get('last_error') else ""}

First, analyze what you see on screen and the current situation.
Then, decide on the best action to take.
"""
            }
        ]
        
        # Create or extend the messages list in the state
        messages = state.get("messages", [])
        
        # Add the current message to the list
        current_message = {"role": "user", "content": message_content}
        messages.append(current_message)
        
        # Filter messages to prevent context overflow
        filtered_messages = self.filter_messages(messages)
        
        # Log the filtering
        if len(filtered_messages) < len(messages):
            self.output.print(f"Filtered messages: {len(messages)} → {len(filtered_messages)}")
        
        # Make the API call
        self.output.print("Calling Claude for action decision...")
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=filtered_messages,  # Use filtered messages
            temperature=self.temperature,
            max_tokens=1000
        )
        
        # Add Claude's response to the messages
        assistant_message = {"role": "assistant", "content": response.content}
        messages.append(assistant_message)
        
        # Extract the decision text
        decision_text = ""
        for content in response.content:
            if content.type == "text":
                decision_text = content.text
                break
        
        self.output.print_section("CLAUDE'S DECISION", decision_text)
        
        # Update the state with the original full messages list
        # (maintaining history but using filtered messages for API calls)
        return {
            **state,
            "messages": messages,
            "decision_text": decision_text,
            "action_count": state.get("action_count", 0) + 1
        }
    
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
        
        # Make the API call
        self.output.print("Calling Claude for summary...")
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": message_content}],
            temperature=self.temperature,
            max_tokens=500
        )
        
        # Extract the summary text
        summary_text = ""
        for content in response.content:
            if content.type == "text":
                summary_text = content.text
                break
        
        self.output.print_section("SUMMARY", summary_text)
        
        # Store in knowledge base
        self.kb.update("player_progress", "last_summary", summary_text)
        
        # Get the current messages
        messages = state.get("messages", [])
        
        # If we have a substantial conversation history, perform cleanup
        if len(messages) > 6:
            self.output.print_section("CONTEXT CLEANUP", "Pruning old messages to prevent context overflow")
            
            # Strategy: 
            # 1. Keep system messages
            # 2. Remove old message exchanges
            # 3. Add a summary message
            
            # Find messages to remove (keeping the most recent 2 exchanges)
            messages_to_keep = []
            
            # Keep any system messages at the beginning
            i = 0
            while i < len(messages) and messages[i].get("role") == "system":
                messages_to_keep.append(messages[i])
                i += 1
            
            # Keep the most recent messages (last 2 exchanges = 4 messages)
            if len(messages) > i + 4:
                messages_to_keep.extend(messages[-4:])
            else:
                messages_to_keep.extend(messages[i:])
            
            # Create a new message array with a system message containing the summary
            new_messages = [
                {"role": "system", "content": f"Previous gameplay summary: {summary_text}"}
            ]
            new_messages.extend(messages_to_keep)
            
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
    
    def execute_action_node(self, state: GameState) -> GameState:
        """Execute the action decided by the agent."""
        self.output.print_section("EXECUTING ACTION")
        
        decision_text = state.get("decision_text", "")
        
        # Simple action extraction - in production, this should be more robust with more NLP
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
        
        # Execute the action
        result = ""
        error = None
        
        try:
            if action == "wait":
                result = self.emulator.wait_frames(hold_frames)
                self.kb.add_action("wait", f"Waited for {hold_frames} frames")
            else:
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
    
    async def run_game_loop(self):
        """Run the main game loop asynchronously."""
        last_frame_time = time.time()
        
        while self.running and not self.paused:
            # Calculate time since last frame
            now = time.time()
            elapsed = now - last_frame_time
            
            # If it's time for a new frame
            if elapsed >= self.frame_time:
                # Run the frame
                self.emulator.pyboy.tick()
                last_frame_time = now
            
            # Small sleep to avoid CPU hogging
            await asyncio.sleep(0.001)
    
    async def run_agent_loop(self):
        """Run the agent decision loop asynchronously."""
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
        
        while self.running and not self.paused:
            try:
                # Stream the graph execution
                config = {"configurable": {"thread_id": self.thread_id}}
                
                async for chunk in self.graph.astream(
                    initial_state, 
                    config, 
                    stream_mode="updates"
                ):
                    # Process updates if needed (for debugging/logging)
                    for node, update in chunk.items():
                        self.output.print(f"Node {node} completed")
                
                # Wait a bit after each agent cycle to allow emulator to run
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.output.print_section("ERROR", f"Error in agent loop: {e}")
                await asyncio.sleep(1)  # Avoid rapid error loops
    
    async def start(self):
        """Start the game and agent asynchronously."""
        self.output.print_section(
            "STARTING POKÉMON AGENT", 
            f"Model: {self.model}, Temperature: {self.temperature}"
        )
        
        # Start the emulator
        self.emulator.start_game(skip_intro=True)
        
        # Set flags
        self.running = True
        self.paused = False
        
        # Start game and agent loops
        game_loop_task = asyncio.create_task(self.run_game_loop())
        agent_loop_task = asyncio.create_task(self.run_agent_loop())
        
        # Wait for both loops
        await asyncio.gather(game_loop_task, agent_loop_task)
    
    def stop(self):
        """Stop the agent."""
        self.running = False
        self.emulator.save_state("final.state")
        self.emulator.close()
        
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
    
    # Create agent
    agent = PokemonAgent(
        rom_path=args.rom_path,
        model_name=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        headless=args.headless,
        speed=args.speed,
        sound=not args.no_sound,
        output_to_file=args.log_to_file,
        log_file=args.log_file,
        summary_interval=args.summary_interval
    )
    
    try:
        # Run event loop
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping...")
    finally:
        agent.stop()