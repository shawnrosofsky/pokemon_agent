import os
import time
import json
import base64
import traceback
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Annotated
import uuid
from io import BytesIO
from PIL import Image
from IPython.display import display, Markdown, HTML

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool, StructuredTool, Tool, BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_anthropic import ChatAnthropic

# Pydantic imports
from pydantic import BaseModel, Field

# Import PyBoy for game interaction
from pyboy import PyBoy
from pyboy.utils import WindowEvent

def print_dict(dictionary, indent=0):
    """
    Prints a dictionary in a readable format with proper indentation.
    
    Args:
        dictionary (dict): The dictionary to print
        indent (int, optional): Initial indentation level. Defaults to 0.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_dict(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")
            
class KnowledgeBase:
    """
    Knowledge base for storing and retrieving game-related information.
    Maintains game state, player progress, and strategic information.
    """
    
    def __init__(self, save_path="knowledge_base.json"):
        self.save_path = save_path
        self.data = {
            "game_state": {},
            "player_progress": {},
            "strategies": {},
            "map_data": {},
            "previous_states": [],
            "action_history": []
        }
        self.load()
    
    def save(self):
        """Save the knowledge base to disk."""
        with open(self.save_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def load(self):
        """Load the knowledge base from disk if it exists."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"Error loading knowledge base: {e}")
    
    def update(self, section, key, value):
        """Update a specific section and key in the knowledge base."""
        if section not in self.data:
            self.data[section] = {}
        self.data[section][key] = value
        self.save()
    
    def get(self, section, key=None):
        """Get data from the knowledge base."""
        if section not in self.data:
            return None
        if key is None:
            return self.data[section]
        return self.data[section].get(key)
    
    def add_action(self, action, result):
        """Add an action and its result to the history."""
        self.data["action_history"].append({
            "action": action,
            "result": result,
            "timestamp": time.time()
        })
        # Limit history size
        if len(self.data["action_history"]) > 100:
            self.data["action_history"] = self.data["action_history"][-100:]
        self.save()
    
    def add_game_state(self, state):
        """Add a game state snapshot to history."""
        self.data["previous_states"].append({
            "state": state,
            "timestamp": time.time()
        })
        # Limit history size
        if len(self.data["previous_states"]) > 10:
            self.data["previous_states"] = self.data["previous_states"][-10:]
        self.save()
    
    def summarize_recent_history(self, max_items=5):
        """Summarize recent action history for context management."""
        recent_actions = self.data["action_history"][-max_items:] if self.data["action_history"] else []
        summary = []
        
        for item in recent_actions:
            summary.append(f"Action: {item['action']}, Result: {item['result']}")
        
        return "\n".join(summary)


class GameEmulator:
    """
    Interface for the Pokémon game emulator (PyBoy).
    Handles game controls and state observation.
    """
    
    # Game Boy button mappings
    BUTTONS = {
        'up': WindowEvent.PRESS_ARROW_UP,
        'down': WindowEvent.PRESS_ARROW_DOWN,
        'left': WindowEvent.PRESS_ARROW_LEFT,
        'right': WindowEvent.PRESS_ARROW_RIGHT,
        'a': WindowEvent.PRESS_BUTTON_A,
        'b': WindowEvent.PRESS_BUTTON_B,
        'start': WindowEvent.PRESS_BUTTON_START,
        'select': WindowEvent.PRESS_BUTTON_SELECT
    }
    
    RELEASE_BUTTONS = {
        'up': WindowEvent.RELEASE_ARROW_UP,
        'down': WindowEvent.RELEASE_ARROW_DOWN,
        'left': WindowEvent.RELEASE_ARROW_LEFT,
        'right': WindowEvent.RELEASE_ARROW_RIGHT,
        'a': WindowEvent.RELEASE_BUTTON_A,
        'b': WindowEvent.RELEASE_BUTTON_B,
        'start': WindowEvent.RELEASE_BUTTON_START,
        'select': WindowEvent.RELEASE_BUTTON_SELECT
    }
    
    # Memory addresses for Pokémon Red/Blue
    MEMORY_MAP = {
        # Player and Map
        'player_x': 0xD362,            # X position on map
        'player_y': 0xD361,            # Y position on map
        'map_id': 0xD35E,              # Current map ID
        'player_direction': 0xD367,    # Direction (0: down, 4: up, 8: left, 0xC: right)
        
        # Battle system
        'battle_type': 0xD057,         # Non-zero when in battle
        'enemy_pokemon_hp': 0xCFE7,    # Enemy Pokémon HP (2 bytes, little endian)
        'enemy_pokemon_level': 0xCFE8, # Enemy Pokémon level
        'enemy_pokemon_species': 0xCFDE, # Enemy Pokémon species ID
        
        # Player Pokémon
        'pokemon_count': 0xD163,       # Number of Pokémon in party
        'first_pokemon_hp': 0xD16C,    # First Pokémon's current HP (2 bytes)
        'first_pokemon_max_hp': 0xD18D, # First Pokémon's max HP (2 bytes)
        'first_pokemon_level': 0xD18C, # First Pokémon's level
        
        # Menu and text
        'text_progress': 0xC6AC,       # Text box progress (non-zero when text is displaying)
        
        # Badges and progress
        'badges': 0xD356,              # Badges obtained (each bit = one badge)
        'money': 0xD347,               # Money (3 bytes, BCD format)
    }
    
    def __init__(self, rom_path, headless=False, speed=1, sound=False):
        """Initialize the game emulator with the ROM."""
        self.rom_path = rom_path
        window = "null" if headless else "SDL2"
        self.pyboy = PyBoy(rom_path, window=window, sound=sound, sound_emulated=sound)
        self.pyboy.set_emulation_speed(speed)
        
        # Setup screen manager
        self.screen_manager = self.pyboy.screen
        
        # Game state tracking
        self.frame_count = 0
        self.save_dir = "saves"
        os.makedirs(self.save_dir, exist_ok=True)
    
    def start_game(self, skip_intro=True):
        """Start the game and optionally skip intro sequence."""
        # Let the game boot
        for _ in range(100):
            self.pyboy.tick()
            self.frame_count += 1
        
        if skip_intro:
            # Skip intro sequence
            for _ in range(3):
                print(self.press_button('start', hold_frames=5))
                time.sleep(0.1)
        
        # Wait for game to settle
        for _ in range(60):
            self.pyboy.tick()
            self.frame_count += 1
    
    def press_button(self, button, hold_frames=10):
        """Press a button for the specified number of frames."""
        if button not in self.BUTTONS:
            print(f"Warning: Unknown button '{button}'")
            return f"Error: Unknown button '{button}'"
        
        # Press button
        self.pyboy.send_input(self.BUTTONS[button])
        
        # Hold for frames
        for _ in range(hold_frames):
            self.pyboy.tick()
            self.frame_count += 1
        
        # Release button
        self.pyboy.send_input(self.RELEASE_BUTTONS[button])
        self.pyboy.tick()
        self.frame_count += 1
        
        return f"Pressed {button} for {hold_frames} frames"
    
    def get_memory_value(self, address):
        """Get value from a memory address."""
        if isinstance(address, str):
            if address in self.MEMORY_MAP:
                address = self.MEMORY_MAP[address]
            else:
                print(f"Warning: Unknown memory address alias '{address}'")
                return None
        
        return self.pyboy.memory[address]
    
    def get_2byte_value(self, address):
        """Get a 2-byte value (little endian)."""
        if isinstance(address, str):
            if address in self.MEMORY_MAP:
                address = self.MEMORY_MAP[address]
            else:
                return None
        
        low_byte = self.get_memory_value(address)
        high_byte = self.get_memory_value(address + 1)
        return (high_byte << 8) | low_byte
    
    def get_screen(self):
        """Get current screen as numpy array."""
        return self.screen_manager.ndarray
    
    def get_screen_as_pil(self):
        """Get current screen as PIL image."""
        screen_array = self.get_screen()
        return Image.fromarray(screen_array)
    
    def get_screen_base64(self):
        """Get current screen as base64 string."""
        pil_image = self.get_screen_as_pil()
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def get_game_state(self):
        """Get comprehensive game state."""
        # Player position and map
        player_x = self.get_memory_value('player_x')
        player_y = self.get_memory_value('player_y')
        current_map = self.get_memory_value('map_id')
        player_direction = self.get_memory_value('player_direction')
        
        # Battle information
        in_battle = self.get_memory_value('battle_type') != 0
        
        # Pokémon party
        pokemon_count = self.get_memory_value('pokemon_count')
        
        # First Pokémon info
        first_pokemon = {}
        if pokemon_count > 0:
            first_pokemon = {
                'hp': self.get_2byte_value('first_pokemon_hp'),
                'max_hp': self.get_2byte_value('first_pokemon_max_hp'),
                'level': self.get_memory_value('first_pokemon_level')
            }
        
        # Enemy Pokémon if in battle
        enemy_pokemon = {}
        if in_battle:
            enemy_pokemon = {
                'hp': self.get_2byte_value('enemy_pokemon_hp'),
                'level': self.get_memory_value('enemy_pokemon_level'),
                'species': self.get_memory_value('enemy_pokemon_species')
            }
        
        # Text status
        text_active = self.get_memory_value('text_progress') != 0
        
        # Progress indicators
        badges = self.get_memory_value('badges')
        badge_count = bin(badges).count('1') if badges is not None else 0
        
        # Format money from BCD
        money_bytes = [
            self.get_memory_value(self.MEMORY_MAP['money']),
            self.get_memory_value(self.MEMORY_MAP['money'] + 1),
            self.get_memory_value(self.MEMORY_MAP['money'] + 2)
        ]
        
        money = 0
        for byte in money_bytes:
            if byte is not None:
                money = money * 100 + ((byte >> 4) * 10 + (byte & 0x0F))
        
        return {
            'frame': self.frame_count,
            'player': {
                'position': (player_x, player_y),
                'map': current_map,
                'direction': player_direction,
                'badges': badge_count,
                'money': money
            },
            'battle': {
                'active': in_battle,
                'enemy_pokemon': enemy_pokemon if in_battle else None
            },
            'party': {
                'count': pokemon_count,
                'first_pokemon': first_pokemon if pokemon_count > 0 else None
            },
            'ui': {
                'text_active': text_active
            }
        }
    
    def format_game_state(self, state):
        """Format game state as readable text."""
        text = "CURRENT GAME STATE:\n"
        
        # Player info
        player = state['player']
        text += f"Player Position: ({player['position'][0]}, {player['position'][1]}) on Map {player['map']}\n"
        text += f"Direction: {player['direction']}, Money: ${player['money']}, Badges: {player['badges']}\n"
        
        # Battle info
        if state['battle']['active']:
            enemy = state['battle']['enemy_pokemon']
            text += "\nIN BATTLE:\n"
            text += f"Enemy Pokémon: Species #{enemy['species']}, Level {enemy['level']}, HP: {enemy['hp']}\n"
        
        # Party info
        party = state['party']
        text += f"\nPokémon Party: {party['count']} Pokémon\n"
        if party['first_pokemon']:
            first = party['first_pokemon']
            text += f"First Pokémon: Level {first['level']}, HP: {first['hp']}/{first['max_hp']}\n"
        
        # UI state
        ui = state['ui']
        if ui['text_active']:
            text += "\nText is currently being displayed.\n"
        
        return text
    
    def save_state(self, filename=None):
        """Save game state to file."""
        if filename is None:
            filename = f"{self.save_dir}/state_frame_{self.frame_count}.state"
        else:
            filename = f"{self.save_dir}/{filename}"
        
        with open(filename, "wb") as f:
            self.pyboy.save_state(f)

        return filename
    
    def load_state(self, filename):
        """Load game state from file."""
        filepath = f"{self.save_dir}/{filename}" if not filename.startswith(self.save_dir) else filename
        
        try:
            self.pyboy.load_state(filepath)
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False
    
    def wait_frames(self, num_frames):
        """Wait for specified number of frames."""
        for _ in range(num_frames):
            self.pyboy.tick()
            self.frame_count += 1
        
        return f"Waited for {num_frames} frames"
    
    def close(self):
        """Clean up resources."""
        self.pyboy.stop()


# Define state type for the LangGraph
class PokemonAgentState(TypedDict):
    """State for the Pokémon LangGraph agent."""
    # Current observation
    observation: str
    # Current game state as text
    game_state: Dict[str, Any]
    # Current screen as base64
    screen: str
    # The conversation history
    messages: List[Any]
    # The next action to take
    next_action: Optional[str]
    # Knowledge base summary
    knowledge_summary: str
    # Additional context
    context: Dict[str, Any]
    # Tool error tracking
    tool_error: Optional[str]


# Define tools for the LangGraph
@tool
def press_button(button: str, hold_frames: int = 10, emulator=None) -> str:
    """
    Press a button on the Game Boy for a specified number of frames.
    
    Args:
        button: The button to press ('up', 'down', 'left', 'right', 'a', 'b', 'start', 'select')
        hold_frames: Number of frames to hold the button (default: 10)
        
    Returns:
        str: Result of pressing the button
    """
    if emulator is None:
        return "Error: Emulator not available"
    
    return emulator.press_button(button, hold_frames)

@tool
def wait_frames(num_frames: int = 30, emulator=None) -> str:
    """
    Wait for a specified number of frames without taking any action.
    This is a safe action when unsure what to do or when recovering from an error.
    
    Args:
        num_frames: Number of frames to wait (default: 30)
        
    Returns:
        str: Result of waiting
    """
    if emulator is None:
        return "Error: Emulator not available"
    
    return emulator.wait_frames(num_frames)

@tool
def navigate_to(destination: str, emulator=None, knowledge_base=None) -> str:
    """
    Navigate to a specific location in the game.
    
    Args:
        destination: The destination location to navigate to
        
    Returns:
        str: Result of the navigation attempt
    """
    if emulator is None or knowledge_base is None:
        return "Error: Emulator or knowledge base not available"
    
    # Get current position
    state = emulator.get_game_state()
    current_pos = state['player']['position']
    current_map = state['player']['map']
    
    # Check if we have path information in knowledge base
    paths = knowledge_base.get("map_data", "paths") or {}
    key = f"{current_map}_{destination}"
    
    if key in paths:
        # Execute saved path
        path = paths[key]
        result = f"Navigating from map {current_map} to {destination} using stored path.\n"
        
        for step in path[:5]:  # Execute first few steps
            result += emulator.press_button(step) + "\n"
        
        return result
    else:
        # No path found, use simple navigation
        return f"No stored path found for {destination}. Please use press_button for manual navigation."

@tool
def update_knowledge(section: str, key: str, value: str, knowledge_base=None) -> str:
    """
    Update the knowledge base with new information.
    
    Args:
        section: Section to update (game_state, player_progress, strategies, map_data)
        key: Key within the section
        value: Value to store
        
    Returns:
        str: Confirmation of the update
    """
    if knowledge_base is None:
        return "Error: Knowledge base not available"
    
    knowledge_base.update(section, key, value)
    return f"Updated knowledge base: {section}.{key} = {value}"


class PokemonLangGraphAgent:
    """
    Pokémon agent implementation using LangGraph with improved error handling.
    """
    
    def __init__(self, rom_path, model_name="claude-3-7-sonnet-20250219", 
                 api_key=None, temperature=1.0,
                 max_tokens=10000,
                 thinking=False, thinking_budget=4000,
                 headless=False, speed=1, sound=False,
                 checkpoint_dir="checkpoints"):
        """Initialize the LangGraph agent."""
        # Set up API key
        self.api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set ANTHROPIC_API_KEY env var or pass directly.")
        
        # Set up model
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.thinking_budget = thinking_budget
        if thinking:
            self.thinking_params = {"type": "enabled", "budget_tokens": thinking_budget}
        else:
            self.thinking_params = {"type": "disabled"} 
        
        # Initialize game emulator
        self.emulator = GameEmulator(rom_path, headless=headless, speed=speed, sound=sound)
        
        # Initialize knowledge base
        self.knowledge_base = KnowledgeBase()
        
        # Set up and configure the tools with access to resources
        self.tools = self._configure_tools()
        
        # Properly create the LLM
        self.llm = ChatAnthropic(
            model_name=self.model_name,
            anthropic_api_key=self.api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            thinking=self.thinking_params,
        )
        
        # Now bind the tools using the correct method
        self.llm = self.llm.bind_tools(self.tools)
        
        # Set up checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Configure LangGraph checkpointing
        self.checkpointer = MemorySaver()
        
        # Build the graph
        self.graph = self._build_graph()
        
        # For summarization
        self.conversation_turn_count = 0
        self.last_summary = ""
        self.max_turns_before_summary = 10
        
        # Statistics
        self.action_stats = {}
        self.start_time = None
    
    def _configure_tools(self):
        """Configure the tools with access to the emulator and knowledge base."""
        # Create copies of the tool functions with bound resources
        def press_button_bound(button: str, hold_frames: int = 10) -> str:
            return press_button(button, hold_frames, emulator=self.emulator)
        
        def navigate_to_bound(destination: str) -> str:
            return navigate_to(destination, emulator=self.emulator, knowledge_base=self.knowledge_base)
        
        def update_knowledge_bound(section: str, key: str, value: str) -> str:
            return update_knowledge(section, key, value, knowledge_base=self.knowledge_base)
        
        def wait_frames_bound(num_frames: int = 30) -> str:
            return wait_frames(num_frames, emulator=self.emulator)
        
        # Create Tools with the bound functions
        press_button_tool = Tool(
            name="press_button",
            description="Press a button on the Game Boy for a specified number of frames",
            func=press_button_bound
        )
        
        navigate_tool = Tool(
            name="navigate_to",
            description="Navigate to a specific location in the game",
            func=navigate_to_bound
        )
        
        update_kb_tool = Tool(
            name="update_knowledge",
            description="Update the knowledge base with new information",
            func=update_knowledge_bound
        )
        
        wait_tool = Tool(
            name="wait_frames",
            description="Wait for a specified number of frames without taking any action (safe fallback)",
            func=wait_frames_bound
        )
        
        # Convert to OpenAI-compatible tools for Claude
        return [
            convert_to_openai_tool(press_button_tool),
            convert_to_openai_tool(navigate_tool),
            convert_to_openai_tool(update_kb_tool),
            convert_to_openai_tool(wait_tool)
        ]
    
    def _build_graph(self):
        """Build the LangGraph for the agent."""
        # Define the state graph
        graph_builder = StateGraph(PokemonAgentState)
        
        # Define nodes
        graph_builder.add_node("observe_game", self._observe_game_state)
        graph_builder.add_node("analyze_situation", self._analyze_situation)
        graph_builder.add_node("decide_action", self._decide_action)
        graph_builder.add_node("handle_tool_calls", self._handle_tool_calls)
        graph_builder.add_node("update_knowledge", self._update_knowledge)
        graph_builder.add_node("check_summarization", self._check_summarization)
        
        # Define edges
        graph_builder.add_edge("observe_game", "analyze_situation")
        graph_builder.add_edge("analyze_situation", "decide_action")
        graph_builder.add_edge("decide_action", "handle_tool_calls")
        graph_builder.add_edge("handle_tool_calls", "update_knowledge")
        graph_builder.add_edge("update_knowledge", "check_summarization")
        
        # Conditional edge: either END or loop back to observe
        graph_builder.add_conditional_edges(
            "check_summarization",
            self._should_continue,
            {
                "continue": "observe_game",
                "stop": END
            }
        )
        
        # Set entry point
        graph_builder.set_entry_point("observe_game")
        
        # Compile the graph
        graph = graph_builder.compile(checkpointer=self.checkpointer)
        
        return graph
    
    def _observe_game_state(self, state: PokemonAgentState) -> PokemonAgentState:
        """Node: Observe the current game state."""
        # Get the current game state
        game_state = self.emulator.get_game_state()
        
        # Get the screen as base64
        screen_base64 = self.emulator.get_screen_base64()
        
        # Format game state as text
        state_text = self.emulator.format_game_state(game_state)
        
        # Get knowledge base summary
        kb_summary = self.knowledge_base.summarize_recent_history()
        
        # Update the state
        state["game_state"] = game_state
        state["screen"] = screen_base64
        state["observation"] = state_text
        state["knowledge_summary"] = kb_summary
        
        # Initialize other fields if needed
        if "messages" not in state:
            state["messages"] = []
        if "context" not in state:
            state["context"] = {}
        if "tool_error" not in state:
            state["tool_error"] = None
        
        return state
    
    def _analyze_situation(self, state: PokemonAgentState) -> PokemonAgentState:
        """Node: Analyze the current game situation."""
        # Create system message
        system_message = SystemMessage(content="""
You are an expert Pokémon game analyst. Your job is to analyze the current game state and provide insights.
Focus on:
1. The player's current situation, location, and surroundings
2. Battle status if in a battle
3. Player's Pokémon team status
4. Visible UI elements and text
5. Immediate obstacles or objectives
6. Progress towards game goals

Provide a clear, concise analysis to help decide the next best action.
""")
        
        # Create human message with observation and screen
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

Recent actions:
{state['knowledge_summary']}

{f"Note about previous action: {state['tool_error']}" if state["tool_error"] else ""}

Based on the screen and game state information, what's happening right now?
"""
            }
        ])
        
        # Get response from LLM
        try:
            messages = [system_message, human_message]
            response = self.llm.invoke(messages)
            
            # Update state with analysis
            state["messages"] = messages + [response]
            state["context"]["analysis"] = response.content
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            # If there's an error, provide a simple fallback analysis
            state["context"]["analysis"] = "Unable to analyze the current situation due to an error. Please take a basic action to continue."
        
        return state
    
    def _decide_action(self, state: PokemonAgentState) -> PokemonAgentState:
        """Node: Decide on the next action to take using tool calling."""
        # Create system message for decision making with tool usage
        system_message = SystemMessage(content="""
You are an expert Pokémon player. Your job is to decide the best next action based on the current game state.

Use the available tools to control the game:
1. press_button: Press a Game Boy button (up, down, left, right, a, b, start, select)
2. navigate_to: Navigate to a specific game location (if path is known)
3. update_knowledge: Update the knowledge base with new information
4. wait_frames: Wait for a specified number of frames without taking any action (safe fallback)

For simple actions, use press_button with the appropriate button. You can specify how long to hold 
by setting the hold_frames parameter.

If you're unsure what to do, or if there was an error with the previous action, use wait_frames
instead of pressing buttons randomly. Waiting is safer than pressing buttons when uncertain.

Always choose the action that will make the most progress in the game given the current situation.
""")
        
        # Create human message with analysis and any previous error
        human_message = HumanMessage(content=f"""
Based on the analysis, please take the most appropriate action.

Analysis:
{state["context"]["analysis"]}

{f"IMPORTANT - Error with previous action: {state['tool_error']}" if state["tool_error"] else ""}

Choose ONE tool to use for the next action. If there was an error or you're uncertain, consider using wait_frames instead of making potentially harmful button presses.
""")
        
        # Get response from LLM with tool calls
        try:
            messages = [system_message, human_message]
            response = self.llm.invoke(messages)
            
            # Update state with action decision
            state["messages"].extend([system_message, human_message, response])
            
            # Store the tool call info (if any)
            if hasattr(response, "tool_calls") and response.tool_calls:
                state["context"]["tool_calls"] = response.tool_calls
            else:
                # If no tool calls, default to waiting as a safe fallback
                state["context"]["fallback_action"] = "wait"
                state["tool_error"] = "No tool was called, using safe wait fallback."
        
        except Exception as e:
            print(f"Error in decision making: {e}")
            # If there's an error, default to waiting
            state["context"]["fallback_action"] = "wait"
            state["tool_error"] = f"Error occurred during decision making: {str(e)}. Using wait fallback."
        
        return state
    
    # Fixed _handle_tool_calls method to handle positional arguments correctly
    def _handle_tool_calls(self, state: PokemonAgentState) -> PokemonAgentState:
        """Node: Handle tool calls from the LLM with safe fallbacks."""
        # Clear any previous tool error
        state["tool_error"] = None
        
        # Check if we have tool calls to execute
        if "tool_calls" in state["context"] and state["context"]["tool_calls"]:
            tool_calls = state["context"]["tool_calls"]
            
            try:
                # Process the first tool call
                tool_call = tool_calls[0]
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                # Log the tool call
                print(f"Tool call: {tool_name} with args: {tool_args}")
                print(tool_calls)
                
                # Execute the appropriate tool
                if tool_name == "press_button":
                    # Handle different argument formats
                    if "button" in tool_args:
                        # Named argument format
                        button = tool_args.get("button", "")
                        hold_frames = tool_args.get("hold_frames", 10)
                    elif "__arg1" in tool_args:
                        # Positional argument format
                        button = tool_args.get("__arg1", "")
                        hold_frames = tool_args.get("__arg2", 10)
                    else:
                        # Fallback for unknown format
                        button = ""
                        hold_frames = 10
                        state["tool_error"] = f"Unknown argument format in press_button: {tool_args}"
                    
                    # Check if button is valid
                    if button not in self.emulator.BUTTONS:
                        state["tool_error"] = f"Invalid button: '{button}'. Using wait fallback."
                        result = self.emulator.wait_frames(30)
                        state["next_action"] = "wait"
                        state["context"]["last_action"] = "wait_30"
                        state["context"]["last_result"] = result
                    else:
                        # Press the button
                        result = self.emulator.press_button(button, hold_frames)
                        
                        # Update action stats
                        if button in self.action_stats:
                            self.action_stats[button] += 1
                        else:
                            self.action_stats[button] = 1
                        
                        # Store the action and result
                        state["next_action"] = button
                        state["context"]["last_action"] = button
                        state["context"]["last_result"] = result
                    
                elif tool_name == "navigate_to":
                    # Handle different argument formats
                    if "destination" in tool_args:
                        destination = tool_args.get("destination", "")
                    elif "__arg1" in tool_args:
                        destination = tool_args.get("__arg1", "")
                    else:
                        destination = ""
                        state["tool_error"] = f"Unknown argument format in navigate_to: {tool_args}"
                    
                    result = navigate_to(destination, self.emulator, self.knowledge_base)
                    
                    # Update action stats for navigation
                    if "navigate" in self.action_stats:
                        self.action_stats["navigate"] += 1
                    else:
                        self.action_stats["navigate"] = 1
                    
                    # Store the action and result
                    state["next_action"] = f"navigate_to_{destination}"
                    state["context"]["last_action"] = f"navigate_to_{destination}"
                    state["context"]["last_result"] = result
                    
                elif tool_name == "update_knowledge":
                    # Handle different argument formats
                    if "section" in tool_args:
                        section = tool_args.get("section", "")
                        key = tool_args.get("key", "")
                        value = tool_args.get("value", "")
                    elif "__arg1" in tool_args:
                        section = tool_args.get("__arg1", "")
                        key = tool_args.get("__arg2", "")
                        value = tool_args.get("__arg3", "")
                    else:
                        section = ""
                        key = ""
                        value = ""
                        state["tool_error"] = f"Unknown argument format in update_knowledge: {tool_args}"
                    
                    result = update_knowledge(section, key, value, self.knowledge_base)
                    
                    # Update action stats for knowledge updates
                    if "update_kb" in self.action_stats:
                        self.action_stats["update_kb"] += 1
                    else:
                        self.action_stats["update_kb"] = 1
                    
                    # Store the action and result
                    state["next_action"] = f"update_kb_{section}_{key}"
                    state["context"]["last_action"] = f"update_kb_{section}_{key}"
                    state["context"]["last_result"] = result
                
                elif tool_name == "wait_frames":
                    # Handle different argument formats
                    if "num_frames" in tool_args:
                        num_frames = tool_args.get("num_frames", 30)
                    elif "__arg1" in tool_args:
                        num_frames = tool_args.get("__arg1", 30)
                    else:
                        num_frames = 30
                        state["tool_error"] = f"Unknown argument format in wait_frames: {tool_args}"
                    
                    # Execute the wait tool
                    result = self.emulator.wait_frames(num_frames)
                    
                    # Update action stats for waiting
                    if "wait" in self.action_stats:
                        self.action_stats["wait"] += 1
                    else:
                        self.action_stats["wait"] = 1
                    
                    # Store the action and result
                    state["next_action"] = "wait"
                    state["context"]["last_action"] = f"wait_{num_frames}"
                    state["context"]["last_result"] = result
                
                else:
                    # Unknown tool - use wait as a safe fallback
                    state["tool_error"] = f"Unknown tool: {tool_name}. Using wait fallback."
                    
                    # Wait for 30 frames
                    result = self.emulator.wait_frames(30)
                    
                    # Store the action and result
                    state["next_action"] = "wait"
                    state["context"]["last_action"] = "wait_30"
                    state["context"]["last_result"] = result
                
                # Create tool message with result
                tool_message = ToolMessage(
                    content=result,
                    name=tool_name,
                    tool_call_id=tool_call['id']
                )
                
                # Add the tool message to the conversation
                state["messages"].append(tool_message)
                
            except Exception as e:
                # Handle tool execution errors
                error_msg = f"Error executing tool: {str(e)}"
                traceback_str = traceback.format_exc()
                print(f"{error_msg}\n{traceback_str}")
                
                # Store the error
                state["tool_error"] = error_msg
                
                # Use wait as a safe fallback
                try:
                    # Wait for 30 frames
                    result = self.emulator.wait_frames(30)
                    
                    # Store the fallback action and result
                    state["next_action"] = "wait"
                    state["context"]["last_action"] = "wait_30"
                    state["context"]["last_result"] = result
                    
                    # Update action stats for waiting
                    if "wait" in self.action_stats:
                        self.action_stats["wait"] += 1
                    else:
                        self.action_stats["wait"] = 1
                except Exception as e2:
                    print(f"Even wait fallback failed: {e2}")
                    state["context"]["last_result"] = "Both tool call and wait fallback failed."
        
        # Rest of method remains the same for fallback cases
        return state
    
    def _update_knowledge(self, state: PokemonAgentState) -> PokemonAgentState:
        """Node: Update the knowledge base with new information."""
        # Add action to history
        action = state["context"].get("last_action", "unknown")
        result = state["context"].get("last_result", "unknown")
        
        self.knowledge_base.add_action(action, result)
        self.knowledge_base.add_game_state(state["game_state"])
        self.conversation_turn_count += 1
        
        return state
    
    def _check_summarization(self, state: PokemonAgentState) -> PokemonAgentState:
        """Node: Check if we need to summarize conversation history."""
        if self.conversation_turn_count >= self.max_turns_before_summary:
            # Create system message for summarization
            system_message = SystemMessage(content="""
Summarize the recent game progress and key events from the conversation history.
Focus on:
1. Current location and objective
2. Recent battles and encounters
3. Party status
4. Important items obtained
5. Progress towards game goals

Be concise but include all essential information.
""")
            
            # Create human message with conversation history
            history_text = "\n".join([
                f"Action: {item['action']}, Result: {item['result']}" 
                for item in self.knowledge_base.data["action_history"][-20:]
            ])
            
            human_message = HumanMessage(content=f"""
Please summarize the recent game progress:

{history_text}

Previous summary (if any):
{self.last_summary}
""")
            
            try:
                # Get summary from LLM
                summary_response = self.llm.invoke([system_message, human_message])
                
                # Update last summary
                self.last_summary = summary_response.content
                
                # Reset conversation turn counter
                self.conversation_turn_count = 0
                
                # Clear messages but keep the summary
                state["messages"] = [
                    SystemMessage(content="Previous progress summary: " + self.last_summary)
                ]
            except Exception as e:
                print(f"Error generating summary: {e}")
                # Just continue without summarization if it fails
                pass
        
        return state
    
    def _should_continue(self, state: PokemonAgentState) -> str:
        """Conditional edge: Determine if we should continue or stop."""
        # Check if max steps reached
        if state["context"].get("step_count", 0) >= state["context"].get("max_steps", float('inf')):
            return "stop"
        
        # Check if user requested stop
        if state["context"].get("stop_requested", False):
            return "stop"
        
        # Continue by default
        return "continue"
    
    def start(self, skip_intro=True):
        """Start the game."""
        self.emulator.start_game(skip_intro=skip_intro)
        self.start_time = time.time()
    
    def run(self, num_steps=None):
        """Run the agent for a specified number of steps."""
        if num_steps is None:
            num_steps = float('inf')
            
        if self.start_time is None:
            self.start()
        
        try:
            print("Starting Pokémon LangGraph agent...")
            
            # Create thread ID for checkpointing
            thread_id = str(uuid.uuid4())
            
            # Create initial state
            initial_state: PokemonAgentState = {
                "observation": "",
                "game_state": {},
                "screen": "",
                "messages": [],
                "next_action": None,
                "knowledge_summary": "",
                "context": {
                    "step_count": 0,
                    "max_steps": num_steps,
                    "thread_id": thread_id,
                    "stop_requested": False
                },
                "tool_error": None
            }
            
            # Run the graph
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": None},
            for step, state in enumerate(self.graph.stream(initial_state, {"thread_id": thread_id, "recursion_limit": 10000000000000000})):
                try:
                    # For debugging in Jupyter notebooks
                    if 'display' in globals():
                        ...
                        # display(f"Step {step}: \n{self.emulator.format_game_state(state['game_state'])}")
                        # display(f"Step {step}: \n{state['observation']}")
                except:
                    pass
                # print_dict(state)
                print(f'Step {step+1}:')
                if state.get("messages"):
                    print(state["messages"])
                if state.get("next_action"):
                    # Print progress
                    print(f"Step {step+1}: {state['next_action']} -> {state['context'].get('last_result', '')}")
                    
                    # Print action stats every 10 steps
                    if (step + 1) % 10 == 0:
                        print("\nAction distribution:")
                        total = sum(self.action_stats.values())
                        for act, count in sorted(self.action_stats.items(), key=lambda x: x[1], reverse=True):
                            print(f"  {act}: {count} ({count/total*100:.1f}%)")
                    
                    # Save game state periodically
                    if (step + 1) % 50 == 0:
                        save_path = self.emulator.save_state(f"step_{step+1}.state")
                        print(f"Saved game state to {save_path}")
                
                # Break if we're done
                if state.get("_graph_done"):
                    break
                
                # Small delay to prevent maxing CPU
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Error running agent: {e}")
            traceback_str = traceback.format_exc()
            print(f"Traceback:\n{traceback_str}")
        finally:
            # Save final state
            self.emulator.save_state("final.state")
            
            # Save action statistics
            with open("action_stats.json", 'w') as f:
                json.dump(self.action_stats, f, indent=2)
            
            # Print summary
            elapsed_time = time.time() - self.start_time
            print(f"\nRun complete! Elapsed time: {elapsed_time:.2f}s")
            print(f"Frame count: {self.emulator.frame_count}")
            
            # Clean up
            self.emulator.close()
    
    def close(self):
        """Clean up resources."""
        self.emulator.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a Pokémon AI agent using LangGraph')
    parser.add_argument('rom_path', help='Path to the Pokémon ROM file')
    parser.add_argument('--api-key', help='API key (will use ANTHROPIC_API_KEY env var if not provided)')
    parser.add_argument('--model', default='claude-3-7-sonnet-20250219', help='Model to use')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no window)')
    parser.add_argument('--steps', type=int, help='Number of steps to run (infinite if not specified)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for the LLM')
    parser.add_argument('--speed', type=int, default=1, help='Game speed multiplier')
    parser.add_argument('--max-tokens', type=int, default=10000, help='Maximum tokens for LLM')
    parser.add_argument('--no-sound', action='store_true', help='Disable sound')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Directory for checkpoints')
    parser.add_argument('--thinking', action='store_true', help='Enable thinking mode')
    parser.add_argument('--thinking_budget', type=int, default=2000, help='Thinking budget in tokens')
    
    args = parser.parse_args()
    
    # Create and run the agent
    sound = not args.no_sound
    agent = PokemonLangGraphAgent(
        rom_path=args.rom_path,
        model_name=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        thinking_budget=args.thinking_budget,
        headless=args.headless,
        speed=args.speed,
        sound=sound,
        checkpoint_dir=args.checkpoint_dir
    )
    
    try:
        agent.start()
        agent.run(num_steps=args.steps)
    finally:
        agent.close()