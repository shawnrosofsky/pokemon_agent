"""
LangGraph Tools Adapter - Defines tools for the LangGraph Pokemon Agent.
"""

import time
import asyncio
from typing import Dict, Any, List, Optional, Union, Literal

from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph


class ButtonPressInput(BaseModel):
    """Input model for button press tool."""
    button: Literal["up", "down", "left", "right", "a", "b", "start", "select"]
    hold_frames: int = Field(default=10, ge=1, le=30)


class WaitFramesInput(BaseModel):
    """Input model for wait frames tool."""
    num_frames: int = Field(default=30, ge=1, le=60)


class UpdateKnowledgeInput(BaseModel):
    """Input model for knowledge base update tool."""
    section: Literal["game_state", "player_progress", "strategies", "map_data"]
    key: str
    value: str


class GetGameInfoInput(BaseModel):
    """Input model for game info tool."""
    info_type: Literal["player_position", "map_id", "pokemon_status", "battle_status"]


class PokemonToolsAdapter:
    """Adapter for Pokemon tools to be used with LangGraph."""
    
    def __init__(self, game_emulator, knowledge_base):
        """
        Initialize tools with a game emulator and knowledge base.
        
        Args:
            game_emulator: GameEmulatorAdapter instance to execute actions
            knowledge_base: KnowledgeBase instance to store/retrieve information
        """
        self.emulator = game_emulator
        self.kb = knowledge_base
        self.action_stats = {}  # Statistics for tracking tool usage
        self.last_error = None  # Track the most recent error
        
        # Lock for synchronization
        self._lock = asyncio.Lock()
    
    async def press_button(self, input_data: ButtonPressInput) -> str:
        """Execute the press_button tool."""
        button = input_data.button
        hold_frames = input_data.hold_frames
        
        # Track action stats
        async with self._lock:
            if button in self.action_stats:
                self.action_stats[button] += 1
            else:
                self.action_stats[button] = 1
        
        # Execute via emulator
        result = await self.emulator.async_press_button(button, hold_frames)
        return result
    
    async def wait_frames(self, input_data: WaitFramesInput) -> str:
        """Execute the wait_frames tool."""
        num_frames = input_data.num_frames
        
        # Track stats
        async with self._lock:
            if "wait" in self.action_stats:
                self.action_stats["wait"] += 1
            else:
                self.action_stats["wait"] = 1
        
        # Execute via emulator
        result = await self.emulator.async_wait_frames(num_frames)
        return result
    
    async def update_knowledge(self, input_data: UpdateKnowledgeInput) -> str:
        """Execute the update_knowledge tool."""
        section = input_data.section
        key = input_data.key
        value = input_data.value
        
        # Track stats
        async with self._lock:
            if "update_kb" in self.action_stats:
                self.action_stats["update_kb"] += 1
            else:
                self.action_stats["update_kb"] = 1
        
        # Update knowledge base
        self.kb.update(section, key, value)
        return f"Updated knowledge base: {section}.{key} = {value}"
    
    async def get_game_info(self, input_data: GetGameInfoInput) -> str:
        """Execute the get_game_info tool."""
        info_type = input_data.info_type
        
        # Track stats
        async with self._lock:
            if "get_info" in self.action_stats:
                self.action_stats["get_info"] += 1
            else:
                self.action_stats["get_info"] = 1
        
        # Get current game state
        game_state = await self.emulator.async_get_game_state()
        
        if info_type == "player_position":
            return f"Player position: {game_state['position']}"
        elif info_type == "map_id":
            return f"Current map ID: {game_state['map_id']}"
        elif info_type == "pokemon_status":
            if game_state['pokemon_count'] > 0:
                return f"First Pokémon HP: {game_state['first_pokemon_hp']}/{game_state['first_pokemon_max_hp']}"
            else:
                return "No Pokémon in party"
        elif info_type == "battle_status":
            return f"In battle: {game_state['in_battle']}, Text active: {game_state['text_active']}"
        else:
            return f"Unknown info type: {info_type}"
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get action statistics.
        
        Returns:
            Dictionary of action counts
        """
        return self.action_stats
    
    def reset_stats(self) -> None:
        """Reset action statistics."""
        self.action_stats = {}
    
    def register_with_graph(self, graph: StateGraph):
        """
        Register tools with a LangGraph StateGraph.
        
        Args:
            graph: The StateGraph to register tools with
        """
        # Create tool node for button press
        graph.add_node(
            "press_button",
            ToolNode(
                self.press_button,
                tool_input=ButtonPressInput,
                description="Press a button on the Game Boy",
            )
        )
        
        # Create tool node for wait frames
        graph.add_node(
            "wait_frames",
            ToolNode(
                self.wait_frames,
                tool_input=WaitFramesInput,
                description="Wait for a specified number of frames without taking any action",
            )
        )
        
        # Create tool node for update knowledge
        graph.add_node(
            "update_knowledge",
            ToolNode(
                self.update_knowledge,
                tool_input=UpdateKnowledgeInput,
                description="Update the knowledge base with new information",
            )
        )
        
        # Create tool node for get game info
        graph.add_node(
            "get_game_info",
            ToolNode(
                self.get_game_info,
                tool_input=GetGameInfoInput,
                description="Get specific information about the game state",
            )
        )
    
    def build_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Build tool definitions for use with Claude's Tools API.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "name": "press_button",
                "description": "Press a button on the Game Boy",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "button": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right", "a", "b", "start", "select"],
                            "description": "The button to press"
                        },
                        "hold_frames": {
                            "type": "integer",
                            "description": "Number of frames to hold the button",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 30
                        }
                    },
                    "required": ["button"]
                }
            },
            {
                "name": "wait_frames",
                "description": "Wait for a specified number of frames without taking any action",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "num_frames": {
                            "type": "integer",
                            "description": "Number of frames to wait",
                            "default": 30,
                            "minimum": 1,
                            "maximum": 60
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "update_knowledge",
                "description": "Update the knowledge base with new information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "enum": ["game_state", "player_progress", "strategies", "map_data"],
                            "description": "Section to update"
                        },
                        "key": {
                            "type": "string",
                            "description": "Key within the section"
                        },
                        "value": {
                            "type": "string",
                            "description": "Value to store"
                        }
                    },
                    "required": ["section", "key", "value"]
                }
            },
            {
                "name": "get_game_info",
                "description": "Get specific information about the game state",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "info_type": {
                            "type": "string",
                            "enum": ["player_position", "map_id", "pokemon_status", "battle_status"],
                            "description": "Type of information to retrieve"
                        }
                    },
                    "required": ["info_type"]
                }
            }
        ]


class AsyncToolExecutor:
    """
    Executes tools asynchronously based on LLM decisions.
    
    This adapter connects Claude's tool use output format to the actual tool execution,
    handling errors and providing appropriate responses.
    """
    
    def __init__(self, tools_adapter: PokemonToolsAdapter):
        """
        Initialize the tool executor.
        
        Args:
            tools_adapter: PokemonToolsAdapter instance
        """
        self.tools_adapter = tools_adapter
        self.available_tools = {
            "press_button": self.tools_adapter.press_button,
            "wait_frames": self.tools_adapter.wait_frames,
            "update_knowledge": self.tools_adapter.update_knowledge,
            "get_game_info": self.tools_adapter.get_game_info
        }
    
    async def execute_tool_from_assistant_response(self, response_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute tools based on assistant response content.
        
        Args:
            response_content: Content from assistant response
            
        Returns:
            Tool execution results
        """
        tool_results = {}
        
        for content in response_content:
            if content.get("type") == "tool_use":
                tool_name = content.get("name")
                tool_input = content.get("input", {})
                tool_id = content.get("id")
                
                if tool_name in self.available_tools:
                    try:
                        # Execute the tool
                        if tool_name == "press_button":
                            result = await self.tools_adapter.press_button(ButtonPressInput(**tool_input))
                        elif tool_name == "wait_frames":
                            result = await self.tools_adapter.wait_frames(WaitFramesInput(**tool_input))
                        elif tool_name == "update_knowledge":
                            result = await self.tools_adapter.update_knowledge(UpdateKnowledgeInput(**tool_input))
                        elif tool_name == "get_game_info":
                            result = await self.tools_adapter.get_game_info(GetGameInfoInput(**tool_input))
                        else:
                            result = f"Unknown tool: {tool_name}"
                        
                        # Store the result
                        tool_results[tool_id] = {
                            "tool_use_id": tool_id,
                            "name": tool_name,
                            "result": result
                        }
                    except Exception as e:
                        # Handle errors
                        tool_results[tool_id] = {
                            "tool_use_id": tool_id,
                            "name": tool_name,
                            "result": f"Error: {str(e)}"
                        }
        
        return tool_results


# Simple test if the file is run directly
if __name__ == "__main__":
    import sys
    import json
    
    print("Pokemon LangGraph Tools Adapter - Test Mode")
    print("This is a library module and not meant to be run directly.")
    print("However, we can display the tool definitions:")
    
    # Creating a mock emulator and knowledge base for testing
    class MockEmulator:
        async def async_press_button(self, button, hold_frames): 
            return f"Pressed {button}"
        async def async_wait_frames(self, num_frames): 
            return f"Waited {num_frames} frames"
        async def async_get_game_state(self): 
            return {"position": (0, 0), "map_id": 0, "pokemon_count": 0, "in_battle": False, "text_active": False}
    
    class MockKnowledgeBase:
        def update(self, section, key, value): 
            return True
    
    # Create tools with mock objects
    tools = PokemonToolsAdapter(MockEmulator(), MockKnowledgeBase())
    
    # Print tool definitions
    tool_definitions = tools.build_tool_definitions()
    print(json.dumps(tool_definitions, indent=2))