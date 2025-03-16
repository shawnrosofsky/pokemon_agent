"""
Pokemon Tools Module - Contains tool definitions and execution functions
for the Pokemon Agent system.
"""

import time
from typing import Dict, Any, List, Optional, Union


class PokemonTools:
    """
    Tools for interacting with Pokemon Red/Blue game.
    The actual execution is delegated to a GameEmulator instance.
    """
    
    def __init__(self, game_emulator, knowledge_base):
        """
        Initialize tools with a game emulator and knowledge base.
        
        Args:
            game_emulator: GameEmulator instance to execute actions
            knowledge_base: KnowledgeBase instance to store/retrieve information
        """
        self.emulator = game_emulator
        self.kb = knowledge_base
        self.action_stats = {}  # Statistics for tracking tool usage
        self.last_error = None  # Track the most recent error
    
    def define_tools(self) -> List[Dict[str, Any]]:
        """
        Define the tools that Claude can use.
        
        Returns:
            List of tool definitions in the format expected by Claude API
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
                            "default": 10
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
                            "default": 30
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
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> str:
        """
        Execute a tool based on name and parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            
        Returns:
            Result of the tool execution as a string
        """
        try:
            # Call the appropriate method based on tool name
            if tool_name == "press_button":
                return self._press_button(params)
            elif tool_name == "wait_frames":
                return self._wait_frames(params)
            elif tool_name == "update_knowledge":
                return self._update_knowledge(params)
            elif tool_name == "get_game_info":
                return self._get_game_info(params)
            else:
                error_msg = f"Unknown tool: {tool_name}. Using wait fallback."
                self.last_error = error_msg
                return self.emulator.wait_frames(30)
                
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            self.last_error = error_msg
            print(error_msg)
            return self.emulator.wait_frames(30)
    
    def _press_button(self, params: Dict[str, Any]) -> str:
        """Execute the press_button tool."""
        button = params.get("button", "")
        hold_frames = params.get("hold_frames", 10)
        
        # Track action stats
        if button in self.action_stats:
            self.action_stats[button] += 1
        else:
            self.action_stats[button] = 1
        
        return self.emulator.press_button(button, hold_frames)
    
    def _wait_frames(self, params: Dict[str, Any]) -> str:
        """Execute the wait_frames tool."""
        num_frames = params.get("num_frames", 30)
        
        # Track stats
        if "wait" in self.action_stats:
            self.action_stats["wait"] += 1
        else:
            self.action_stats["wait"] = 1
        
        return self.emulator.wait_frames(num_frames)
    
    def _update_knowledge(self, params: Dict[str, Any]) -> str:
        """Execute the update_knowledge tool."""
        section = params.get("section", "")
        key = params.get("key", "")
        value = params.get("value", "")
        
        # Track stats
        if "update_kb" in self.action_stats:
            self.action_stats["update_kb"] += 1
        else:
            self.action_stats["update_kb"] = 1
        
        self.kb.update(section, key, value)
        return f"Updated knowledge base: {section}.{key} = {value}"
    
    def _get_game_info(self, params: Dict[str, Any]) -> str:
        """Execute the get_game_info tool."""
        info_type = params.get("info_type", "")
        
        # Track stats
        if "get_info" in self.action_stats:
            self.action_stats["get_info"] += 1
        else:
            self.action_stats["get_info"] = 1
        
        # Get current game state
        game_state = self.emulator.get_game_state()
        
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


# Simple test if the file is run directly
if __name__ == "__main__":
    import sys
    import json
    
    print("Pokemon Tools Module - Test Mode")
    print("This is a library module and not meant to be run directly.")
    print("However, we can display the tool definitions:")
    
    # Creating a mock emulator and knowledge base for testing
    class MockEmulator:
        def press_button(self, button, hold_frames): return f"Pressed {button}"
        def wait_frames(self, num_frames): return f"Waited {num_frames} frames"
        def get_game_state(self): return {"position": (0, 0), "map_id": 0, "pokemon_count": 0, "in_battle": False, "text_active": False}
    
    class MockKnowledgeBase:
        def update(self, section, key, value): return True
    
    # Create tools with mock objects
    tools = PokemonTools(MockEmulator(), MockKnowledgeBase())
    
    # Print tool definitions
    tool_definitions = tools.define_tools()
    print(json.dumps(tool_definitions, indent=2))