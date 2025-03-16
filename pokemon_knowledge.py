"""
Pokemon Knowledge Base Module - Manages long-term memory for the Pokemon Agent.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Union


class KnowledgeBase:
    """Simple knowledge base to store game information and history."""
    
    def __init__(self, save_path="knowledge_base.json"):
        """
        Initialize the knowledge base.
        
        Args:
            save_path: Path to the JSON file to store knowledge
        """
        self.save_path = save_path
        self.data = {
            "game_state": {},
            "player_progress": {},
            "strategies": {},
            "map_data": {},
            "action_history": []
        }
        self.load()
    
    def load(self):
        """Load knowledge base from disk if it exists."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"Error loading knowledge base: {e}")
    
    def save(self):
        """Save knowledge base to disk."""
        with open(self.save_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def update(self, section, key, value):
        """
        Update a section of the knowledge base.
        
        Args:
            section: Section name (e.g., "game_state", "player_progress")
            key: Key within the section
            value: Value to store
        """
        if section not in self.data:
            self.data[section] = {}
        self.data[section][key] = value
        self.save()
    
    def get(self, section, key=None):
        """
        Get data from the knowledge base.
        
        Args:
            section: Section name to retrieve
            key: Optional key within the section
            
        Returns:
            The requested data, or None if not found
        """
        if section not in self.data:
            return None
        if key is None:
            return self.data[section]
        return self.data[section].get(key)
    
    def add_action(self, action, result):
        """
        Add an action to the history.
        
        Args:
            action: Action that was taken
            result: Result of the action
        """
        self.data["action_history"].append({
            "action": action,
            "result": result,
            "timestamp": time.time()
        })
        # Limit history size
        if len(self.data["action_history"]) > 100:
            self.data["action_history"] = self.data["action_history"][-100:]
        self.save()
    
    def get_recent_actions(self, count=5):
        """
        Get recent actions as formatted text.
        
        Args:
            count: Number of recent actions to retrieve
            
        Returns:
            String of formatted action history
        """
        recent = self.data["action_history"][-count:] if self.data["action_history"] else []
        return "\n".join([f"Action: {a['action']}, Result: {a['result']}" for a in recent])
    
    def add_map_location(self, map_id, location_name, description):
        """
        Add information about a map location.
        
        Args:
            map_id: Map ID number
            location_name: Name of the location
            description: Description of the location
        """
        if "locations" not in self.data["map_data"]:
            self.data["map_data"]["locations"] = {}
            
        self.data["map_data"]["locations"][str(map_id)] = {
            "name": location_name,
            "description": description
        }
        self.save()
    
    def add_path(self, from_map, to_map, button_sequence):
        """
        Add a path between maps.
        
        Args:
            from_map: Starting map ID
            to_map: Destination map ID
            button_sequence: Sequence of buttons to press
        """
        if "paths" not in self.data["map_data"]:
            self.data["map_data"]["paths"] = {}
            
        path_key = f"{from_map}_{to_map}"
        self.data["map_data"]["paths"][path_key] = button_sequence
        self.save()
    
    def get_map_info(self, map_id):
        """
        Get information about a map.
        
        Args:
            map_id: Map ID to retrieve
            
        Returns:
            Map information or None if not found
        """
        if "locations" not in self.data["map_data"]:
            return None
        
        return self.data["map_data"]["locations"].get(str(map_id))
    
    def get_path(self, from_map, to_map):
        """
        Get path between maps.
        
        Args:
            from_map: Starting map ID
            to_map: Destination map ID
            
        Returns:
            Button sequence or None if not found
        """
        if "paths" not in self.data["map_data"]:
            return None
            
        path_key = f"{from_map}_{to_map}"
        return self.data["map_data"]["paths"].get(path_key)
    
    def record_game_progress(self, progress_type, description):
        """
        Record game progress.
        
        Args:
            progress_type: Type of progress (e.g., "badge", "item", "event")
            description: Description of the progress
        """
        if "milestones" not in self.data["player_progress"]:
            self.data["player_progress"]["milestones"] = []
            
        self.data["player_progress"]["milestones"].append({
            "type": progress_type,
            "description": description,
            "timestamp": time.time()
        })
        self.save()
    
    def clear(self, section=None):
        """
        Clear knowledge base data.
        
        Args:
            section: Optional section to clear, or None to clear all
        """
        if section is None:
            self.data = {
                "game_state": {},
                "player_progress": {},
                "strategies": {},
                "map_data": {},
                "action_history": []
            }
        elif section in self.data:
            self.data[section] = {}
            
        self.save()


# Simple test if the file is run directly
if __name__ == "__main__":
    import sys
    
    print("Pokemon Knowledge Base Module - Test Mode")
    
    # Create a test knowledge base
    test_kb = KnowledgeBase("test_knowledge.json")
    
    # Add some test data
    test_kb.update("game_state", "current_location", "Pallet Town")
    test_kb.add_action("press_a", "Talked to NPC")
    test_kb.add_action("press_up", "Moved up")
    test_kb.add_map_location(1, "Pallet Town", "The starting town of your journey")
    
    # Display the data
    print("\nKnowledge Base Contents:")
    print(f"Current location: {test_kb.get('game_state', 'current_location')}")
    print("\nRecent actions:")
    print(test_kb.get_recent_actions())
    print("\nMap data:")
    print(f"Pallet Town info: {test_kb.get_map_info(1)}")
    
    print("\nTest completed. Knowledge saved to test_knowledge.json")