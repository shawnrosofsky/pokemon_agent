"""
LangGraph Knowledge Base Module - Enhanced knowledge store for LangGraph Pokémon Agent
with cross-thread memory capabilities.
"""

import os
import json
import time
import datetime
from typing import Dict, List, Any, Optional, Union, TypedDict
from pydantic import BaseModel, Field

# Import LangGraph memory components
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore


class MemoryItem(BaseModel):
    """Model for a memory item."""
    content: str
    timestamp: float = Field(default_factory=time.time)
    context: Optional[str] = None
    importance: int = 1  # Higher = more important
    type: str = "general"  # general, event, progress, etc.


class RecentActionItem(BaseModel):
    """Model for a recent action."""
    action: str
    result: str
    timestamp: float = Field(default_factory=time.time)


class KnowledgeBase:
    """Enhanced knowledge base to store game information and history with cross-thread memory."""
    
    def __init__(self, save_path: str = "knowledge_base.json", store: Optional[Any] = None):
        """
        Initialize the knowledge base.
        
        Args:
            save_path: Path to the JSON file to store knowledge
            store: Optional InMemoryStore for cross-thread memory
        """
        self.save_path = save_path
        self.data = {
            "game_state": {},
            "player_progress": {},
            "strategies": {},
            "map_data": {},
            "action_history": []
        }
        
        # Load existing knowledge from file if available
        self.load()
        
        # Setup in-memory store for cross-thread memory if provided
        self.store = store or InMemoryStore()
        
        # Create namespaces for different types of memories
        self.namespaces = {
            "action_history": "actions",
            "long_term_memory": "memories",
            "progress_milestones": "milestones",
            "map_knowledge": "map_data",
        }
        
        # Initialize namespaces
        self._ensure_namespaces()
    
    def _ensure_namespaces(self):
        """Ensure all namespaces exist in the store."""
        # In the current LangGraph version, namespaces are automatically created
        # when you first put something in them, so we don't need to explicitly
        # initialize them. This method is kept for compatibility.
        pass
    
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
    
    def update(self, section: str, key: str, value: Any):
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
        
        # Also store in cross-thread memory for important sections
        if section in ["player_progress", "map_data"]:
            memory_item = MemoryItem(
                content=f"{key}: {value}",
                context=section,
                type=section
            )
            self.add_memory(memory_item, section)
    
    def get(self, section: str, key: Optional[str] = None):
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
    
    def add_action(self, action: str, result: str):
        """
        Add an action to the history.
        
        Args:
            action: Action that was taken
            result: Result of the action
        """
        action_item = {
            "action": action,
            "result": result,
            "timestamp": time.time()
        }
        
        # Add to local store
        self.data["action_history"].append(action_item)
        
        # Limit history size in local store
        if len(self.data["action_history"]) > 100:
            self.data["action_history"] = self.data["action_history"][-100:]
        self.save()
        
        # Add to cross-thread memory
        self.store.put(
            self.namespaces["action_history"],
            str(time.time()),
            RecentActionItem(action=action, result=result).dict(),
            index=["action", "result"]
        )
    
    def get_recent_actions(self, count: int = 5) -> str:
        """
        Get recent actions as formatted text.
        
        Args:
            count: Number of recent actions to retrieve
            
        Returns:
            String of formatted action history
        """
        recent = self.data["action_history"][-count:] if self.data["action_history"] else []
        return "\n".join([f"Action: {a['action']}, Result: {a['result']}" for a in recent])
    
    def add_memory(self, memory: Union[MemoryItem, str], context: Optional[str] = None):
        """
        Add a memory to the cross-thread memory store.
        
        Args:
            memory: Memory item or string content to store
            context: Optional context for the memory
        """
        if isinstance(memory, str):
            memory = MemoryItem(content=memory, context=context)
        
        # Store in cross-thread memory
        self.store.put(
            self.namespaces["long_term_memory"],
            str(time.time()),
            memory.dict(),
            index=["content", "type"]
        )
    
    def search_memories(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """
        Search memories using semantic search.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of memory items matching the query
        """
        # For now, simple text search since we don't have embeddings
        results = self.store.search(
            self.namespaces["long_term_memory"],
            {"content": {"$contains": query}},
            limit=limit
        )
        
        return [MemoryItem(**item) for item in results]
    
    def add_map_location(self, map_id: int, location_name: str, description: str):
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
        
        # Store in cross-thread memory
        self.store.put(
            self.namespaces["map_knowledge"],
            f"map_{map_id}",
            {
                "map_id": map_id,
                "name": location_name,
                "description": description
            },
            index=["name", "description"]
        )
    
    def add_path(self, from_map: int, to_map: int, button_sequence: str):
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
        
        # Store in cross-thread memory
        self.store.put(
            self.namespaces["map_knowledge"],
            f"path_{from_map}_{to_map}",
            {
                "from_map": from_map,
                "to_map": to_map,
                "button_sequence": button_sequence
            },
            index=["button_sequence"]
        )
    
    def get_map_info(self, map_id: int) -> Optional[Dict[str, Any]]:
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
    
    def get_path(self, from_map: int, to_map: int) -> Optional[str]:
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
    
    def record_milestone(self, milestone_type: str, description: str, importance: int = 3):
        """
        Record a game progress milestone.
        
        Args:
            milestone_type: Type of milestone (e.g., "badge", "item", "event")
            description: Description of the milestone
            importance: Importance level (1-5, with 5 being most important)
        """
        if "milestones" not in self.data["player_progress"]:
            self.data["player_progress"]["milestones"] = []
            
        milestone = {
            "type": milestone_type,
            "description": description,
            "timestamp": time.time(),
            "importance": importance
        }
        
        self.data["player_progress"]["milestones"].append(milestone)
        self.save()
        
        # Store in cross-thread memory
        self.store.put(
            self.namespaces["progress_milestones"],
            f"{milestone_type}_{time.time()}",
            milestone,
            index=["type", "description"]
        )
        
        # Also add as a memory with high importance
        self.add_memory(
            MemoryItem(
                content=f"Milestone: {description}",
                type="milestone",
                importance=importance + 1  # Make milestone memories more important
            )
        )
    
    def get_key_memories(self, limit: int = 3) -> str:
        """
        Get the most important memories for context.
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            String of formatted memories
        """
        # Get the most recent milestones by importance
        milestones = self.store.list(
            self.namespaces["progress_milestones"],
            sort_by="timestamp",
            descending=True,
            limit=limit
        )
        
        if not milestones:
            return "No significant milestones recorded yet."
        
        # Format milestones as text
        memory_text = "Key progress:\n"
        for milestone in milestones:
            memory_text += f"- {milestone['description']}\n"
        
        return memory_text
    
    def clear(self, section: Optional[str] = None):
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


# Test code
if __name__ == "__main__":
    # Create a test knowledge base with cross-thread memory
    store = InMemoryStore()
    test_kb = KnowledgeBase("test_knowledge.json", store=store)
    
    # Add some test data
    test_kb.update("game_state", "current_location", "Pallet Town")
    test_kb.add_action("press_a", "Talked to NPC")
    test_kb.add_action("press_up", "Moved up")
    test_kb.add_map_location(1, "Pallet Town", "The starting town of your journey")
    
    # Add a memory
    test_kb.add_memory("The player obtained their first Pokémon, a Bulbasaur", "progress")
    
    # Record a milestone
    test_kb.record_milestone("starter", "Chose Bulbasaur as starter Pokémon", 5)
    
    # Display the data
    print("\nKnowledge Base Contents:")
    print(f"Current location: {test_kb.get('game_state', 'current_location')}")
    print("\nRecent actions:")
    print(test_kb.get_recent_actions())
    print("\nMap data:")
    print(f"Pallet Town info: {test_kb.get_map_info(1)}")
    print("\nKey memories:")
    print(test_kb.get_key_memories())
    
    print("\nTest completed. Knowledge saved to test_knowledge.json")