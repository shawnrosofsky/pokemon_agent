"""
Pokemon Agent - Main module for the Pokemon AI agent system.
Integrates the emulator, knowledge base, and tools with Claude API.
"""

import os
import time
import json
import traceback
import anthropic
from typing import Dict, List, Any, Optional, Tuple

# Import our modules
from pokemon_emulator import GameEmulator
from pokemon_knowledge import KnowledgeBase
from pokemon_tools import PokemonTools


class OutputManager:
    """Manages output of Claude's thoughts and agent actions."""
    
    def __init__(self, output_to_file=False, log_file=None):
        self.output_to_file = output_to_file
        self.log_file = log_file or "pokemon_agent.log"
        
        # Create log file if needed
        if self.output_to_file:
            with open(self.log_file, 'w') as f:
                f.write(f"=== Pokémon Agent Log - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    def print(self, text, end="\n", flush=True):
        """Print text to console and optionally to file."""
        print(text, end=end, flush=flush)
        
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
    """LLM Agent for playing Pokémon using Anthropic Claude."""
    
    def __init__(self, rom_path, model_name="claude-3-7-sonnet-20250219", 
                 api_key=None, temperature=0.7, headless=False, speed=1,
                 sound=True, output_to_file=False, log_file=None):
        """Initialize the agent."""
        # Setup API
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set ANTHROPIC_API_KEY env var or pass directly.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model_name
        self.temperature = temperature
        
        # Setup components
        self.emulator = GameEmulator(rom_path, headless=headless, speed=speed, sound=sound)
        self.kb = KnowledgeBase()
        self.tools_manager = PokemonTools(self.emulator, self.kb)
        
        # Setup output manager
        self.output = OutputManager(output_to_file=output_to_file, log_file=log_file)
        
        # Conversation management
        self.message_history = []
        self.turn_count = 0
        self.max_turns_before_summary = 10
        self.last_summary = ""
        
        # Last error tracking
        self.last_error = None
    
    def start(self):
        """Start the game."""
        self.emulator.start_game(skip_intro=True)
    
    def get_action(self) -> Tuple[str, str]:
        """
        Get current game state, analyze with Claude, and execute the action in a single step.
        This function combines analysis and action execution for efficiency.
        Uses streaming to show Claude's thought process in real-time.
        
        Returns:
            Tuple of (action, result)
        """
        try:
            # Get game state
            game_state = self.emulator.get_game_state()
            screen_base64 = self.emulator.get_screen_base64()
            state_text = self.emulator.format_game_state(game_state)
            recent_actions = self.kb.get_recent_actions()
            
            # System prompt
            system_prompt = """
You are an expert Pokémon player. Analyze the game state and use a tool to take the best action.

First, think step-by-step about:
1. Where the player is currently and what's visible on screen
2. What progress has been made in the game so far
3. What the immediate goal should be
4. Available actions and their potential outcomes

Then, use a tool to execute the best action. Always use a tool - do not just describe what to do.
If you're unsure what to do, use wait_frames instead of pressing buttons randomly.
"""

            # Build message content
            message_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screen_base64
                    }
                },
                {
                    "type": "text",
                    "text": f"""
Current game state:
{state_text}

Recent actions:
{recent_actions}

{f"Note about previous action: {self.last_error}" if self.last_error else ""}

First, analyze what you see on screen and the current situation.
Then, use a tool to take the most appropriate action.
"""
                }
            ]

            # Clean and validate message history
            valid_messages = []
            if self.message_history:
                # Check for proper message sequence
                valid = True
                tool_use_positions = []
                
                # Find all tool_use positions
                for i, msg in enumerate(self.message_history):
                    if msg.get("role") == "assistant" and any(
                        isinstance(c, dict) and c.get("type") == "tool_use" 
                        for c in msg.get("content", []) if isinstance(c, dict)):
                        tool_use_positions.append(i)
                
                # Verify each tool_use has a corresponding tool_result
                for pos in tool_use_positions:
                    if pos + 1 >= len(self.message_history) or self.message_history[pos + 1].get("role") != "user":
                        valid = False
                        break
                    user_msg = self.message_history[pos + 1]
                    has_tool_result = any(
                        isinstance(c, dict) and c.get("type") == "tool_result" 
                        for c in user_msg.get("content", []) if isinstance(c, dict))
                    if not has_tool_result:
                        valid = False
                        break
                
                if valid:
                    # Use last few exchanges if valid
                    valid_messages = self.message_history[-4:] if len(self.message_history) > 4 else self.message_history
                else:
                    # Reset history if invalid
                    self.output.print("Invalid message history pattern detected, resetting conversation")
                    self.message_history = []
                    
                    # Insert summary if available
                    if self.last_summary:
                        valid_messages = [
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Previous progress summary: {self.last_summary}"
                                    }
                                ]
                            }
                        ]
            
            # Add current message
            messages = valid_messages + [
                {
                    "role": "user",
                    "content": message_content
                }
            ]
            
            # Get tool definitions
            tools = self.tools_manager.define_tools()
            
            # Call Claude with streaming enabled
            self.output.print_section("CLAUDE'S THOUGHT PROCESS")
            
            stream = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=messages,
                max_tokens=1000,
                temperature=self.temperature,
                tools=tools,
                stream=True
            )
            
            # Process the stream
            accumulated_content = []
            current_text = ""
            current_block_type = None
            current_tool_use = None
            
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text":
                        # For text blocks
                        current_text += event.delta.text
                        self.output.print(event.delta.text, end="", flush=True)
                    elif event.delta.type == "tool_use":
                        # For tool use blocks
                        if event.delta.name:
                            self.output.print(f"\n[Using tool: {event.delta.name}]", end="", flush=True)
                            if current_tool_use is None:
                                current_tool_use = {"type": "tool_use", "name": event.delta.name}
                            else:
                                current_tool_use["name"] = event.delta.name
                        if event.delta.input:
                            self.output.print(f" with parameters: {event.delta.input}", end="", flush=True)
                            if current_tool_use is None:
                                current_tool_use = {"type": "tool_use", "input": event.delta.input}
                            else:
                                current_tool_use["input"] = event.delta.input
                        if event.delta.id:
                            if current_tool_use is None:
                                current_tool_use = {"type": "tool_use", "id": event.delta.id}
                            else:
                                current_tool_use["id"] = event.delta.id
                
                elif event.type == "message_delta":
                    # Message complete
                    pass
                
                elif event.type == "content_block_start":
                    # Start of a new content block
                    current_block_type = event.content_block.type
                    if current_block_type == "tool_use":
                        self.output.print(f"\n[Starting to use a tool]", end="", flush=True)
                        current_tool_use = {"type": "tool_use"}
                        # Only add if it has an ID
                        if hasattr(event.content_block, 'id') and event.content_block.id:
                            current_tool_use["id"] = event.content_block.id
                
                elif event.type == "content_block_stop":
                    # End of a content block
                    if current_block_type == "text" and current_text:
                        accumulated_content.append({
                            "type": "text",
                            "text": current_text
                        })
                        current_text = ""
                    elif current_block_type == "tool_use" and current_tool_use:
                        accumulated_content.append(current_tool_use)
                        current_tool_use = None
                    
                    current_block_type = None
            
            self.output.print("")  # End the streaming output with a newline
            
            # Add to conversation history
            self.message_history.append({
                "role": "user",
                "content": message_content
            })
            self.message_history.append({
                "role": "assistant", 
                "content": accumulated_content
            })
            
            # Process any tool calls
            tool_call = None
            for content in accumulated_content:
                if content.get("type") == "tool_use":
                    tool_call = content
                    break
            
            if tool_call:
                # Execute the tool
                tool_name = tool_call.get("name")
                tool_params = tool_call.get("input", {})
                tool_id = tool_call.get("id")
                
                self.output.print_section("ACTION EXECUTION")
                self.output.print(f"Executing {tool_name} with params: {tool_params}")
                result = self.tools_manager.execute_tool(tool_name, tool_params)
                self.output.print(f"Result: {result}")
                
                # Track action taken
                action = tool_name
                if tool_name == "press_button" and "button" in tool_params:
                    action = tool_params["button"]
                
                # Add to knowledge base
                self.kb.add_action(action, result)
                
                # Add tool result to conversation history
                user_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result
                        }
                    ]
                }
                self.message_history.append(user_msg)
                
                return action, result
            else:
                # No tool call, default to waiting
                self.output.print_section("ACTION EXECUTION")
                self.output.print("No tool call detected, using wait fallback")
                result = self.emulator.wait_frames(30)
                self.kb.add_action("wait", result)
                
                # Add a standard response to keep conversation flowing
                self.message_history.append({
                    "role": "user",
                    "content": [{"type": "text", "text": "I couldn't determine a specific action to take. Please help."}]
                })
                
                return "wait", result
            
        except Exception as e:
            error_msg = f"Error getting action: {e}"
            self.output.print_section("ERROR")
            self.output.print(error_msg)
            traceback_str = traceback.format_exc()
            self.output.print(traceback_str)
            self.last_error = str(e)
            
            # Fallback to wait
            try:
                result = self.emulator.wait_frames(30)
                self.kb.add_action("wait", "Error fallback: " + str(e))
                return "wait", result
            except:
                return "error", "Failed to execute wait fallback"
    
    def check_for_summarization(self):
        """Check if we need to summarize and reset conversation."""
        self.turn_count += 1
        
        if self.turn_count >= self.max_turns_before_summary:
            try:
                self.output.print_section("GENERATING SUMMARY", f"After {self.turn_count} turns")
                
                # Get game state for additional context
                game_state = self.emulator.get_game_state()
                state_text = self.emulator.format_game_state(game_state)
                
                # Request a summary from Claude
                summary_response = self.client.messages.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""
Please summarize the recent game progress:

Current game state:
{state_text}

Recent actions:
{self.kb.get_recent_actions(20)}

Previous summary (if any):
{self.last_summary}

Summarize:
1. Current location and objective
2. Recent battles and encounters
3. Party status
4. Progress towards game goals

Keep the summary concise but informative.
"""
                        }
                    ],
                    max_tokens=500,
                    temperature=1.0
                )
                
                # Extract summary text
                summary_text = ""
                for content in summary_response.content:
                    if content.type == "text":
                        summary_text = content.text
                        break
                
                # Update last summary
                self.last_summary = summary_text
                
                # Store in knowledge base
                self.kb.update("player_progress", "last_summary", summary_text)
                
                # Reset conversation
                self.turn_count = 0
                self.message_history = [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Previous progress summary: {self.last_summary}"
                            }
                        ]
                    }
                ]
                
                self.output.print_section("SUMMARY", summary_text)
                self.output.print("Context reset with summary")
                
            except Exception as e:
                self.output.print_section("SUMMARIZATION ERROR", str(e))
                # Continue without resetting if summarization fails
                self.turn_count = self.max_turns_before_summary - 5  # Try again in 5 turns
    
    def run(self, num_steps=None):
        """Run the agent for specified steps."""
        step_count = 0
        
        try:
            self.output.print_section("STARTING POKÉMON AGENT", f"Model: {self.model}, Temperature: {self.temperature}")
            
            while num_steps is None or step_count < num_steps:
                # Get action in a single step
                action, result = self.get_action()
                
                # Increment counter
                step_count += 1
                
                # Print progress
                self.output.print_section(f"STEP {step_count} SUMMARY", f"Action: {action}\nResult: {result}")
                
                # Show stats every 10 steps
                if step_count % 10 == 0:
                    stats_text = "Action distribution:\n"
                    stats = self.tools_manager.get_stats()
                    total = sum(stats.values()) or 1
                    for act, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                        stats_text += f"  {act}: {count} ({count/total*100:.1f}%)\n"
                    self.output.print_section(f"STATISTICS AFTER {step_count} STEPS", stats_text)
                
                # Save state periodically
                if step_count % 50 == 0:
                    save_path = self.emulator.save_state(f"step_{step_count}.state")
                    self.output.print(f"Saved game state to {save_path}")
                
                # Check for summarization
                self.check_for_summarization()
                
                # Small delay
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            self.output.print_section("INTERRUPTED", "Stopped by user")
        except Exception as e:
            self.output.print_section("ERROR", f"Error running agent: {e}")
            traceback_str = traceback.format_exc()
            self.output.print(traceback_str)
        finally:
            # Save final state
            self.emulator.save_state("final.state")
            
            # Save action stats
            with open("action_stats.json", 'w') as f:
                json.dump(self.tools_manager.get_stats(), f, indent=2)
            
            self.output.print_section("RUN COMPLETE", f"Total steps: {step_count}")
            
            # Clean up
            self.emulator.close()
    
    def close(self):
        """Clean up resources."""
        self.emulator.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a Pokémon AI agent using Claude')
    parser.add_argument('rom_path', help='Path to the Pokémon ROM file')
    parser.add_argument('--api-key', help='API key (will use ANTHROPIC_API_KEY env var if not provided)')
    parser.add_argument('--model', default='claude-3-7-sonnet-20250219', help='Model to use')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no window)')
    parser.add_argument('--steps', type=int, help='Number of steps to run (infinite if not specified)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for the LLM')
    parser.add_argument('--speed', type=int, default=1, help='Game speed multiplier')
    parser.add_argument('--no-sound', action='store_true', help='Disable game sound')
    parser.add_argument('--log-to-file', action='store_true', help='Log output to file')
    parser.add_argument('--log-file', help='Path to log file (default: pokemon_agent.log)')
    
    args = parser.parse_args()
    
    # Create and run the agent
    agent = PokemonAgent(
        rom_path=args.rom_path,
        model_name=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        headless=args.headless,
        speed=args.speed,
        sound=not args.no_sound,
        output_to_file=args.log_to_file,
        log_file=args.log_file
    )
    
    try:
        agent.start()
        agent.run(num_steps=args.steps)
    finally:
        agent.close()