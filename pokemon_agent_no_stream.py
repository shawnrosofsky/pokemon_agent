"""
Pokemon Agent - Main module for the Pokemon AI agent system.
Integrates the emulator, knowledge base, and tools with Claude API.
"""

import os
import re
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
        print(text, end=end, flush=flush)
        if self.output_to_file:
            with open(self.log_file, 'a') as f:
                f.write(text)
                if end:
                    f.write(end)
    
    def print_section(self, title, content=None):
        separator = "=" * 50
        self.print(f"\n{separator}")
        self.print(title)
        self.print(separator)
        if content:
            self.print(content)
            self.print(separator)


class PokemonAgent:
    """LLM Agent for playing Pokémon using Anthropic Claude."""
    
    def __init__(self, rom_path, model_name="claude-3-7-sonnet-20250219", 
                 api_key=None, max_tokens=2000, temperature=1.0, headless=False, speed=1,
                 sound=True, output_to_file=False, log_file=None):
        # Setup API
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set ANTHROPIC_API_KEY env var or pass directly.")
        self.max_tokens = max_tokens
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
        try:
            # Get game state and format it
            game_state = self.emulator.get_game_state()
            screen_base64 = self.emulator.get_screen_base64()
            state_text = self.emulator.format_game_state(game_state)
            recent_actions = self.kb.get_recent_actions()
            
            # Get tool definitions and available tool names
            tools = self.tools_manager.define_tools()
            tool_names = ", ".join([tool["name"] for tool in tools])
            
            # System prompt instructs Claude to return a JSON array with one tool call
            system_prompt = f"""
You are an expert Pokémon player. Analyze the current game state and decide the best action.
Your final answer MUST be a JSON array containing exactly one tool call in the following format:
[{{"type": "tool_use", "name": "<tool_name>", "input": {{ ... }} }}]
Available tools: {tool_names}.
If uncertain, use "wait_frames". Do not include any extra text outside the JSON.
"""
            # Build the user message content
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

Analyze the situation and output a JSON array with one tool call as described.
"""
                }
            ]
            
            # Validate and trim conversation history (same as before)
            valid_messages = []
            if self.message_history:
                valid = True
                tool_use_positions = []
                for i, msg in enumerate(self.message_history):
                    if msg.get("role") == "assistant" and any(
                        isinstance(c, dict) and c.get("type") == "tool_use"
                        for c in msg.get("content", []) if isinstance(c, dict)):
                        tool_use_positions.append(i)
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
                    valid_messages = self.message_history[-4:] if len(self.message_history) > 4 else self.message_history
                else:
                    self.output.print("Invalid message history pattern detected, resetting conversation")
                    self.message_history = []
                    if self.last_summary:
                        valid_messages = [{
                            "role": "assistant",
                            "content": [{
                                "type": "text",
                                "text": f"Previous progress summary: {self.last_summary}"
                            }]
                        }]
            
            messages = valid_messages + [{
                "role": "user",
                "content": message_content
            }]
            
            # Make non-streaming API call to Claude
            self.output.print_section("CLAUDE'S THOUGHT PROCESS")
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                tools=tools
            )
            
            # Process the full response: concatenate text blocks and capture tool call JSON
            full_response_text = ""
            tool_call = None
            for block in response.content:
                if block.type == "text":
                    full_response_text += block.text
                elif block.type == "tool_use":
                    try:
                        parsed_tool = {
                            "type": "tool_use",
                            "name": block.name,
                            "input": block.input if hasattr(block, "input") else {}
                        }
                        if parsed_tool["name"]:
                            full_response_text = json.dumps([parsed_tool])
                            tool_call = parsed_tool
                            break
                    except Exception:
                        pass
            
            self.output.print("\nFull response from Claude:")
            self.output.print(full_response_text)
            
            # Extract JSON array using regex
            try:
                match = re.search(r'(\[.*\])', full_response_text, re.DOTALL)
                if match:
                    json_part = match.group(1)
                    parsed_response = json.loads(json_part)
                    if isinstance(parsed_response, list):
                        for item in parsed_response:
                            if isinstance(item, dict) and item.get("type") == "tool_use" and item.get("name"):
                                tool_call = item
                                break
                else:
                    self.output.print("No JSON array found in Claude's response.")
            except Exception as e:
                self.output.print(f"Error parsing JSON: {e}")
            
            # Update conversation history with the full response
            self.message_history.append({
                "role": "user",
                "content": message_content
            })
            assistant_content = []
            if full_response_text.strip():
                assistant_content.append({"type": "text", "text": full_response_text})
            if tool_call and tool_call.get("name"):
                assistant_tool = {
                    "type": "tool_use",
                    "name": tool_call.get("name"),
                    "input": tool_call.get("input", {})
                }
                # Only include "id" if it is a non-empty string
                if tool_call.get("id") and isinstance(tool_call.get("id"), str) and tool_call.get("id").strip():
                    assistant_tool["id"] = tool_call.get("id")
                assistant_content.append(assistant_tool)
            if assistant_content:
                self.message_history.append({
                    "role": "assistant",
                    "content": assistant_content
                })
            
            # Execute the tool call if detected
            if tool_call and tool_call.get("name"):
                tool_name = tool_call.get("name")
                tool_params = tool_call.get("input", {})
                self.output.print_section("ACTION EXECUTION", f"Executing {tool_name} with params: {tool_params}")
                result = self.tools_manager.execute_tool(tool_name, tool_params)
                self.output.print(f"Result: {result}")
                self.kb.add_action(tool_name, result)
                user_msg = {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call.get("id"),
                        "content": result
                    }]
                }
                self.message_history.append(user_msg)
                return tool_name, result
            else:
                self.output.print_section("ACTION EXECUTION", "No tool call detected, using wait fallback")
                result = self.emulator.wait_frames(30)
                self.kb.add_action("wait", result)
                self.message_history.append({
                    "role": "user",
                    "content": [{"type": "text", "text": "I couldn't determine a specific action to take. Please help."}]
                })
                return "wait", result
            
        except Exception as e:
            error_msg = f"Error getting action: {e}"
            self.output.print_section("ERROR", error_msg)
            traceback_str = traceback.format_exc()
            self.output.print(traceback_str)
            self.last_error = str(e)
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
                game_state = self.emulator.get_game_state()
                state_text = self.emulator.format_game_state(game_state)
                summary_response = self.client.messages.create(
                    model=self.model,
                    messages=[{
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
                    }],
                    max_tokens=500, # Make this an argument in future
                    temperature=1.0 # Make this an argument in future
                )
                summary_text = ""
                for content in summary_response.content:
                    if content.type == "text":
                        summary_text = content.text
                        break
                self.last_summary = summary_text
                self.kb.update("player_progress", "last_summary", summary_text)
                self.turn_count = 0
                self.message_history = [{
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": f"Previous progress summary: {self.last_summary}"
                    }]
                }]
                self.output.print_section("SUMMARY", summary_text)
                self.output.print("Context reset with summary")
            except Exception as e:
                self.output.print_section("SUMMARIZATION ERROR", str(e))
                self.turn_count = self.max_turns_before_summary - 5
    
    def run(self, num_steps=None):
        """Run the agent for specified steps."""
        step_count = 0
        try:
            self.output.print_section("STARTING POKÉMON AGENT", f"Model: {self.model}, Temperature: {self.temperature}")
            while num_steps is None or step_count < num_steps:
                action, result = self.get_action()
                step_count += 1
                self.output.print_section(f"STEP {step_count} SUMMARY", f"Action: {action}\nResult: {result}")
                if step_count % 10 == 0:
                    stats_text = "Action distribution:\n"
                    stats = self.tools_manager.get_stats()
                    total = sum(stats.values()) or 1
                    for act, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                        stats_text += f"  {act}: {count} ({count/total*100:.1f}%)\n"
                    self.output.print_section(f"STATISTICS AFTER {step_count} STEPS", stats_text)
                if step_count % 50 == 0:
                    save_path = self.emulator.save_state(f"step_{step_count}.state")
                    self.output.print(f"Saved game state to {save_path}")
                self.check_for_summarization()
                time.sleep(0.01)
        except KeyboardInterrupt:
            self.output.print_section("INTERRUPTED", "Stopped by user")
        except Exception as e:
            self.output.print_section("ERROR", f"Error running agent: {e}")
            traceback_str = traceback.format_exc()
            self.output.print(traceback_str)
        finally:
            self.emulator.save_state("final.state")
            with open("action_stats.json", 'w') as f:
                json.dump(self.tools_manager.get_stats(), f, indent=2)
            self.output.print_section("RUN COMPLETE", f"Total steps: {step_count}")
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
    parser.add_argument('--max_tokens', default=2000, help='Max tokens for Claude response')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no window)')
    parser.add_argument('--steps', type=int, help='Number of steps to run (infinite if not specified)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for the LLM')
    parser.add_argument('--speed', type=int, default=1, help='Game speed multiplier')
    parser.add_argument('--no-sound', action='store_true', help='Disable game sound')
    parser.add_argument('--log-to-file', action='store_true', help='Log output to file')
    parser.add_argument('--log-file', help='Path to log file (default: pokemon_agent.log)')
    args = parser.parse_args()
    
    agent = PokemonAgent(
        rom_path=args.rom_path,
        model_name=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
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
