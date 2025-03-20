#!/usr/bin/env python3
"""
Test script for Anthropic API streaming tool calls.

This script isolates the streaming tool call functionality to diagnose issues
with the Anthropic API when using tools with streaming enabled.
"""

import os
import sys
import json
import time
import anthropic
from typing import Dict, Any, List

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(text, color):
    """Print text with color."""
    print(f"{color}{text}{Colors.ENDC}")

def print_section(title):
    """Print a section header."""
    print("\n" + "="*50)
    print_colored(title, Colors.BOLD + Colors.BLUE)
    print("="*50)

def print_json(obj):
    """Pretty print a JSON object."""
    print(json.dumps(obj, indent=2))

def print_event_details(event, prefix=""):
    """Print details about an event for debugging."""
    print(f"{prefix}Event type: {event.type}")
    
    # Print all available attributes
    for attr_name in dir(event):
        if not attr_name.startswith('_') and attr_name != 'type':
            try:
                attr_value = getattr(event, attr_name)
                if not callable(attr_value):
                    print(f"{prefix}{attr_name}: {attr_value}")
            except Exception as e:
                print(f"{prefix}{attr_name}: Error accessing: {e}")

def define_tools():
    """Define simple tools for testing."""
    return [
        {
            "name": "calculator",
            "description": "Perform a calculation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The operation to perform"
                    },
                    "num1": {
                        "type": "number",
                        "description": "First number"
                    },
                    "num2": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["operation", "num1", "num2"]
            }
        },
        {
            "name": "weather",
            "description": "Get weather information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for"
                    }
                },
                "required": ["location"]
            }
        }
    ]

def execute_tool(name, params):
    """Mock tool execution."""
    print_colored(f"Executing tool: {name} with params: {params}", Colors.GREEN)
    
    if name == "calculator":
        operation = params.get("operation")
        num1 = params.get("num1")
        num2 = params.get("num2")
        
        if operation == "add":
            result = num1 + num2
        elif operation == "subtract":
            result = num1 - num2
        elif operation == "multiply":
            result = num1 * num2
        elif operation == "divide":
            if num2 == 0:
                return "Error: Division by zero"
            result = num1 / num2
        else:
            return "Error: Unknown operation"
            
        return f"Result of {num1} {operation} {num2} = {result}"
    
    elif name == "weather":
        location = params.get("location")
        return f"Mock weather for {location}: 72°F and Sunny"
    
    else:
        return f"Error: Unknown tool {name}"

def test_streaming_tool_call(api_key, verbose=True):
    """Test streaming tool calls with Anthropic API."""
    client = anthropic.Anthropic(api_key=api_key)
    tools = define_tools()
    
    print_section("SETUP")
    print(f"Model: claude-3-7-sonnet-20250219")
    print(f"Tools defined: {[t['name'] for t in tools]}")
    
    # System prompt
    system_prompt = """
You are a helpful assistant that can use tools.
When asked a question that requires calculation or weather information, use the appropriate tool.
"""

    # User prompt
    user_prompt = "What is 42 * 56? After you answer that, tell me the weather in Tokyo."
    
    print_section("SENDING REQUEST")
    print(f"User prompt: {user_prompt}")
    
    # Start timer
    start_time = time.time()
    
    try:
        print_section("STREAMING RESPONSE")
        print_colored("Claude is thinking...", Colors.CYAN)
        
        # Create streaming request
        stream = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens=1000,
            temperature=0.7,
            tools=tools,
            stream=True
        )
        
        # Variables to track streaming state
        accumulated_content = []
        current_text = ""
        current_tool_use = None
        event_count = 0
        tool_uses = []
        current_block_type = None
        
        # Log each event type if in verbose mode
        for event in stream:
            event_count += 1
            
            if verbose:
                print(f"\nEvent {event_count}:")
                print_event_details(event, "  ")
            
            # Handle different event types
            if event.type == "content_block_start":
                # Start of a new content block
                if hasattr(event, 'content_block') and hasattr(event.content_block, 'type'):
                    current_block_type = event.content_block.type
                    if verbose:
                        print(f"  Starting block type: {current_block_type}")
                    
                    if current_block_type == "text":
                        current_text = ""
                    elif current_block_type == "tool_use":
                        print_colored(f"\n[Starting tool use]", Colors.YELLOW)
                        current_tool_use = {"type": "tool_use"}
                        
                        # Try to get ID if available
                        if hasattr(event.content_block, 'id') and event.content_block.id:
                            current_tool_use["id"] = event.content_block.id
                            print_colored(f"[Tool ID: {event.content_block.id}]", Colors.YELLOW)
                
            elif event.type == "content_block_delta":
                if hasattr(event, 'delta') and hasattr(event.delta, 'type'):
                    delta_type = event.delta.type
                    
                    if delta_type == "text":
                        # For text blocks
                        if hasattr(event.delta, 'text'):
                            current_text += event.delta.text
                            sys.stdout.write(event.delta.text)
                            sys.stdout.flush()
                    
                    elif delta_type == "tool_use":
                        # For tool use blocks
                        if current_tool_use is None:
                            current_tool_use = {"type": "tool_use"}
                        
                        # Handle name
                        if hasattr(event.delta, 'name') and event.delta.name is not None:
                            current_tool_use["name"] = event.delta.name
                            print_colored(f"\n[Using tool: {event.delta.name}]", Colors.YELLOW)
                        
                        # Handle input
                        if hasattr(event.delta, 'input') and event.delta.input is not None:
                            # The input can either be a dict or a string
                            if isinstance(event.delta.input, dict):
                                if "input" not in current_tool_use:
                                    current_tool_use["input"] = {}
                                current_tool_use["input"].update(event.delta.input)
                            else:
                                current_tool_use["input"] = event.delta.input
                            print_colored(f"[Parameters: {event.delta.input}]", Colors.YELLOW)
                        
                        # Handle ID
                        if hasattr(event.delta, 'id') and event.delta.id is not None:
                            current_tool_use["id"] = event.delta.id
                            print_colored(f"[Tool ID: {event.delta.id}]", Colors.YELLOW)
            
            elif event.type == "content_block_stop":
                # End of a content block - note that this event has different structure
                if verbose:
                    print(f"  Stopping block type: {current_block_type}")
                
                # Use the tracked current_block_type instead of trying to access event.content_block.type
                if current_block_type == "text" and current_text:
                    accumulated_content.append({
                        "type": "text",
                        "text": current_text
                    })
                    current_text = ""
                    
                elif current_block_type == "tool_use" and current_tool_use:
                    # Make sure all required fields are present
                    if "name" in current_tool_use and "input" in current_tool_use:
                        accumulated_content.append(current_tool_use)
                        tool_uses.append(current_tool_use.copy())
                        print_colored(f"\n[Tool use complete]", Colors.GREEN)
                    elif verbose:
                        print_colored(f"Warning: Incomplete tool_use: {current_tool_use}", Colors.RED)
                    
                    current_tool_use = None
                
                # Reset current block type
                current_block_type = None
            
            elif event.type == "message_delta":
                # Message delta (usually at the end)
                if verbose and hasattr(event, 'delta'):
                    if hasattr(event.delta, 'stop_reason'):
                        print(f"  Message delta stop reason: {event.delta.stop_reason}")
            
            elif event.type == "message_stop":
                # Message complete
                if verbose:
                    print("  Message complete")
        
        print("\n")  # Add newline after streaming completes
        
        # End timer
        end_time = time.time()
        elapsed = end_time - start_time
        
        print_section("RESPONSE SUMMARY")
        print(f"Total events: {event_count}")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Tool uses: {len(tool_uses)}")
        
        if tool_uses:
            print_section("EXECUTING TOOLS")
            for i, tool_use in enumerate(tool_uses):
                print_colored(f"\nTool {i+1}:", Colors.BOLD)
                print_json(tool_use)
                
                # Execute the tool
                if "name" in tool_use and "input" in tool_use:
                    result = execute_tool(tool_use["name"], tool_use["input"])
                    print_colored(f"Result: {result}", Colors.GREEN)
                    
                    # Create tool result message
                    tool_result_content = {
                        "type": "tool_result",
                        "tool_call_id": tool_use.get("id"),
                        "content": result
                    }
                    
                    # Here you would add the tool result to the conversation
                    print_colored(f"Tool result message:", Colors.CYAN)
                    print_json(tool_result_content)
        
        print_section("ACCUMULATED CONTENT")
        print_json(accumulated_content)
        
        return True, accumulated_content
        
    except Exception as e:
        print_section("ERROR")
        print_colored(f"Error: {str(e)}", Colors.RED)
        import traceback
        traceback.print_exc()
        return False, str(e)

def test_non_streaming_tool_call(api_key):
    """Test non-streaming tool calls with Anthropic API for comparison."""
    client = anthropic.Anthropic(api_key=api_key)
    tools = define_tools()
    
    print_section("NON-STREAMING TEST")
    print(f"Model: claude-3-7-sonnet-20250219")
    
    # System prompt
    system_prompt = """
You are a helpful assistant that can use tools.
When asked a question that requires calculation or weather information, use the appropriate tool.
"""

    # User prompt
    user_prompt = "What is 42 * 56? After you answer that, tell me the weather in Tokyo."
    
    print(f"User prompt: {user_prompt}")
    
    # Start timer
    start_time = time.time()
    
    try:
        print_colored("Claude is thinking...", Colors.CYAN)
        
        # Create non-streaming request
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens=1000,
            temperature=0.7,
            tools=tools
        )
        
        # End timer
        end_time = time.time()
        elapsed = end_time - start_time
        
        print_section("NON-STREAMING RESPONSE")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        
        tool_calls = []
        
        # Process response
        for content in response.content:
            if content.type == "text":
                print(content.text)
            elif content.type == "tool_use":
                print_colored(f"\n[Using tool: {content.name}]", Colors.YELLOW)
                print_colored(f"[Parameters: {content.input}]", Colors.YELLOW)
                
                tool_call = {
                    "type": "tool_use",
                    "name": content.name,
                    "input": content.input,
                    "id": content.id
                }
                tool_calls.append(tool_call)
                
                # Execute the tool
                result = execute_tool(content.name, content.input)
                print_colored(f"Result: {result}", Colors.GREEN)
        
        return True, response.content
        
    except Exception as e:
        print_section("ERROR")
        print_colored(f"Error: {str(e)}", Colors.RED)
        import traceback
        traceback.print_exc()
        return False, str(e)

def main():
    """Main function to run the test."""
    # Get API key from environment or command line
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        print_colored("Error: No API key found. Set ANTHROPIC_API_KEY environment variable.", Colors.RED)
        sys.exit(1)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test Anthropic API tool calls')
    parser.add_argument('--non-streaming', action='store_true', help='Test non-streaming API call')
    parser.add_argument('--verbose', action='store_true', help='Show verbose output for streaming events')
    parser.add_argument('--both', action='store_true', help='Test both streaming and non-streaming')
    args = parser.parse_args()
    
    print_section("ANTHROPIC API TOOL CALLS TEST")
    
    if args.both or (not args.non_streaming):
        # Test streaming
        success, content = test_streaming_tool_call(api_key, args.verbose)
        if not success:
            print_colored("Streaming test failed!", Colors.RED)
    
    if args.both or args.non_streaming:
        # Test non-streaming for comparison
        success, content = test_non_streaming_tool_call(api_key)
        if not success:
            print_colored("Non-streaming test failed!", Colors.RED)
    
    print_section("TEST COMPLETE")

if __name__ == "__main__":
    main()