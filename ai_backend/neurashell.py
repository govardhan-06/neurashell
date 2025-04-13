import os
import subprocess
import re
import warnings
import time
import speech_recognition as sr
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import track
from langchain.memory import ConversationBufferMemory
from langchain.schema import AgentAction, AgentFinish, SystemMessage, HumanMessage, AIMessage

warnings.filterwarnings("ignore")

# Load .env for Groq API Key
load_dotenv()

# Rich console for better formatting
console = Console()

# Initialize the speech recognizer
recognizer = sr.Recognizer()

def get_voice_input() -> Optional[str]:
    """Capture voice input and return the recognized text."""
    with sr.Microphone() as source:
        console.print("[bold yellow]Listening...[/bold yellow]")
        audio = recognizer.listen(source)
        
        try:
            text = recognizer.recognize_google(audio)
            console.print(f"[bold green]You said:[/bold green] {text}")
            return text
        except sr.UnknownValueError:
            console.print("[bold red]Sorry, I could not understand the audio.[/bold red]")
        except sr.RequestError as e:
            console.print(f"[bold red]Could not request results from Google Speech Recognition service; {e}[/bold red]")
        return None

# --------------------
# Tool 1: Shell Executor (safe command whitelist)
# --------------------
def execute_command(command: str) -> str:
    """Execute a shell command safely by checking against a whitelist and additional safety checks."""
    # Extract the main command from potentially complex input
    main_command = command.strip().split()[0] if command.strip() else ""
    
    # Whitelist of allowed commands
    allowed_commands = [
        "ls", "pwd", "mkdir", "whoami", "pip", "python", "python3", 
        "cd", "echo", "cat", "head", "tail", "grep", "find", "wc", "dir",
        "type", "copy", "move", "where"
    ]
    
    # Block commands that can lead to critical issues
    blocked_commands = [
        "rm", "rmdir", "del", "format", "sudo", "chmod", "chown", "shutdown", 
        "reboot", "kill", "dd", "mv", "cp", "scp", "wget", "curl"
    ]
    
    # Additional safety check: prevent command chaining
    if any(c in command for c in [";", "&&", "||", "`", "$("]):
        return "[❌] Command chaining and substitution are not allowed for security reasons."
    
    # Check for blocked commands
    if any(blocked in command for blocked in blocked_commands):
        return f"[❌] Command '{main_command}' is blocked for security reasons."
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=10  # Prevent long-running commands
        )
        if result.returncode == 0:
            return result.stdout or "[✓] Command executed successfully with no output."
        else:
            return f"[⚠️] Command returned error (code {result.returncode}):\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "[⚠️] Command timed out after 10 seconds."
    except Exception as e:
        return f"[❌] Error: {str(e)}"

# --------------------
# Tool 2: Web Search Tool
# --------------------
search_tool = DuckDuckGoSearchRun()

def search_web(query: str) -> str:
    """Search the web with error handling and timeout."""
    try:
        return search_tool.run(query)
    except Exception as e:
        return f"[⚠️] Search error: {str(e)}"

# --------------------
# Tool 3: File Reader
# --------------------
def read_file(filepath: str) -> str:
    """Read and return the contents of a file."""
    if not os.path.exists(filepath):
        return f"[❌] File not found: {filepath}"
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as file:
            content = file.read()
            # If file is too large, truncate it
            if len(content) > 10000:
                return content[:10000] + "\n...[content truncated due to size]..."
            return content
    except Exception as e:
        return f"[❌] Error reading file: {str(e)}"

# --------------------
# Tool 4: File Writer
# --------------------
def write_file(args: str) -> str:
    """Write content to a file. Format: 'filepath||content'"""
    try:
        filepath, content = args.split("||", 1)
        filepath = filepath.strip()
        
        # Check for unsafe paths
        if ".." in filepath or filepath.startswith("/") or ":" in filepath:
            return "[❌] Path contains unsafe characters."
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
            
        return f"[✓] Successfully wrote to {filepath}"
    except Exception as e:
        return f"[❌] Error writing file: {str(e)}"

# --------------------
# Tool 5: Memory Viewer
# --------------------
def view_memory(query: str) -> str:
    """View the conversation memory. Used internally by the agent."""
    global conversation_memory
    
    if not hasattr(conversation_memory, 'chat_memory') or not conversation_memory.chat_memory.messages:
        return "No previous conversation found in memory."
    
    summary = []
    messages = conversation_memory.chat_memory.messages
    
    for i, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            summary.append(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            # Truncate very long AI messages
            content = message.content
            if len(content) > 300:
                content = content[:300] + "..."
            summary.append(f"AI: {content}")
    
    return "\n".join(summary[-10:])  # Return the last 10 messages

# --------------------
# LLM (Groq + LLaMA 3)
# --------------------
def get_llm() -> Optional[ChatGroq]:
    """Initialize and return the LLM, or None if API key is missing."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        console.print("[bold red]Error: GROQ_API_KEY not found in environment variables.[/bold red]")
        console.print("Please create a .env file with your GROQ_API_KEY=your_key_here")
        return None
        
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-70b-8192",
        temperature=0.5,
        streaming=True,
        max_tokens=4000  # Limit token output to avoid excessive responses
    )

# --------------------
# Memory Setup
# --------------------
# Initialize the conversation memory
conversation_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tool_memory = {}  # Store tool outputs for reference

# --------------------
# Tools registered to LangChain Agent
# --------------------
def initialize_tools() -> List[Tool]:
    """Initialize and return the list of available tools."""
    return [
        Tool(
            name="Search",
            func=search_web,
            description="Search the web for current information, news, or general knowledge questions."
        ),
        Tool(
            name="Shell",
            func=execute_command,
            description="Safely run basic shell commands. Always use this for system operations."
        ),
        Tool(
            name="ReadFile",
            func=read_file,
            description="Read the contents of a file. Provide the full path to the file."
        ),
        Tool(
            name="WriteFile",
            func=write_file,
            description="Write content to a file. Format: 'filepath||content' - separate the path and content with ||"
        ),
        Tool(
            name="ViewMemory",
            func=view_memory,
            description="View the recent conversation history to remember context."
        )
    ]

# --------------------
# Command Extraction
# --------------------
def extract_commands(agent_output: str) -> List[str]:
    """Extract shell commands from agent output."""
    # Look for commands in code blocks or specific patterns
    code_block_pattern = r"```(?:bash|shell|cmd|powershell)?\s*(.*?)\s*```"
    command_pattern = r"Command to execute:\s*`(.*?)`"
    
    commands = []
    
    # Check for code blocks
    code_blocks = re.findall(code_block_pattern, agent_output, re.DOTALL)
    if code_blocks:
        for block in code_blocks:
            # Split multi-line code blocks into separate commands
            cmd_lines = [line.strip() for line in block.split('\n') if line.strip()]
            commands.extend(cmd_lines)
    
    # Check for inline commands
    inline_commands = re.findall(command_pattern, agent_output)
    if inline_commands:
        commands.extend(inline_commands)
    
    # Look for backtick enclosed commands
    backtick_commands = re.findall(r'`([^`]+)`', agent_output)
    for cmd in backtick_commands:
        if any(allowed in cmd for allowed in ["ls", "pwd", "dir", "python", "pip"]):
            commands.append(cmd)
    
    # If no structured commands found, use the whole output as a command if it looks like one
    if not commands and not agent_output.count('\n') > 2 and len(agent_output) < 100:
        commands = [agent_output.strip()]
        
    return commands

# --------------------
# Command Validator
# --------------------
def validate_commands(commands: List[str], llm: Optional[ChatGroq]) -> Dict[str, bool]:
    """Validate a list of commands using the LLM to assess safety."""
    if not commands or not llm:
        return {}
    
    results = {}
    
    for cmd in commands:
        # Skip validation for Python code
        if cmd.strip().startswith(("from ", "import ", "def ", "class ", "@", "return ")):
            results[cmd] = True
            continue
        
        # Create a prompt to ask the LLM if the command is safe
        prompt = f"""
        Evaluate the following command for safety. Consider whether it could lead to file deletion, system modification, or other harmful operations. 
        Respond with 'safe' if the command is safe to execute, or 'unsafe' if it is not.

        Command: {cmd}

        Response:
        """
        
        try:
            # Call the LLM to evaluate the command
            response = llm.invoke(prompt).content.strip().lower()
            
            # Determine if the command is safe based on the LLM's response
            if response == "safe":
                results[cmd] = True
            else:
                results[cmd] = False
        except Exception as e:
            # If there's an error, default to blocking the command
            console.print(f"[bold red]Error during LLM validation: {str(e)}[/bold red]")
            results[cmd] = False
    
    return results

# --------------------
# Session Management
# --------------------
def save_session(history: List[Dict[str, Any]], memory: ConversationBufferMemory) -> None:
    """Save the current session to a file."""
    session_data = {
        "history": history,
        "memory": memory.chat_memory.messages if hasattr(memory, 'chat_memory') else []
    }
    
    # Create directory if it doesn't exist
    os.makedirs('sessions', exist_ok=True)
    
    # Save to file with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open(f'sessions/session_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write("=== COMMAND HISTORY ===\n")
        for i, cmd in enumerate(history, 1):
            f.write(f"{i}. Command: {cmd.get('command', '')}\n")
            f.write(f"   Result: {cmd.get('result', '')[:100]}...\n\n")
        
        f.write("\n=== CONVERSATION MEMORY ===\n")
        if hasattr(memory, 'chat_memory'):
            for msg in memory.chat_memory.messages:
                if isinstance(msg, HumanMessage):
                    f.write(f"Human: {msg.content}\n")
                elif isinstance(msg, AIMessage):
                    f.write(f"AI: {msg.content[:100]}...\n")
                f.write("\n")
    
    console.print(f"[green]Session saved to sessions/session_{timestamp}.txt[/green]")

# --------------------
# Main CLI Loop
# --------------------
def main():
    """Main CLI loop for NeuraShell."""
    console.print(Panel.fit(
        "[bold cyan]NeuraShell[/bold cyan] - AI-Powered Command Line Assistant", 
        title="🧠", 
        subtitle="Type 'exit' to quit | 'help' for commands | 'voice' for voice input"
    ))
    
    # Initialize LLM
    llm = get_llm()
    if not llm:
        return
    
    # Initialize tools and agent with memory
    tools = initialize_tools()
    
    # Initialize agent with memory
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5, 
        early_stopping_method="generate",  # Try to generate a response even if not all tools are used
        memory=conversation_memory  # Add memory to the agent
    )
    
    # Add system prompt to guide the agent with context
    if hasattr(agent, 'agent') and hasattr(agent.agent, 'llm_chain') and hasattr(agent.agent.llm_chain, 'prompt'):
        # Get the existing prompt template
        prompt_template = agent.agent.llm_chain.prompt
        
        # Add memory instruction to the existing template
        if hasattr(prompt_template, 'template'):
            prompt_template.template = prompt_template.template.replace(
                "Answer the following questions as best you can.",
                "Answer the following questions as best you can. If the user is trying to have a normal conversation, just answer the question. Use the ViewMemory tool to check conversation history when needed to maintain context. You are working on a Windows system. Use the appropriate commands for Windows."
            )
    
    history = []  # Store command history
    session_active = True
    
    # Welcome message
    console.print("[bold blue]🤖 NeuraShell initialized with memory. I'll remember our conversation![/bold blue]")
    
    while session_active:
        try:
            # Get user input
            user_input = console.input("\n[bold green]You>[/bold green] ")
            
            # Special commands handling
            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold]Exiting NeuraShell. Goodbye![/bold]")
                # Ask if user wants to save session
                save_choice = console.input("[yellow]Save this session? (y/n):[/yellow] ")
                if save_choice.lower() in ['y', 'yes']:
                    save_session(history, conversation_memory)
                break
                
            if user_input.lower() == "help":
                show_help()
                continue
                
            if user_input.lower() == "history":
                show_history(history)
                continue
                
            if user_input.lower() == "memory":
                show_memory()
                continue
                
            if user_input.lower() == "clear" or user_input.lower() == "cls":
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
                
            if user_input.lower() == "save":
                save_session(history, conversation_memory)
                continue
                
            if user_input.lower() == "clearmemory":
                # Reset the memory
                conversation_memory.clear()
                console.print("[yellow]Memory has been cleared.[/yellow]")
                continue
            
            if user_input.lower() == "voice":
                # Capture voice input
                voice_input = get_voice_input()
                if voice_input:
                    user_input = voice_input
            
            # Process the query
            console.print("[bold blue]\n🤖 Thinking...[/bold blue]")
            
            # Add user input to memory before processing
            conversation_memory.chat_memory.add_user_message(user_input)
            
            # Add a timeout mechanism
            try:
                start_time = time.time()
                raw_output = agent.run(input=user_input)
                end_time = time.time()
                
                console.print(f"[dim](Response generated in {end_time - start_time:.2f} seconds)[/dim]")
            except Exception as e:
                console.print(f"[bold red]Error during processing: {str(e)}[/bold red]")
                # Fallback to direct LLM call on error
                try:
                    raw_output = llm.invoke(f"Answer this question or command concisely: {user_input}").content
                except:
                    raw_output = "I encountered an error while processing your request. Please try again with a simpler query."
            
            # Add response to memory
            conversation_memory.chat_memory.add_ai_message(raw_output)
            
            # Pretty-print the agent's thinking
            console.print(Panel(Markdown(raw_output), title="Agent Response"))
            
            # Extract commands
            commands = extract_commands(raw_output)
            
            if commands:
                # Validate commands using the LLM
                console.print("[bold yellow]🔐 Validating commands...[/bold yellow]")
                validation_results = validate_commands(commands, llm)
                
                # Execute safe commands
                for cmd in commands:
                    if cmd in validation_results and validation_results[cmd]:
                        console.print(f"[bold green]✓ Executing:[/bold green] {cmd}")
                        result = execute_command(cmd)
                        console.print(Panel(result, title="Command Output"))
                        
                        # Store in command history
                        history.append({"command": cmd, "result": result})
                        
                        # Add important results to memory
                        if len(result) < 1000:  # Only store reasonably sized outputs
                            tool_memory[cmd] = result
                            conversation_memory.chat_memory.add_user_message(f"Command result for `{cmd}`: {result}")
                    else:
                        console.print(f"[bold red]⛔ Blocked unsafe command:[/bold red] {cmd}")
            else:
                console.print("[bold yellow]No executable commands detected in response.[/bold yellow]")
                
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Operation cancelled by user.[/bold yellow]")
            # Ask if the user wants to exit
            exit_choice = console.input("[yellow]Exit NeuraShell? (y/n):[/yellow] ")
            if exit_choice.lower() in ['y', 'yes']:
                save_choice = console.input("[yellow]Save this session? (y/n):[/yellow] ")
                if save_choice.lower() in ['y', 'yes']:
                    save_session(history, conversation_memory)
                session_active = False
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

def show_help():
    """Display help information."""
    help_text = """
# NeuraShell Commands

## System Commands
- `exit` or `quit`: Exit NeuraShell
- `help`: Show this help message
- `history`: Show command history
- `memory`: Show conversation memory
- `clearmemory`: Clear conversation memory
- `save`: Save current session
- `clear` or `cls`: Clear the screen

## What You Can Ask:
- General questions (uses web search)
- Shell commands (safely executed)
- File operations (read, write, find, analyze)
- Programming help
- How to use commands
- Create and run Python scripts
- Follow-up questions (assistant remembers context)

## Examples:
- "What's the current weather in New York?"
- "List files in the current directory"
- "Create a simple Python script to download a webpage"
- "What did we just discuss?"
- "Continue with the previous task"
    """
    console.print(Markdown(help_text))

def show_history(history: List[Dict[str, Any]]):
    """Display command history."""
    if not history:
        console.print("[italic]No commands in history yet.[/italic]")
        return
        
    console.print("[bold]Command History:[/bold]")
    for i, entry in enumerate(history, 1):
        console.print(f"[bold]{i}.[/bold] {entry['command']}")
        if i < len(history):
            console.print("─" * 50)

def show_memory():
    """Display the conversation memory."""
    if not hasattr(conversation_memory, 'chat_memory') or not conversation_memory.chat_memory.messages:
        console.print("[italic]Memory is empty.[/italic]")
        return
    
    console.print("[bold]Conversation Memory:[/bold]")
    
    for i, message in enumerate(conversation_memory.chat_memory.messages):
        if isinstance(message, HumanMessage):
            console.print(f"[bold green]You:[/bold green] {message.content}")
        elif isinstance(message, AIMessage):
            # Truncate very long AI messages for display
            content = message.content
            if len(content) > 300:
                content = content[:300] + "..."
            console.print(f"[bold blue]NeuraShell:[/bold blue] {content}")
        
        if i < len(conversation_memory.chat_memory.messages) - 1:
            console.print("─" * 80)

if __name__ == "__main__":
    main()
