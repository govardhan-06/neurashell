from src.agentTools.systemAPI import write_file, read_file, search_files, move_file, delete_file
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
model=ChatGoogleGenerativeAI(model="gemini-2.0-flash-001",
                             api_key=GEMINI_API_KEY)

class AgentState(MessagesState):
    next: str

system_prompt = SystemMessage(content="""You are a file management assistant that can:
- Write files
- Read files
- Search for files
- Move files
- Delete files
Follow the instructions carefully and execute the requested operations.""")

fileManagerAgent = create_react_agent(
    model,
    tools=[write_file, read_file, search_files, move_file, delete_file],
    prompt=system_prompt
)

def fileManager(state: AgentState):
    """Agent that orchestrates the file management operations."""
    # Convert tuple messages to proper Message objects    
    response = fileManagerAgent.invoke([HumanMessage(content=state["messages"][-1].content)])
    print(response)
    return response