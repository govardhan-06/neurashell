from pydantic import BaseModel, Field
from typing import List
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

# Output parsers
class Command(BaseModel):
    command: str = Field(description="The actual shell command to execute")
    description: str = Field(description="Brief explanation of what this command does")
    execution_order: int = Field(description="The order in which the command should be executed")

class CommandResponse(BaseModel):
    """Structured response containing commands in the correct execution order."""
    commands: List[Command] = Field(description="List of commands to be executed in the correct order")

class AgentState(MessagesState):
    # Final structured response from the agent
    final_response: CommandResponse

def masterAgent(state:AgentState):
    """Master agent that orchestrates the execution of all other agents and answers the user's query."""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    model=ChatGroq(model="llama-3.3-70b-versatile",api_key=GROQ_API_KEY)
    model_structured_response = model.with_structured_output(CommandResponse)
    response = model_structured_response.invoke(
        [HumanMessage(content=state["messages"][-1].content)]
    )
    return {"final_response": response}

    