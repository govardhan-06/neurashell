from langgraph.graph import StateGraph, END
from src.agents.master import AgentState, masterAgent
from src.agents.fileManager import fileManagerAgent

neura=StateGraph(AgentState)

# neura.add_node("masterAgent",masterAgent)
neura.add_node("fileManagerAgent",fileManagerAgent)
# neura.set_entry_point("masterAgent")
neura.set_entry_point("fileManagerAgent")
# neura.add_edge("masterAgent", END)
neura.add_edge("fileManagerAgent", END)

neuraGraph=neura.compile()