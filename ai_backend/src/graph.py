from langgraph.graph import StateGraph, END
from src.agents.master import AgentState, masterAgent

neura=StateGraph(AgentState)

neura.add_node("masterAgent",masterAgent)
neura.set_entry_point("masterAgent")
neura.add_edge("masterAgent", END)

neuraGraph=neura.compile()