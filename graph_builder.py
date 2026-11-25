from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from .agent_state import AgentState
from .node import editor_node, researcher_node, supervisor_node, writer_node


builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor_node)

builder.add_node("researcher", researcher_node)

builder.add_node("writer", writer_node)

builder.add_node("editor", editor_node)



builder.add_edge(START, "supervisor")

builder.add_edge("researcher", "supervisor")

builder.add_edge("writer", "supervisor")

builder.add_edge("editor", "supervisor")



builder.add_conditional_edges(

    "supervisor",

    lambda state: state.get("next", "FINISH"),

    {

        "researcher": "researcher",

        "writer": "writer",

        "editor": "editor",

        "FINISH": END,

    },

)



graph = builder.compile(checkpointer=MemorySaver())