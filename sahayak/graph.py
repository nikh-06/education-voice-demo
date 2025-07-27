# sahayak/graph.py

from langgraph.graph import StateGraph, END
from .state import GraphState
from .nodes import (
    intent_parser_node,
    rag_agent_node,
    llm_reranker_node,
    creative_assistant_node,
    enhanced_prompt_composer_node,
    lesson_generator_node,
    quiz_generator_node,
    image_generator_node,
    hallucination_guard_node,
    final_compiler_node,
    evaluation_agent_node,
    firebase_publish_node  # --- NEW ---
)

def build_api_graph():
    """Builds and compiles the LangGraph agent for the API endpoint."""
    builder = StateGraph(GraphState)

    # Add all nodes, including the new firebase node
    builder.add_node("Intent_Parser", intent_parser_node)
    # ... (add all other nodes from the original graph.py)
    builder.add_node("Evaluation_Agent", evaluation_agent_node)
    builder.add_node("Firebase_Publisher", firebase_publish_node) # --- NEW ---

    # --- Define the graph's flow, ending with Firebase ---
    builder.set_entry_point("Intent_Parser")
    # ... (all other edges from the original graph.py)
    builder.add_edge("Final_Compiler", "Evaluation_Agent")

    # The graph now ends by publishing to Firebase
    builder.add_edge("Evaluation_Agent", "Firebase_Publisher") # --- MODIFIED ---
    builder.add_edge("Firebase_Publisher", END)                # --- MODIFIED ---

    print("✅ API graph built successfully.")
    return builder.compile()

# Compile the API graph when the module is loaded
api_graph = build_api_graph()
