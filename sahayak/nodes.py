# sahayak/nodes.py

import datetime
import os
import sqlite3

from langchain_google_vertexai import ChatVertexAI
from langchain_community.tools.tavily_search import TavilySearchResults

from .state import GraphState, Intent, EvaluationReport
from .utils import generate_image_with_fallback, DB_PATH

# --- Tool and Model Setup ---
llm = ChatVertexAI(model_name="gemini-1.5-pro")
tavily_tool = TavilySearchResults(max_results=3)

# --- Agent Node Definitions ---

def intent_parser_node(state: GraphState):
    """Parses the user's request into a structured format."""
    print("---NODE: INTENT PARSER---")
    structured_llm = llm.with_structured_output(Intent)
    prompt = f"Parse the following user request to extract the lesson topic and grade level.\n\nRequest: \"{state['user_request']}\""
    try:
        parsed_intent = structured_llm.invoke(prompt)
        print(f"✅ Intent Parsed: Topic='{parsed_intent.topic}', Grade='{parsed_intent.grade_level}'")
        return {"topic": parsed_intent.topic, "grade_level": parsed_intent.grade_level}
    except Exception as e:
        return {"error": f"Failed to parse user request: {e}"}

def rag_agent_node(state: GraphState):
    """Retrieves a broad set of documents from the vector store."""
    print("---NODE: RAG AGENT (RETRIEVAL)---")
    try:
        retrieved_docs = state["retriever"].invoke(state["topic"])
        print(f"✅ Retrieved {len(retrieved_docs)} documents for reranking.")
        return {"retrieved_docs": retrieved_docs}
    except Exception as e:
        return {"error": f"Failed to retrieve documents: {e}"}

def llm_reranker_node(state: GraphState):
    """Uses an LLM to rerank and select the most relevant documents."""
    print("---NODE: LLM RERANKER---")
    topic, grade_level, documents = state["topic"], state["grade_level"], state["retrieved_docs"]

    formatted_docs = ""
    for i, doc in enumerate(documents):
        formatted_docs += f"--- Document {i+1} (Source: Page {doc.metadata.get('page', 'N/A')}) ---\n{doc.page_content}\n\n"

    reranker_prompt = f"""You are an expert curriculum assistant. Your task is to select the most relevant information for a lesson plan for '{grade_level}' students on the topic: '{topic}'.
    From the {len(documents)} documents below, select the TOP 3-4 that are most relevant and essential. Your goal is to eliminate redundant or tangential information.
    Return ONLY the full text of the chosen documents, separated by the exact delimiter '---'. Do not add any commentary.
    ---
    {formatted_docs}"""

    try:
        response = llm.invoke(reranker_prompt)
        reranked_docs_text = [doc.strip() for doc in response.content.strip().split('---') if doc.strip()]

        if not reranked_docs_text:
            return {"error": "Reranker returned no documents."}

        print(f"✅ Reranker selected {len(reranked_docs_text)} documents.")
        grounded_content = "\n\n---\n\n".join(reranked_docs_text)
        return {"grounded_content": grounded_content}
    except Exception as e:
        return {"error": f"Failed during the reranking process: {e}"}

def creative_assistant_node(state: GraphState):
    print("---NODE: CREATIVE ASSISTANT---")
    topic, grade = state["topic"], state["grade_level"]
    query = llm.invoke(f"Generate a search query for culturally relevant analogies to teach '{topic}' to {grade} students in India.").content.strip()
    search_results = tavily_tool.invoke(query)
    results_content = "\n\n".join([res.get('content', '') for res in search_results if isinstance(res, dict) and res.get('content')])
    synthesis = llm.invoke(f"Create a simple, two-sentence analogy to explain '{topic}' to {grade} students in India, using these search results:\n\n{results_content}").content.strip()
    return {"supplemental_content": synthesis}

def enhanced_prompt_composer_node(state: GraphState):
    print("---NODE: ENHANCED PROMPT COMPOSER---")
    # This node remains the same as in the original file
    # ... (code omitted for brevity, it's unchanged) ...
    return {"lesson_prompt": "...", "quiz_prompt": "..."} # Placeholder

def lesson_generator_node(state: GraphState):
    print("---NODE: LESSON GENERATOR---")
    # This node remains the same as in the original file
    # ... (code omitted for brevity, it's unchanged) ...
    return {"lesson_plan": "..."} # Placeholder

def quiz_generator_node(state: GraphState):
    print("---NODE: QUIZ GENERATOR---")
    # This node remains the same as in the original file
    # ... (code omitted for brevity, it's unchanged) ...
    return {"quiz": "..."} # Placeholder

def image_generator_node(state: GraphState):
    print("---NODE: IMAGE GENERATOR---")
    try:
        return {"image_url": generate_image_with_fallback(f"Educational diagram about {state['topic']}.")}
    except Exception:
        return {"image_url": "No image generated."}

def hallucination_guard_node(state: GraphState):
    print("---NODE: HALLUCINATION GUARD---")
    # This node remains the same as in the original file
    # ... (code omitted for brevity, it's unchanged) ...
    return {"verification_report": "..."} # Placeholder

def final_compiler_node(state: GraphState):
    print("---NODE: FINAL COMPILER---")
    # This node remains the same as in the original file
    # ... (code omitted for brevity, it's unchanged) ...
    return {"compilation_complete": True, "compiled_lesson": "..."} # Placeholder

def evaluation_agent_node(state: GraphState):
    print("---NODE: EVALUATION AGENT---")
    # This node remains the same as in the original file
    # ... (code omitted for brevity, it's unchanged) ...
    return {"evaluation_report": EvaluationReport(...)} # Placeholder

def firebase_publish_node(state: GraphState):
    """Publishes the final lesson plan to Firebase Realtime Database."""
    print("---NODE: FIREBASE PUBLISHER---")
    if state.get("error"):
        print(f"❌ Cannot publish to Firebase due to an error: {state['error']}")
        return {}

    user_uuid = state.get("user_uuid")
    if not user_uuid:
        print("❌ Cannot publish to Firebase: user_uuid is missing from state.")
        return {"error": "User UUID not found for publishing."}

    # Prepare the data payload
    final_lesson_plan = {
        "topic": state.get("topic"),
        "grade_level": state.get("grade_level"),
        "lesson_plan": state.get("compiled_lesson"),
        "quiz": state.get("quiz"),
        "evaluation": state.get("evaluation_report", {}).dict() if state.get("evaluation_report") else {},
        "status": "completed",
        "timestamp": datetime.datetime.now().isoformat()
    }

    try:
        # The path in Firebase where the data will be stored
        ref = db.reference(f"/lesson_plans/{user_uuid}")
        ref.set(final_lesson_plan)
        print(f"✅ Successfully published lesson for user {user_uuid} to Firebase.")
    except Exception as e:
        print(f"❌ Failed to publish to Firebase: {e}")
        # Optionally update Firebase with an error status
        error_ref = db.reference(f"/lesson_plans/{user_uuid}")
        error_ref.set({"status": "failed", "error": str(e)})
        return {"error": f"Firebase publishing failed: {e}"}

    return {}
