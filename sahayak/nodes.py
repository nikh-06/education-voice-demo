# sahayak/nodes.py

import datetime
from firebase_admin import db
from langchain_google_vertexai import ChatVertexAI
from langchain_community.tools.tavily_search import TavilySearchResults

from .state import GraphState, Intent, EvaluationReport
from .utils import generate_image_with_fallback

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

        topic_check_prompt = f"Analyze if the following text contains substantial information about '{topic}'. The text is a curated selection of source material. Return ONLY 'yes' or 'no'."
        has_topic = llm.invoke(topic_check_prompt).content.lower().strip()

        if has_topic != "yes":
            return {"error": f"I apologize, but I don't have enough reliable information about '{topic}' in my source material to create a lesson."}

        return {"grounded_content": grounded_content}
    except Exception as e:
        return {"error": f"Failed during the reranking process: {e}"}

def creative_assistant_node(state: GraphState):
    """Generates a culturally relevant analogy for the lesson."""
    print("---NODE: CREATIVE ASSISTANT---")
    topic, grade = state["topic"], state["grade_level"]
    query = llm.invoke(f"Generate a search query for culturally relevant analogies to teach '{topic}' to {grade} students in India.").content.strip()
    search_results = tavily_tool.invoke(query)
    results_content = "\n\n".join([res.get('content', '') for res in search_results if isinstance(res, dict) and res.get('content')])
    synthesis = llm.invoke(f"Create a simple, two-sentence analogy to explain '{topic}' to {grade} students in India, using these search results:\n\n{results_content}").content.strip()
    return {"supplemental_content": synthesis}

def enhanced_prompt_composer_node(state: GraphState):
    """Creates the final prompts for the lesson and quiz generators."""
    print("---NODE: ENHANCED PROMPT COMPOSER---")
    grounded_content, supplemental_content, topic, grade_level = state["grounded_content"], state["supplemental_content"], state["topic"], state["grade_level"]

    lesson_prompt = f"""Create a lesson plan about '{topic}'.
Primary Source Material (Facts):
---
{grounded_content}
---
Creative Element (Analogy):
---
{supplemental_content}
---
Task: Create a detailed lesson plan for {grade_level} with the following structure:
1. Topic
2. Target Grade ({grade_level})
3. Objectives
4. Materials
5. Introduction (incorporate the creative analogy)
6. Activities
7. Assessment
Important Guidelines:
- Use only facts explicitly stated in the source material.
- Maintain professional, academic language.
- Format in clean markdown.
- Start directly with the topic, no introductory text."""

    quiz_prompt = f"""Create a worksheet about '{topic}' for {grade_level}.
Primary Source Material:
---
{grounded_content}
---
Requirements:
- Create 3-4 questions appropriate for a {grade_level} understanding.
- Include an answer key.
- Base all questions strictly on the source material.
- Format in clean markdown.
- Start directly with the worksheet title."""

    return {"lesson_prompt": lesson_prompt, "quiz_prompt": quiz_prompt}

def lesson_generator_node(state: GraphState):
    """Generates the lesson plan from the prompt."""
    print("---NODE: LESSON GENERATOR---")
    lesson_plan = llm.invoke(state['lesson_prompt']).content
    return {"lesson_plan": lesson_plan}

def quiz_generator_node(state: GraphState):
    """Generates the quiz from the prompt."""
    print("---NODE: QUIZ GENERATOR---")
    quiz = llm.invoke(state['quiz_prompt']).content
    return {"quiz": quiz}

def image_generator_node(state: GraphState):
    """Generates a visual aid for the lesson."""
    print("---NODE: IMAGE GENERATOR---")
    try:
        return {"image_url": generate_image_with_fallback(f"Educational diagram about {state['topic']}.")}
    except Exception as e:
        print(f"⚠️ Image generation failed: {e}")
        return {"image_url": "No image generated."}

def hallucination_guard_node(state: GraphState):
    """Fact-checks the lesson plan against the source material."""
    print("---NODE: HALLUCINATION GUARD---")
    verification_prompt = f"Fact-check the 'Lesson Plan' against the 'Source Text'. If all claims are supported, respond with 'All claims verified.'. Otherwise, list unsupported claims.\n\nSource Text:\n{state['grounded_content']}\n\nLesson Plan:\n{state['lesson_plan']}"
    report = llm.invoke(verification_prompt).content
    return {"verification_report": report}

def final_compiler_node(state: GraphState):
    """Compiles the final lesson plan with the image URL."""
    print("---NODE: FINAL COMPILER---")
    if state.get("error") or not all(state.get(k) for k in ["lesson_plan", "quiz", "image_url", "verification_report"]):
        return {} # Don't compile if there's an error or missing data

    compiled_lesson = state['lesson_plan']
    if state['image_url'] and state['image_url'] != "No image generated.":
        visual_aid_section = f"\n\n### Visual Aid Suggestion\n\n![{state['topic']}]({state['image_url']})\n"
        compiled_lesson += visual_aid_section
    return {"compilation_complete": True, "compiled_lesson": compiled_lesson}

def evaluation_agent_node(state: GraphState):
    """Assesses the generated content against a rubric."""
    print("---NODE: EVALUATION AGENT---")
    if not state.get("compilation_complete"):
        return {}

    evaluator_llm = llm.with_structured_output(EvaluationReport)
    prompt = f"You are an expert curriculum reviewer. Evaluate the following educational content for {state['grade_level']} students based on: 1. Clarity, 2. Engagement, 3. Educational Value. Provide a score (1-5) and concise feedback for each.\n\n**Lesson Plan:**\n{state.get('compiled_lesson')}\n\n**Quiz:**\n{state.get('quiz')}"

    try:
        report = evaluator_llm.invoke(prompt)
        print(f"✅ Evaluation Complete: Clarity={report.clarity_score}/5")
        return {"evaluation_report": report}
    except Exception as e:
        return {"error": f"Failed to generate evaluation report: {e}"}

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
