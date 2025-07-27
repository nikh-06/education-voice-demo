# sahayak/state.py

from typing import TypedDict, List
from langchain.schema import Document
from pydantic import BaseModel, Field

# --- Pydantic Models for Structured Output ---
class Intent(BaseModel):
    # ... (rest of the class is unchanged)
    topic: str = Field(description="The main subject or topic for the lesson plan.")
    grade_level: str = Field(description="The target grade level for the lesson.")

class EvaluationReport(BaseModel):
    # ... (rest of the class is unchanged)
    clarity_score: int = Field(description="Clarity score (1-5), is it easy to understand for the grade level?")
    clarity_feedback: str = Field(description="Qualitative feedback on clarity.")
    engagement_score: int = Field(description="Engagement score (1-5), is the analogy and content interesting?")
    engagement_feedback: str = Field(description="Qualitative feedback on engagement.")
    educational_value_score: int = Field(description="Educational value score (1-5), does it meet learning objectives?")
    educational_value_feedback: str = Field(description="Qualitative feedback on educational value.")


# --- Graph State Definition ---
class GraphState(TypedDict):
    """The state of the graph."""
    # --- MODIFIED ---
    user_uuid: str  # To identify the user for Firebase publishing

    # Input fields
    user_request: str
    topic: str
    grade_level: str
    retriever: object
    # ... (rest of the state is unchanged)
