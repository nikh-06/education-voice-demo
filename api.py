# api.py

import sys
import time

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

from sahayak.graph import api_graph
# Import necessary components from your sahayak package
from sahayak.utils import initialize_firebase, setup_rag_pipeline

# Initialize FastAPI app
app = FastAPI(
    title="Project Sahayak API",
    description="API for generating educational lesson plans.",
    version="1.0.0"
)

# --- One-time Setup ---
@app.on_event("startup")
def startup_event():
    """Actions to run on API startup."""
    print("🚀 API starting up...")
    try:
        # Initialize Firebase
        initialize_firebase()
        # Setup the RAG pipeline retriever globally
        app.state.rag_retriever = setup_rag_pipeline("source_material.pdf")
        print("✅ RAG pipeline retriever is ready.")
    except Exception as e:
        print(f"❌ Critical startup error: {e}")
        sys.exit(1)

# --- Request Model ---
class LessonRequest(BaseModel):
    user_request: str
    user_uuid: str

# --- Background Task Function ---
def run_lesson_generation(initial_state: dict):
    """The function that will be run in the background."""
    print(f"🚀 Starting background task for user: {initial_state.get('user_uuid')}")
    start_time = time.time()

    # The final state is not strictly needed here as the result is published to Firebase
    final_state = {}
    for event in api_graph.stream(initial_state):
        for key, value in event.items():
            print(f"--- Finished Node: {key} ---")
            if value is not None and isinstance(value, dict):
                final_state.update(value)

    end_time = time.time()
    print(f"✅ Background task finished in {end_time - start_time:.2f} seconds.")


# --- API Endpoint ---
@app.post("/generate_lesson_plan")
async def generate_lesson_plan(request: LessonRequest, background_tasks: BackgroundTasks):
    """
    Accepts a user request and starts the lesson generation process in the background.
    """
    if not request.user_request or not request.user_uuid:
        raise HTTPException(status_code=400, detail="user_request and user_uuid are required.")

    # Prepare the initial state for the graph
    initial_state = {
        "user_request": request.user_request,
        "user_uuid": request.user_uuid,
        "retriever": app.state.rag_retriever,
    }

    # Add the graph execution as a background task
    background_tasks.add_task(run_lesson_generation, initial_state)

    # Immediately return a response to the client
    return {"message": "Lesson plan generation started. The result will be delivered to your app via Firebase."}

# To run this API, use the command:
# uvicorn api:app --reload
