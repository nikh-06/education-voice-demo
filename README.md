# Project Sahayak

Project Sahayak is an agentic workflow platform for generating educational lesson plans and quizzes using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs). It supports both API and interactive CLI/voice workflows, and integrates with Firebase for real-time delivery of results.

## Features
- **Automated Lesson Plan Generation:** Generate structured, curriculum-aligned lesson plans and quizzes from user prompts.
- **Retrieval-Augmented Generation (RAG):** Uses your own PDF source material for grounded, factual content.
- **Human-in-the-Loop Feedback:** Supports correction and approval workflows for high-quality output.
- **Voice and Text Input:** Accepts both typed and spoken user requests.
- **API Access:** FastAPI-based REST API for integration with apps and services.
- **Firebase Integration:** Delivers results to clients in real time.
- **Extensible Graph Workflow:** Modular agent nodes for intent parsing, retrieval, generation, evaluation, and feedback.

## Quickstart

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Prepare Source Material
Place your curriculum PDF as `source_material.pdf` in the project root.

### 3. Run the API Server
```bash
uvicorn api:app --reload
```

### 4. Generate a Lesson Plan (API Example)
```bash
curl -X POST "http://localhost:8000/generate_lesson_plan" \
  -H "Content-Type: application/json" \
  -d '{
    "user_request": "Create a lesson plan on the water cycle for 5th grade.",
    "user_uuid": "user-12345"
  }'
```

### 5. Interactive CLI/Voice Mode
Run the main workflow interactively:
```bash
python main.py
```

## Project Structure
- `api.py` — FastAPI server for lesson plan generation
- `main.py` — CLI/voice workflow for lesson generation
- `sahayak/` — Core agentic workflow, graph, and utility modules
- `outputs/` — Generated lesson plans and quizzes
- `source_material.pdf` — Your curriculum source document
- `firebase-service-account.json` — Firebase credentials (for real-time delivery)

## Human-in-the-Loop Feedback
The workflow supports human review and correction. After lesson/quiz generation, a human can:
- Approve or reject the content
- Provide feedback for corrections
- Loop back to regenerate content with feedback

Feedback is logged and can be used to improve future generations and retrieval quality.

## Extending the Workflow
- Add new agent nodes in `sahayak/graph.py` or related modules
- Integrate additional data sources or LLMs
- Customize evaluation, feedback, or delivery mechanisms

## License
MIT License

## Authors
- [Your Name]

## Environment Variables

Before running the project, ensure you have the following environment variables set in a `.env` file or your environment:

- `IMAGEN_API_KEY` — API key for image generation
- `TAVILY_API_KEY` — API key for Tavily search
- `GOOGLE_CLOUD_PROJECT` — Google Cloud project ID

### Firestore Configuration (for real-time delivery)
- `FIRESTORE_ENABLED` — Set to `true` to enable Firestore integration
- `FIRESTORE_PROJECT_ID` — Your Firestore project ID
- `FIRESTORE_COLLECTION` — The Firestore collection name for storing lessons

**Do not include actual secret values in your code or repository.**

---
For more details, see the code and comments in each module.
