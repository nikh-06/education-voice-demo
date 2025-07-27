# sahayak/utils.py

import datetime
import os
import sqlite3
import speech_recognition as sr
import requests
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIImageGeneratorChat

# --- Environment Setup ---
load_dotenv()
DB_PATH = "sahayak_memory.db"

# Initialize image generation tool once
try:
    image_generation_tool = VertexAIImageGeneratorChat(model="imagegeneration@006")
    print("✅ Vertex AI Image Generation initialized")
except Exception as e:
    print(f"⚠️ Vertex AI Image Generation not available: {e}. Using fallback.")
    image_generation_tool = None

# --- Firebase Initialization ---
def initialize_firebase():
    """Initializes the Firebase Admin SDK."""
    try:
        firebase_admin.get_app()
        print("🔥 Firebase app already initialized.")
    except ValueError:
        cred = credentials.Certificate("firebase-service-account.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://agentic-ai-11-default-rtdb.firebaseio.com/' # IMPORTANT: Replace with your actual URL
        })
        print("🔥 Firebase app initialized successfully.")

# --- Database and RAG Setup ---
def setup_memory_database():
    """Initializes the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            topic TEXT NOT NULL,
            grade_level TEXT NOT NULL,
            clarity_score INTEGER,
            engagement_score INTEGER,
            educational_value_score INTEGER,
            lesson_file TEXT,
            quiz_file TEXT
        );
    """)
    conn.commit()
    conn.close()
    print("✅ Database setup complete.")

def setup_rag_pipeline(source_document_path: str):
    """Sets up the RAG pipeline."""
    loader = PyPDFLoader(source_document_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = splitter.split_documents(docs)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs_split, embedding_model)
    return vectorstore.as_retriever(search_kwargs={'k': 10})

# --- Image Generation ---
def generate_image_with_fallback(prompt: str) -> str:
    """Generates an image using Vertex AI with a Hugging Face fallback."""
    if image_generation_tool:
        try:
            print("Attempting Vertex AI image generation...")
            urls = image_generation_tool.invoke(prompt)
            if urls and urls[0]:
                return urls[0]
        except Exception as e:
            print(f"⚠️ Vertex AI Image Generation failed: {e}")

    print("Attempting Hugging Face image generation...")
    models = ["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"]
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}", "Content-Type": "application/json"}
    formatted_prompt = f"educational textbook illustration, black and white line drawing, {prompt}, simple clean lines"

    for model in models:
        try:
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            response = requests.post(api_url, headers=headers, json={"inputs": formatted_prompt, "options": {"wait_for_model": True}}, timeout=45)
            if response.status_code == 200:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                image_dir = "outputs/images"
                os.makedirs(image_dir, exist_ok=True)
                image_path = f"{image_dir}/{timestamp}_generated.png"
                with open(image_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ Image generated using {model} and saved to {image_path}")
                return image_path
            else:
                print(f"⚠️ Model {model} failed with status {response.status_code}")
        except Exception as e:
            print(f"⚠️ Error with model {model}: {e}")
    return "No image generated."

# --- Voice Input ---
def listen_for_voice_command(language="en-IN"):
    """Listens for a voice command and transcribes it."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Calibrating for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=1)
        print(f"Listening in {language}... Please speak.")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            print("Recognizing...")
            text = r.recognize_google(audio, language=language)
            print(f"✅ Voice Input Recognized: '{text}'")
            return text
        except sr.WaitTimeoutError:
            print("⚠️ Listening timed out.")
        except sr.UnknownValueError:
            print("❌ Could not understand the audio.")
        except sr.RequestError as e:
            print(f"❌ Recognition service error; {e}")
    return None
