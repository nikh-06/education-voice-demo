# main.py

import sys
import time

from sahayak.graph import sahayak_graph
from sahayak.nodes import save_outputs_node  # Import the save function specifically
from sahayak.utils import setup_memory_database, setup_rag_pipeline, listen_for_voice_command


def main():
    """Main function to run the Sahayak agent."""
    # Initial setup
    setup_memory_database()
    pdf_path = "source_material.pdf"
    try:
        rag_retriever = setup_rag_pipeline(pdf_path)
    except Exception as e:
        print(f"❌ Error setting up RAG pipeline: {e}")
        sys.exit(1)

    print("\n\n--- Welcome to Project Sahayak ---")

    # Get user input
    user_input = None
    while not user_input:
        choice = input("Press 'V' for voice input or 'T' for text input, then Enter: ").strip().upper()
        if choice == 'T':
            user_input = input("Hello! What lesson can I prepare for you today?\n> ")
        elif choice == 'V':
            lang_choice = input("Select language ('en' for English, 'hi' for Hindi): ").strip().lower()
            lang_code = "hi-IN" if lang_choice == "hi" else "en-IN"
            user_input = listen_for_voice_command(language=lang_code)
        else:
            print("Invalid choice.")

    if not user_input or not user_input.strip():
        print("No input received. Exiting.")
        sys.exit(1)

    # Prepare initial state and run the graph
    initial_state = {
        "user_request": user_input,
        "retriever": rag_retriever,
    }

    print("\n🚀 Starting Graph Execution... 🚀")
    start_time = time.time()

    final_state = {}
    for event in sahayak_graph.stream(initial_state):
        for key, value in event.items():
            print(f"--- Finished Node: {key} ---")
            final_state.update(value)

    end_time = time.time()
    final_state['execution_time'] = end_time - start_time

    # Handle final output
    if final_state.get("error"):
        print(f"\n--- ❗ERROR OCCURRED ---\n{final_state['error']}")
    else:
        # Call the standalone save function with the final state
        save_outputs_node(final_state)
        print("\n\n✅✅✅ --- Content Generation Complete --- ✅✅✅")


if __name__ == "__main__":
    main()
