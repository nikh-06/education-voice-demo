# feedback_analyzer.py

import sqlite3
import pandas as pd
import argparse
import json
import os

DB_PATH = "sahayak_memory.db"
TRAINING_DATA_FILE = "training_dataset.jsonl"

def analyze_performance():
    """
    Connects to the database, analyzes the interaction history,
    and prints a performance summary.
    """
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database file not found at '{DB_PATH}'.")
        print("Please run the main application first to generate some data.")
        return

    print(f"--- Analyzing Performance from '{DB_PATH}' ---")
    conn = sqlite3.connect(DB_PATH)
    
    try:
        df = pd.read_sql_query("SELECT * FROM interactions", conn)
    except pd.io.sql.DatabaseError:
        print("⚠️ No interactions found in the database yet.")
        conn.close()
        return

    if df.empty:
        print("⚠️ The database is empty. No performance data to analyze.")
        conn.close()
        return

    # --- Overall Performance ---
    print("\n## Overall Average Scores ##")
    avg_scores = df[['clarity_score', 'engagement_score', 'educational_value_score']].mean()
    print(avg_scores.round(2))

    # --- Performance by Topic ---
    print("\n## Average Scores by Topic ##")
    topic_scores = df.groupby('topic')[['clarity_score', 'engagement_score', 'educational_value_score']].mean()
    print(topic_scores.round(2))
    
    # --- Identify Top and Bottom Performers ---
    df['average_score'] = df[['clarity_score', 'engagement_score', 'educational_value_score']].mean(axis=1)
    
    print("\n## Top 3 Performing Lessons ##")
    top_3 = df.nlargest(3, 'average_score')
    for index, row in top_3.iterrows():
        print(f"- Score: {row['average_score']:.2f} | Topic: {row['topic']} ({row['grade_level']}) | File: {row['lesson_file']}")

    print("\n## Bottom 3 Performing Lessons ##")
    bottom_3 = df.nsmallest(3, 'average_score')
    for index, row in bottom_3.iterrows():
        print(f"- Score: {row['average_score']:.2f} | Topic: {row['topic']} ({row['grade_level']}) | File: {row['lesson_file']}")

    conn.close()

def export_for_tuning():
    """
    Connects to the database and exports the data into a JSONL format
    suitable for model fine-tuning.
    """
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database file not found at '{DB_PATH}'.")
        return

    print(f"\n--- Exporting data to '{TRAINING_DATA_FILE}' ---")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM interactions", conn)
    conn.close()

    if df.empty:
        print("⚠️ No data to export.")
        return

    with open(TRAINING_DATA_FILE, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            # For fine-tuning, the 'prompt' should ideally be what the LLM sees.
            # Here, we construct a simplified prompt.
            prompt = f"Create a lesson plan and quiz about '{row['topic']}' for {row['grade_level']} students."

            # We need to read the content of the lesson plan for the 'response'.
            try:
                with open(row['lesson_file'], 'r', encoding='utf-8') as lesson_f:
                    # We strip the reports we added at the end to get the pure LLM output
                    response_text = lesson_f.read().split("\n---\n### Evaluation Report\n---")[0]
            except FileNotFoundError:
                print(f"⚠️ Warning: Could not find lesson file {row['lesson_file']}. Skipping record.")
                continue
            
            # The 'chosen' response is the one we have scores for.
            # In a more advanced scenario, you might have a 'rejected' response as well.
            training_record = {
                "prompt": prompt,
                "chosen_response": response_text,
                "scores": {
                    "clarity": row['clarity_score'],
                    "engagement": row['engagement_score'],
                    "educational_value": row['educational_value_score']
                }
            }
            f.write(json.dumps(training_record) + '\n')
            
    print(f"✅ Successfully exported {len(df)} records to {TRAINING_DATA_FILE}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and export feedback data for Project Sahayak."
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export the data to a JSONL file for fine-tuning instead of showing analysis."
    )
    args = parser.parse_args()

    if args.export:
        export_for_tuning()
    else:
        analyze_performance()