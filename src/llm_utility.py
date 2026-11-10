
import os
import sys
import json
import csv
import time
from pathlib import Path
from datetime import datetime, timezone

from google import genai
from google.genai import types

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gateway import deidentify

TEST_FILE = ROOT / "data" / "test.jsonl"
OUTPUT_DIR = ROOT / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_JSON = OUTPUT_DIR / "llm_utility_results.json"
RESULT_CSV = OUTPUT_DIR / "llm_utility_scoring.csv"

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

NUMBER_OF_NOTES = 5
MAX_ATTEMPTS = 4

PROMPT = """
You are assisting with a synthetic patient-note analysis task.

Read the supplied note and summarize its important information
in 1-2 concise sentences.

Instructions:
- Preserve all relevant clinical or contextual facts.
- Do not invent information or make unsupported assumptions.
- Do not guess the identity or value represented by a redaction.
- Treat bracketed placeholders such as [NAME_1] and [DATE_1]
  as intentionally redacted information.
- Do not attempt to reconstruct redacted information.
- Do not mention that you are an AI.

Return only the summary.
"""


# --------------------------------------------------
# LOAD TEST NOTES
# --------------------------------------------------

def load_test_notes():

    notes = []

    with open(TEST_FILE, "r", encoding="utf-8") as file:

        for line in file:

            if not line.strip():
                continue

            row = json.loads(line)
            notes.append(" ".join(row["tokens"]))

    if len(notes) < NUMBER_OF_NOTES:
        raise ValueError("Test dataset contains fewer than five notes.")

    return notes[:NUMBER_OF_NOTES]


# --------------------------------------------------
# SAVE CHECKPOINTS
# --------------------------------------------------

def save_checkpoint(results):

    report = {
        "experiment": "LLM Utility Comparison",
        "model": MODEL_NAME,
        "test_notes": NUMBER_OF_NOTES,
        "generation_temperature": 0,
        "prompt": PROMPT,
        "last_updated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "completed_pairs": sum(
            1 for item in results
            if item.get("masked_answer")
        ),
        "results": results
    }

    # Save JSON after every successful API response.
    with open(RESULT_JSON, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    # Save scoring sheet after every successful API response.
    fieldnames = [
        "note_number",
        "original_answer",
        "masked_answer",
        "original_usefulness_1_to_5",
        "masked_usefulness_1_to_5",
        "fact_preservation_1_to_5",
        "hallucination_observed",
        "meaning_preserved",
        "reviewer_comments"
    ]

    with open(
        RESULT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item in results:
            writer.writerow({
                "note_number": item["note_number"],
                "original_answer": item.get("original_answer", ""),
                "masked_answer": item.get("masked_answer", ""),
                "original_usefulness_1_to_5": "",
                "masked_usefulness_1_to_5": "",
                "fact_preservation_1_to_5": "",
                "hallucination_observed": "",
                "meaning_preserved": "",
                "reviewer_comments": ""
            })


# --------------------------------------------------
# LOAD EXISTING CHECKPOINT
# --------------------------------------------------

def load_checkpoint():

    if not RESULT_JSON.exists():
        return []

    try:
        with open(RESULT_JSON, "r", encoding="utf-8") as file:
            report = json.load(file)

        if report.get("model") != MODEL_NAME:
            print(
                "Existing checkpoint uses a different model. "
                "Starting a new experiment."
            )
            return []

        results = report.get("results", [])

        print(
            f"Resuming checkpoint: {len(results)} "
            "previously started notes found."
        )

        return results

    except (json.JSONDecodeError, OSError, AttributeError):
        print("Could not read checkpoint. Starting a new experiment.")
        return []


# --------------------------------------------------
# GEMINI REQUEST WITH RETRIES
# --------------------------------------------------

def ask_gemini(client, note):

    prompt = PROMPT + "\n\nNOTE:\n" + note

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:

            print(f"Gemini request attempt {attempt}/{MAX_ATTEMPTS}")

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=250,
                    candidate_count=1
                )
            )

            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")

            return response.text.strip()

        except Exception as error:

            error_text = str(error).upper()

            temporary_error = any(
                term in error_text
                for term in [
                    "503",
                    "UNAVAILABLE",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "500",
                    "INTERNAL"
                ]
            )

            if temporary_error and attempt < MAX_ATTEMPTS:

                wait_seconds = 5 * (2 ** (attempt - 1))

                print(
                    f"Temporary Gemini error: {error}\n"
                    f"Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

            else:
                print("Gemini request failed:", error)
                raise

    raise RuntimeError("Gemini request failed after retries.")


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def main():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is missing. "
            "Configure it in PowerShell."
        )

    print("=" * 65)
    print("LLM UTILITY EVALUATION")
    print("=" * 65)

    print("Model:", MODEL_NAME)
    print("Number of notes:", NUMBER_OF_NOTES)

    original_notes = load_test_notes()
    results = load_checkpoint()

    # Index existing results by note number.
    result_lookup = {
        item["note_number"]: item
        for item in results
    }

    client = genai.Client(api_key=api_key)

    try:

        for index, original_note in enumerate(original_notes, 1):

            print(f"\nProcessing note {index}/{NUMBER_OF_NOTES}")

            if index not in result_lookup:

                result = {
                    "note_number": index,
                    "original_note": original_note,
                    "masked_note": "",
                    "original_answer": "",
                    "masked_answer": "",
                    "mapping_count": 0
                }

                results.append(result)
                result_lookup[index] = result

            else:
                result = result_lookup[index]

            # Checkpoint original note text.
            if result["original_note"] != original_note:
                raise ValueError(
                    f"Test note {index} differs from saved checkpoint. "
                    "Do not mix results from different test data."
                )

            # De-identification.
            if not result.get("masked_note"):

                masked_note, private_mapping = deidentify(
                    original_note
                )

                result["masked_note"] = masked_note
                result["mapping_count"] = len(private_mapping)

                print(
                    "Identifiers replaced:",
                    len(private_mapping)
                )

                # Mapping values are deliberately discarded.
                del private_mapping

                save_checkpoint(results)

            # Original-note request.
            if not result.get("original_answer"):

                print("Generating original-note answer...")

                original_answer = ask_gemini(
                    client,
                    original_note
                )

                result["original_answer"] = original_answer

                # Save immediately.
                save_checkpoint(results)

                print("Original answer:")
                print(original_answer)

            else:
                print("Original answer already saved. Skipping request.")

            # Masked-note request.
            if not result.get("masked_answer"):

                print("\nGenerating masked-note answer...")

                masked_answer = ask_gemini(
                    client,
                    result["masked_note"]
                )

                result["masked_answer"] = masked_answer

                # Save immediately.
                save_checkpoint(results)

                print("Masked answer:")
                print(masked_answer)

            else:
                print("Masked answer already saved. Skipping request.")

            print("\n" + "-" * 65)

    finally:
        client.close()

    completed = sum(
        1 for item in results
        if item.get("original_answer") and item.get("masked_answer")
    )

    print("\n" + "=" * 65)
    print("LLM UTILITY EXPERIMENT STATUS")
    print("=" * 65)

    print("Completed note pairs:", completed, "/", NUMBER_OF_NOTES)
    print("Expected Gemini requests:", NUMBER_OF_NOTES * 2)
    print("JSON report:", RESULT_JSON)
    print("Scoring sheet:", RESULT_CSV)

    if completed == NUMBER_OF_NOTES:
        print("\nALL FIVE NOTE PAIRS COMPLETED.")
    else:
        print("\nPARTIAL RESULTS SAVED. Run again to resume.")

    print("Private mappings were not saved or sent to Gemini.")


if __name__ == "__main__":
    main()
