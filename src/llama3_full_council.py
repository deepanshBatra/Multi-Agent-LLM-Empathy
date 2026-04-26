import os
import json
import pandas as pd
import concurrent.futures
import threading
import sys
from tqdm import tqdm
from src.llama_sft_function_calls import *
from src.load_data import load_data_from_csv

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "test_sent_emo.csv")
BIO_CARDS_PATH = os.path.join(BASE_DIR, "logs", "speaker_bio_cards.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "logs", "council_llama3_1_results.csv")

# --- UTILS ---
def load_speaker_bio_cards(json_file=BIO_CARDS_PATH):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Bio cards error: {e}")
        return {}

def get_speaker_bio_card_text(speaker_name, bio_cards_dict):
    if not bio_cards_dict or speaker_name not in bio_cards_dict:
        return None
    bio_card = bio_cards_dict[speaker_name]
    if isinstance(bio_card, dict):
        return "\n".join([f"{k}: {v}" for k, v in bio_card.items()])
    return str(bio_card)

def extract_json_field(raw_str, field):
    import re
    try:
        raw_str = str(raw_str)
        
        # Strip markdown code fences if present
        raw_str = re.sub(r'^```json\s*', '', raw_str)
        raw_str = re.sub(r'\s*```$', '', raw_str)
        raw_str = raw_str.strip()
        
        # Find the JSON object with non-greedy matching
        match = re.search(r'\{.*?\}(?!.*\{)', raw_str, re.DOTALL)
        if match:
            json_str = match.group()
            data = json.loads(json_str)
            value = data.get(field, "unknown")
            
            # Type conversion for numeric fields
            if field == "confidence" and isinstance(value, (int, float)):
                return float(value)
            return value
    except Exception as e:
        print(f"JSON extraction error for field '{field}': {e}", flush=True)
        return "unknown"
    
    return "unknown"

# --- PIPELINE ---
def run_llama3_council_scene(scene_obj, bio_cards, global_history):
    utterances = scene_obj["utterances"]
    dialogue_id = scene_obj["dialogue_id"]
    scene_script = "\n".join([f"{u['Speaker']}: {u['Utterance']}" for u in utterances])

    # Level 1: Global Context
    print(f"Calling Context Manager for Scene {dialogue_id}...", flush=True)
    global_context = call_llama3_context_manager(scene_script)
    print(f"Calling Relational Graph for Scene {dialogue_id}...", flush=True)
    social_graph = call_llama3_relational_graph(scene_script)

    scene_results = []
    
    # Process utterances in the scene
    for idx, u in enumerate(utterances):
        target_text = u.get('Utterance', "")
        speaker = u.get('Speaker', "Unknown")
        rec_id = u.get('Recognition_ID', "unknown_id")
        actual_emotion = u.get('Emotion', 'unknown')
        
        previous_turn = utterances[idx - 1] if idx > 0 else None
        prev_text = previous_turn.get('Utterance', "") if previous_turn else ""
        prev_speaker = previous_turn.get('Speaker', "Unknown") if previous_turn else "Unknown"

        print(f"Processing Utterance {idx+1}/{len(utterances)}: {rec_id}", flush=True)

        # Level 2: Specialists (Run in Parallel for the utterance)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            f_profile = executor.submit(call_llama3_profiler, target_text, social_graph)
            f_sentiment = executor.submit(call_llama3_sentiment, target_text)
            f_shift = executor.submit(call_llama3_emotional_shift, prev_text, prev_speaker, target_text, speaker, global_context)
            
            profile = f_profile.result()
            sentiment = f_sentiment.result()
            shift = f_shift.result()
            
            f_dynamics = executor.submit(call_llama3_social_dynamics, target_text, profile, social_graph)
            dynamics = f_dynamics.result()

        # Level 3: Aggregator
        bio_card = get_speaker_bio_card_text(speaker, bio_cards)
        prev_preds = global_history[-3:] if global_history else None
        
        raw_final = call_llama3_council_aggregator(
            rec_id, target_text, global_context, profile, sentiment, dynamics, shift,
            speaker_bio_card=bio_card, previous_predictions=prev_preds
        )
        
        print(f"Aggregator result for {rec_id}: {raw_final[:100]}...", flush=True)

        result = {
            "Dialogue_ID": dialogue_id,
            "Recognition_ID": rec_id,
            "Speaker": speaker,
            "Utterance": target_text,
            "Predicted_Emotion_Raw": raw_final,
            "Actual_Emotion": actual_emotion,
            "predicted_emotion": extract_json_field(raw_final, "predicted_emotion"),
            "confidence": extract_json_field(raw_final, "confidence"),
            "emotional_shift_report": shift
        }
        
        # Update history
        global_history.append({
            "utterance": target_text,
            "emotion": result["predicted_emotion"],
            "shift": "TRUE" if "SHIFT: TRUE" in shift else "FALSE"
        })
        if len(global_history) > 3: global_history.pop(0)
        
        scene_results.append(result)

    return scene_results

def main():
    print("Starting Llama 3.1 MELD Council Pipeline - VERSION 1.0.1 (Actual_Emotion fix)...", flush=True)
    print(f"Data Path: {DATA_PATH}", flush=True)
    print(f"Output File: {OUTPUT_FILE}", flush=True)
    
    # 1. Load Data
    df = load_data_from_csv(DATA_PATH)
    df['Recognition_ID'] = df['Dialogue_ID'].astype(str) + "_" + df['Utterance_ID'].astype(str)
    
    # Preprocess into scenes
    scenes = []
    for diag_id, group in df.groupby('Dialogue_ID'):
        scenes.append({
            "dialogue_id": diag_id,
            "utterances": group[['Utterance', 'Speaker', 'Recognition_ID', 'Emotion']].to_dict(orient='records')
        })
    
    # 2. Check for existing results (Resume capability)
    processed_dialogue_ids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            if 'Dialogue_ID' in existing_df.columns:
                processed_dialogue_ids = set(existing_df['Dialogue_ID'].unique())
                print(f"Found {len(processed_dialogue_ids)} scenes already processed. Resuming...", flush=True)
        except Exception as e:
            print(f"Could not read existing results: {e}. Starting fresh.", flush=True)

    # 3. Process scenes
    bio_cards = load_speaker_bio_cards()
    global_history = []
    
    csv_lock = threading.Lock()
    
    unprocessed_scenes = [s for s in scenes if s['dialogue_id'] not in processed_dialogue_ids]
    # LIMIT FOR TESTING - Remove after verification
    # unprocessed_scenes = unprocessed_scenes[:1] 
    
    print(f"Processing {len(unprocessed_scenes)} scenes.", flush=True)

    for scene in tqdm(unprocessed_scenes, desc="Processing Scenes"):
        try:
            scene_results = run_llama3_council_scene(scene, bio_cards, global_history)
            
            # Incremental save
            if scene_results:
                with csv_lock:
                    results_df = pd.DataFrame(scene_results)
                    # Handle column order consistency if file exists
                    file_exists = os.path.exists(OUTPUT_FILE)
                    results_df.to_csv(OUTPUT_FILE, mode='a', index=False, header=not file_exists)
                    print(f"Saved Scene {scene['dialogue_id']} to {OUTPUT_FILE}", flush=True)
        except Exception as e:
            print(f"Error in Scene {scene['dialogue_id']}: {e}", flush=True)
            continue

    print(f"Pipeline finished. Results saved to {OUTPUT_FILE}", flush=True)

if __name__ == "__main__":
    main()
