import os
import json

DATASET_PATH = r"C:\Users\USER\.gemini\antigravity\scratch\iris_h20\dataset\Sinhala Data 01_LK_2026.json"
OUTPUT_PATH = r"C:\Users\USER\.gemini\antigravity\scratch\iris_h20\sinhala_corpus.json"

def process_sinhala_dataset():
    print("=" * 60)
    print("     PREPARING SINHALA COMPANION DATASET FOR IRIS H20")
    print("=" * 60)
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] File not found at: {DATASET_PATH}")
        return
    print(f"[LOADING] Reading: {DATASET_PATH} ...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    formatted_dialogues = []
    if isinstance(data, list):
        for item in data:
            u_text, i_text = "", ""
            if isinstance(item, dict):
                u_text = item.get("user") or item.get("instruction") or item.get("question") or item.get("prompt") or item.get("input") or ""
                i_text = item.get("iris") or item.get("response") or item.get("answer") or item.get("output") or item.get("assistant") or ""
                if "conversations" in item and isinstance(item["conversations"], list):
                    for turn in item["conversations"]:
                        val = turn.get("value", "")
                        role = turn.get("from", "").lower()
                        if role in ["human", "user"]: u_text = val
                        elif role in ["gpt", "assistant", "iris", "bot"]: i_text = val
            if u_text and i_text:
                formatted_dialogues.append(f"<user> {u_text.strip()} <iris> {i_text.strip()} <eos>")
    print(f"[SUCCESS] Extracted & Formatted: {len(formatted_dialogues)} Sinhala Dialogues!")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(formatted_dialogues, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] Clean training corpus saved to: {OUTPUT_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    process_sinhala_dataset()
