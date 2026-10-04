import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizer import IrisByteTokenizer
from iris_core import IrisUniversalCore

class IrisDialogueDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len=256):
        with open(json_path, "r", encoding="utf-8") as f:
            self.dialogues = json.load(f)
        self.tokenizer = tokenizer
        self.samples = [tokenizer.encode(d)[:max_len] for d in self.dialogues]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch, pad_token, max_len=256):
    batch_max = min(max(len(x) for x in batch), max_len)
    padded_inputs, padded_targets = [], []
    for tokens in batch:
        inp = tokens[:-1]
        tgt = tokens[1:]
        pad_len = batch_max - len(tokens)
        if pad_len > 0:
            inp = inp + [pad_token] * pad_len
            tgt = tgt + [pad_token] * pad_len
        else:
            inp = inp[:batch_max-1]
            tgt = tgt[:batch_max-1]
        padded_inputs.append(inp)
        padded_targets.append(tgt)
    return torch.tensor(padded_inputs, dtype=torch.long), torch.tensor(padded_targets, dtype=torch.long)

def run_turbo_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Hardware: {torch.cuda.get_device_name(0)}")
    corpus_path = r"C:\Users\USER\.gemini\antigravity\scratch\iris_h20\sinhala_corpus.json"
    if not os.path.exists(corpus_path):
        print(f"[ERROR] Run prepare_data.py first!")
        return
    tokenizer = IrisByteTokenizer()
    dataset = IrisDialogueDataset(corpus_path, tokenizer, max_len=256)
    print(f"[DATASET] Loaded {len(dataset):,} Dialogues.")
    loader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True, collate_fn=lambda b: collate_fn(b, tokenizer.PAD_TOKEN, 256))
    model = IrisUniversalCore(vocab_size=tokenizer.VOCAB_SIZE, d_model=384, d_state=64, num_loops=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    epochs = 15
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, steps = 0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits, _ = model(inputs)
                loss = F.cross_entropy(logits.view(-1, tokenizer.VOCAB_SIZE), targets.view(-1), ignore_index=tokenizer.PAD_TOKEN)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            steps += 1
        avg_loss = total_loss / max(steps, 1)
        vram_mb = torch.cuda.memory_allocated() / (1024**2)
        print(f"Epoch [{epoch:2d}/{epochs}] | Loss: {avg_loss:.4f} | VRAM: {vram_mb:.1f} MB")
    print(f"\n[COMPLETE] Training finished in {time.time() - start_time:.2f}s!")
    weights_path = r"C:\Users\USER\.gemini\antigravity\scratch\iris_h20\iris_h20_weights.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"[SAVED] Saved weights to {weights_path}")

if __name__ == "__main__":
    run_turbo_training()
