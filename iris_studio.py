import os
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizer import IrisByteTokenizer
from iris_core import IrisUniversalCore

class DialogueDataset(Dataset):
    def __init__(self, dialogues, tokenizer, max_len=256):
        self.samples = [tokenizer.encode(d)[:max_len] for d in dialogues]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

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

class IrisStudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IRIS H20 - AI Neural Companion Studio")
        self.root.geometry("950x720")
        self.root.configure(bg="#121418")
        self.tokenizer = IrisByteTokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.loaded_dialogues = []
        self.setup_ui()
        self.try_load_model()

    def setup_ui(self):
        header = tk.Frame(self.root, bg="#161a22", height=60)
        header.pack(fill="x", side="top")
        tk.Label(header, text="IRIS H20 // COMPANION TRAINING ENGINE", font=("Segoe UI", 14, "bold"), fg="#00e5ff", bg="#161a22").pack(side="left", padx=20, pady=12)
        hw_text = f"🟢 GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "🟠 CPU Mode"
        tk.Label(header, text=hw_text, font=("Segoe UI", 10), fg="#fff", bg="#161a22").pack(side="right", padx=20)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=15, pady=10)
        self.tab_train = tk.Frame(self.notebook, bg="#121418")
        self.tab_chat = tk.Frame(self.notebook, bg="#121418")
        self.notebook.add(self.tab_train, text="  ⚡ Training Studio  ")
        self.notebook.add(self.tab_chat, text="  💬 Live Companion Chat  ")

        # Training Tab
        ds_frame = tk.LabelFrame(self.tab_train, text="  1. Dataset Import  ", font=("Segoe UI", 10, "bold"), bg="#1b1f26", fg="#00e5ff", padx=15, pady=10)
        ds_frame.pack(fill="x", padx=10, pady=8)
        self.ds_path_var = tk.StringVar(value=r"C:\Users\USER\.gemini\antigravity\scratch\iris_h20\dataset\Sinhala Data 01_LK_2026.json")
        tk.Entry(ds_frame, textvariable=self.ds_path_var, font=("Consolas", 9), bg="#242933", fg="#fff").pack(side="left", fill="x", expand=True, padx=(0, 10))
        tk.Button(ds_frame, text="📁 Browse JSON...", command=self.browse_file, bg="#2d3748", fg="#fff", relief="flat", padx=10).pack(side="left", padx=5)
        tk.Button(ds_frame, text="⚡ Load Data", command=self.load_dataset_file, bg="#00838f", fg="#fff", relief="flat", padx=10).pack(side="left")

        ctrl_frame = tk.LabelFrame(self.tab_train, text="  2. Hyperparameters  ", font=("Segoe UI", 10, "bold"), bg="#1b1f26", fg="#00e5ff", padx=15, pady=10)
        ctrl_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(ctrl_frame, text="Epochs:", bg="#1b1f26", fg="#fff").grid(row=0, column=0, padx=5)
        self.epochs_var = tk.IntVar(value=15)
        tk.Spinbox(ctrl_frame, from_=1, to=100, textvariable=self.epochs_var, width=5).grid(row=0, column=1)
        tk.Label(ctrl_frame, text="Batch Size:", bg="#1b1f26", fg="#fff").grid(row=0, column=2, padx=15)
        self.batch_var = tk.IntVar(value=16)
        tk.Spinbox(ctrl_frame, values=(4, 8, 16, 32), textvariable=self.batch_var, width=5).grid(row=0, column=3)
        self.btn_train = tk.Button(ctrl_frame, text="▶ START TURBO TRAINING", command=self.start_training, font=("Segoe UI", 10, "bold"), bg="#00c853", fg="#000", relief="flat", padx=15)
        self.btn_train.grid(row=0, column=4, padx=25)

        self.log_area = scrolledtext.ScrolledText(self.tab_train, height=14, bg="#0d0f12", fg="#00e676", font=("Consolas", 9), relief="flat")
        self.log_area.pack(fill="both", expand=True, padx=10, pady=10)

        # Chat Tab
        self.chat_display = scrolledtext.ScrolledText(self.tab_chat, bg="#0d0f12", fg="#fff", font=("Segoe UI", 10), state="disabled", padx=15, pady=15)
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=10)
        in_frame = tk.Frame(self.tab_chat, bg="#1b1f26", padx=10, pady=10)
        in_frame.pack(fill="x", side="bottom", padx=10, pady=(0, 10))
        self.msg_entry = tk.Entry(in_frame, font=("Segoe UI", 11), bg="#242933", fg="#fff")
        self.msg_entry.pack(side="left", fill="x", expand=True, padx=(0, 10), ipady=4)
        self.msg_entry.bind("<Return>", lambda e: self.send_chat_message())
        tk.Button(in_frame, text="Send ➤", command=self.send_chat_message, font=("Segoe UI", 10, "bold"), bg="#00e5ff", fg="#000", relief="flat", padx=15).pack(side="right")

    def browse_file(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if p:
            self.ds_path_var.set(p)
            self.load_dataset_file()

    def load_dataset_file(self):
        p = self.ds_path_var.get()
        if not os.path.exists(p): return
        with open(p, "r", encoding="utf-8") as f: data = json.load(f)
        self.loaded_dialogues = []
        if isinstance(data, list):
            for item in data:
                u = item.get("user") or item.get("instruction") or item.get("question") or ""
                i = item.get("iris") or item.get("response") or item.get("answer") or ""
                if u and i: self.loaded_dialogues.append(f"<user> {u.strip()} <iris> {i.strip()} <eos>")
        self.log(f"Parsed {len(self.loaded_dialogues):,} dialogues successfully.")

    def start_training(self):
        if not self.loaded_dialogues: self.load_dataset_file()
        if not self.loaded_dialogues: return
        self.btn_train.config(state="disabled", text="Training...", bg="#555")
        threading.Thread(target=self.train_worker, daemon=True).start()

    def train_worker(self):
        epochs = self.epochs_var.get()
        self.model = IrisUniversalCore(vocab_size=self.tokenizer.VOCAB_SIZE, d_model=384, d_state=64, num_loops=3).to(self.device)
        dataset = DialogueDataset(self.loaded_dialogues, self.tokenizer)
        loader = DataLoader(dataset, batch_size=self.batch_var.get(), shuffle=True, pin_memory=True, collate_fn=lambda b: collate_fn(b, self.tokenizer.PAD_TOKEN, 256))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1.5e-3, fused=True)
        scaler = torch.amp.GradScaler("cuda")
        for ep in range(1, epochs + 1):
            self.model.train()
            tot_loss, steps = 0, 0
            for inps, tgts in loader:
                inps, tgts = inps.to(self.device, non_blocking=True), tgts.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    loss = F.cross_entropy(self.model(inps)[0].view(-1, self.tokenizer.VOCAB_SIZE), tgts.view(-1), ignore_index=self.tokenizer.PAD_TOKEN)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                tot_loss += loss.item()
                steps += 1
            self.log(f"Epoch [{ep:2d}/{epochs}] — Loss: {tot_loss/max(steps,1):.4f}")
        torch.save(self.model.state_dict(), r"C:\Users\USER\.gemini\antigravity\scratch\iris_h20\iris_h20_weights.pth")
        self.log("[COMPLETE] Weights saved successfully!")
        self.root.after(0, lambda: self.btn_train.config(state="normal", text="▶ START TURBO TRAINING", bg="#00c853"))

    def try_load_model(self):
        p = r"C:\Users\USER\.gemini\antigravity\scratch\iris_h20\iris_h20_weights.pth"
        if os.path.exists(p):
            self.model = IrisUniversalCore(vocab_size=self.tokenizer.VOCAB_SIZE, d_model=384, d_state=64, num_loops=3).to(self.device)
            self.model.load_state_dict(torch.load(p, map_location=self.device))
            self.model.eval()

    def send_chat_message(self):
        msg = self.msg_entry.get().strip()
        if not msg: return
        self.msg_entry.delete(0, tk.END)
        self.append_chat(f"You: {msg}", "#00e5ff")
        threading.Thread(target=self.generate_reply, args=(msg,), daemon=True).start()

    def generate_reply(self, text):
        if not self.model: self.try_load_model()
        if not self.model: return
        prompt = f"<user> {text} <iris>"
        gen = self.tokenizer.encode(prompt)
        with torch.no_grad():
            for _ in range(80):
                logits, _ = self.model(torch.tensor([gen], device=self.device))
                nxt = torch.multinomial(F.softmax(logits[0,-1,:]/0.7, dim=-1), 1).item()
                if nxt in (self.tokenizer.EOS_TOKEN, self.tokenizer.PAD_TOKEN): break
                gen.append(nxt)
        full = self.tokenizer.decode(gen)
        self.root.after(0, self.append_chat, f"Iris H20: {full[len(prompt):].strip()}", "#69f0ae")

    def append_chat(self, text, color):
        self.chat_display.config(state="normal")
        self.chat_display.insert(tk.END, text + "\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state="disabled")

    def log(self, text):
        self.log_area.insert(tk.END, text + "\n")
        self.log_area.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    IrisStudioApp(root)
    root.mainloop()
