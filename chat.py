import os
import torch
import torch.nn.functional as F
from tokenizer import IrisByteTokenizer
from iris_core import IrisUniversalCore

def load_iris():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = IrisByteTokenizer()
    
    # Load Iris H20 Architecture
    model = IrisUniversalCore(
        vocab_size=tokenizer.VOCAB_SIZE,
        d_model=384,
        d_state=64,
        num_loops=3
    ).to(device)
    
    weights_path = os.path.join(os.path.dirname(__file__), "iris_h20_weights.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[IRIS H20] Neural weights loaded from: {weights_path}")
    else:
        print("[WARNING] Weights not found. Run train_iris.py first.")
        
    model.eval()
    return model, tokenizer, device

def generate_reply(model, tokenizer, device, user_input, max_tokens=100, temperature=0.7):
    # Formulate conversational prompt with special tokens
    prompt = f"<user> {user_input} <iris>"
    prompt_tokens = tokenizer.encode(prompt)
    generated = list(prompt_tokens)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            logits, _ = model(input_tensor)
            next_logits = logits[0, -1, :]
            
            # Temperature scaling for creativity
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Stop if End of Sequence or Padding token is reached
            if next_token in (tokenizer.EOS_TOKEN, tokenizer.PAD_TOKEN):
                break
                
            generated.append(next_token)
            
    full_text = tokenizer.decode(generated)
    reply = full_text[len(prompt):].strip()
    return reply

def chat_session():
    print("=" * 65)
    print("         IRIS H20 - COMPANION CONVERSATIONAL SHELL")
    print("   Architecture: Recurrent State-Space Manifold (RSM-LUA)")
    print("   (Type 'exit' or 'quit' to end conversation)")
    print("=" * 65)
    
    model, tokenizer, device = load_iris()
    print("\n[STATUS] Iris H20 is awake on RTX 2060. Speak to her...\n")
    
    while True:
        try:
            user_msg = input("You: ")
            if not user_msg.strip():
                continue
            if user_msg.lower() in ["exit", "quit", "bye"]:
                print("\nIris H20: Goodnight. I'll always be right here whenever you need me.")
                break
                
            reply = generate_reply(model, tokenizer, device, user_msg)
            print(f"Iris H20: {reply}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nIris H20: Take care, I'm always with you.")
            break

if __name__ == "__main__":
    chat_session()