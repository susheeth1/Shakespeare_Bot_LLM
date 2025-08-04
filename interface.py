# File: interface.py
import torch
import json
import os

from modules.model import NanoLLM
from utils.tokenizer import CharacterTokenizer

# --- Configuration ---
MODEL_DIR = 'generated_model'

def main():
    # --- Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Load Tokenizer ---
    tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.json')
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer file not found at {tokenizer_path}")
        return
    tokenizer = CharacterTokenizer.load(tokenizer_path)
    print(f"Tokenizer loaded from {tokenizer_path}")

    # --- Load Checkpoint and Model ---
    checkpoint_path = os.path.join(MODEL_DIR, 'model_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device) # <-- Load the whole checkpoint
    
    model_config = checkpoint['model_config']
    model = NanoLLM(**model_config)
    model.load_state_dict(checkpoint['model_state_dict']) # <-- Load weights from the checkpoint
    model.to(device)
    model.eval()
    
    # --- Display Model Info ---
    print("\n--- Loaded Model Information ---")
    print(f"Model saved at: {checkpoint.get('saved_at', 'N/A')}")
    print(f"Trained for: {checkpoint.get('final_iter', -1) + 1} iterations")
    print(f"Final Loss: {checkpoint.get('final_loss', -1):.4f}")
    print("------------------------------")


    # --- Interactive Chat Loop ---
    # ... (This part remains the same)
    print("\n--- Shakespeare Bot Interface ---")
    print("Enter a prompt to start the generation.")
    print("Type 'exit' or 'quit' to end the session.")
    print("---------------------------------")
    
    while True:
        prompt = input("\n> ")
        if prompt.lower() in ['exit', 'quit']:
            print("Farewell!")
            break
        
        if not prompt:
            prompt = '\n'
        
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        
        print("Hark, the bot doth ponder...")
        
        generated_indices = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=20)
        generated_text = tokenizer.decode(generated_indices[0].tolist())

        print("\n--- The Shakespeare Replies ---")
        print(generated_text)
        print("------------------------")

if __name__ == "__main__":
    main()