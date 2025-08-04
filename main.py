# File: main.py
import torch
import logging
import json
import os
from datetime import datetime # <-- Add this import

from modules.model import NanoLLM
from utils.data_loader import get_data_loader
from trainer import Trainer

# --- Configuration ---
CONFIG_PATH = 'configs/model_config.json'
SAVE_DIR = 'generated_model'

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def main():
    # --- Setup ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    config = load_config(CONFIG_PATH)
    model_config = config['model_params']
    train_config = config['training_params']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1337)
    os.makedirs(SAVE_DIR, exist_ok=True)
    logging.info(f"Using device: {device}")
    logging.info(f"Model Config: {model_config}")

    # --- Data Loading ---
    get_batch, tokenizer = get_data_loader(config)
    if model_config['vocab_size'] != tokenizer.vocab_size:
        logging.warning(f"Config vocab_size ({model_config['vocab_size']}) does not match tokenizer vocab_size ({tokenizer.vocab_size}). Overriding.")
        model_config['vocab_size'] = tokenizer.vocab_size

    # --- Model Initialization ---
    model = NanoLLM(**model_config)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Number of parameters: {num_params:.2f}M")

    # --- Training ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['learning_rate'])
    trainer = Trainer(model, optimizer, get_batch, config)
    final_iter, final_loss = trainer.train() # <-- Capture returned values

    # --- Saving the Checkpoint ---
    logging.info("Saving model checkpoint...")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': model_config,
        'final_iter': final_iter,
        'final_loss': final_loss,
        'saved_at': datetime.now().isoformat(),
        'torch_version': torch.__version__
    }

    model_save_path = os.path.join(SAVE_DIR, 'model_checkpoint.pth') # <-- Renamed for clarity
    torch.save(checkpoint, model_save_path)
    logging.info(f"Checkpoint saved to {model_save_path}")

    # Save the tokenizer
    tokenizer_save_path = os.path.join(SAVE_DIR, 'tokenizer.json')
    tokenizer.save(tokenizer_save_path)
    logging.info(f"Tokenizer saved to {tokenizer_save_path}")
    
    # --- Generation ---
    # ... (This part remains the same)
    logging.info("Generating a sample text with the trained model...")
    context = torch.tensor([[tokenizer.encode('\n')[0]]], dtype=torch.long, device=device)
    generated_indices = model.generate(context, max_new_tokens=500)
    generated_text = tokenizer.decode(generated_indices[0].tolist())
    
    print("\n--- SAMPLE GENERATED TEXT ---")
    print(generated_text)
    print("-----------------------------\n")

if __name__ == "__main__":
    main()