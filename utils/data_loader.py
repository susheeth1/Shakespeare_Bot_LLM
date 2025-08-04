# File: utils/data_loader.py
import torch
from utils.tokenizer import CharacterTokenizer

def get_data_loader(config: dict):
    """
    Prepares the data and returns a function to get batches.
    """
    # 1. Read the dataset
    with open(config['data_path'], 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Initialize tokenizer and encode the data
    tokenizer = CharacterTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # 3. Split into training and validation sets
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_batch(split: str):
        """
        Generates a small batch of data with inputs x and targets y.
        
        Args:
            split (str): 'train' or 'val' to select the dataset.
        
        Returns:
            A tuple of (x, y) tensors.
        """
        data = train_data if split == 'train' else val_data
        
        # Randomly select starting indices for the batch
        ix = torch.randint(len(data) - config['model_params']['block_size'], (config['training_params']['batch_size'],))
        
        # Create input sequences (x)
        x = torch.stack([data[i:i+config['model_params']['block_size']] for i in ix])
        
        # Create target sequences (y), which are shifted by one position
        y = torch.stack([data[i+1:i+config['model_params']['block_size']+1] for i in ix])
        
        return x.to(device), y.to(device)

    return get_batch, tokenizer