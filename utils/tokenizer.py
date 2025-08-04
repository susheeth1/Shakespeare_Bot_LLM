# File: utils/tokenizer.py
import json

class CharacterTokenizer:
    """A simple character-level tokenizer."""

    def __init__(self, corpus: str = None):
        if corpus:
            self.chars = sorted(list(set(corpus)))
            self.vocab_size = len(self.chars)
            self.stoi = {ch: i for i, ch in enumerate(self.chars)}
            self.itos = {i: ch for i, ch in enumerate(self.chars)}
        else:
            self.chars = []
            self.vocab_size = 0
            self.stoi = {}
            self.itos = {}

    def encode(self, s: str) -> list[int]:
        """Converts a string to a list of integer token IDs."""
        return [self.stoi[c] for c in s]

    def decode(self, l: list[int]) -> str:
        """Converts a list of integer token IDs back to a string."""
        return ''.join([self.itos[i] for i in l])

    def save(self, file_path: str):
        """Saves the vocabulary to a file."""
        vocab_data = {
            'chars': self.chars,
            'vocab_size': self.vocab_size,
            'stoi': self.stoi,
            'itos': self.itos
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
            
    @classmethod
    def load(cls, file_path: str):
        """Loads the tokenizer from a saved vocabulary file."""
        tokenizer = cls()
        with open(file_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        tokenizer.chars = vocab_data['chars']
        # The keys in JSON are strings, so we convert them back to integers for itos
        tokenizer.itos = {int(k): v for k, v in vocab_data['itos'].items()}
        tokenizer.stoi = vocab_data['stoi']
        tokenizer.vocab_size = vocab_data['vocab_size']
        
        return tokenizer