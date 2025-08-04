# File: trainer.py
import torch
import logging
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, get_batch_fn, config):
        self.model = model
        self.optimizer = optimizer
        self.get_batch = get_batch_fn
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    @torch.no_grad()
    def _estimate_loss(self):
        """ Estimates loss on train and val sets. """
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config['training_params']['eval_iters'])
            for k in range(self.config['training_params']['eval_iters']):
                X, Y = self.get_batch(split)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self):
        """ Runs the training loop and returns final training details. """ # <-- Updated docstring
        logging.info(f"Starting training on {self.device}...")
        
        self.model.to(self.device)
        
        pbar = tqdm(range(self.config['training_params']['max_iters']))
        final_loss = None
        final_iter = 0

        for iter_num in pbar:
            if iter_num % self.config['training_params']['eval_interval'] == 0 or iter_num == self.config['training_params']['max_iters'] - 1:
                losses = self._estimate_loss()
                logging.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = self.get_batch('train')
            _, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            final_loss = loss.item()
            final_iter = iter_num
            pbar.set_description(f"iter {iter_num}: loss {final_loss:.4f}")

        logging.info("Training finished.")
        return final_iter, final_loss # <-- Return final values