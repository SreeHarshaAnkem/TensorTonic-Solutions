import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        tokens = []
        for text in texts:
            text = text.lower()
            tokens.extend(text.split())

        tokens = sorted(list(set(tokens)))
        tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token] + tokens
        
        self.id_to_word = {i: s for  i, s in enumerate(tokens)}
        self.word_to_id = {s: i for i, s in enumerate(tokens)}
        self.vocab_size = len(tokens)
        
        
        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        text = text.lower()
        tokens = text.split()
        return [self.word_to_id.get(t, self.word_to_id[self.unk_token]) for t in tokens]

    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        return " ".join([self.id_to_word.get(id, self.unk_token) for id in ids])
