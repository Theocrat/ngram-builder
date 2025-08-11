from __future__ import annotations
from typing import Self

import io
import json
import random

from collections import defaultdict

class NGramGenerator:

    def __init__(self):
        self.n = None
        self.vocab = defaultdict(int)
        self.model = defaultdict(lambda: defaultdict(int))
        self.state = None

    
    def load_file(self, modelfile: str|io.TextIOWrapper) -> None:
        if isinstance(modelfile, str):
            try:
                with open(modelfile) as sourcefile:
                    data = json.load(sourcefile)
            
            except FileNotFoundError:
                raise ValueError(
                    f"Cannot load model: No file named {modelfile}'"
                )
            
            except json.JSONDecodeError:
                raise ValueError(
                    f"Cannot load model: broken JSON in file {modelfile}"
                )
        
        if isinstance(modelfile, io.TextIOWrapper):
            try:
                data = json.load(modelfile)
            except ValueError as ve:
                raise ValueError(f"Cannot load model: {str(ve)}")
            except json.JSONDecodeError:
                raise ValueError("Cannot load model from file: Broken JSON")
            
        try:
            vocab = data["vocab"]
            model = data["model"]

            model_keys = list(model.keys())
            first_key = model_keys[0]
            tentative_n = len(first_key.split()) + 1
            
            for key in model:
                if len(key.split()) != tentative_n - 1:
                    raise ValueError(
                        "Corrupted Model: Ngram length (n) is inconsistent"
                    )
            
            if self.n is None:
                self.n = tentative_n
            elif self.n != tentative_n:
                raise ValueError(
                    "Cannot load model: mismatch in n (Ngram context length)"
                )

            for token, count in vocab.items():
                self.vocab[token] += count
            
            for key, next_token in model.items():
                for token, count in next_token.items():
                    self.model[key][token] += count
            
        except KeyError as bad_key:
            raise ValueError(
                f"Cannot load model: Model JSON doesn't contain field {bad_key}"
            )
        
        except IndexError:
            raise ValueError("Cannot load model: Model appears to be empty")
        

    def __call__(self, init_key: str|tuple[str]|list[str]) -> Self:
        if isinstance(init_key, str):
            self.state = init_key.split()
            if len(self.state) != self.n - 1:
                raise ValueError(
                    f"Cannot generate with starting phrase {self.state}: "
                    f"Number of tokens does not match (N - 1) for this model"
                )
            
        if isinstance(init_key, list):
            self.state = [*init_key]
            if len(self.state) != self.n - 1:
                raise ValueError(
                    f"Cannot generate with starting phrase {self.state}: "
                    f"Number of tokens does not match (N - 1) for this model"
                )
        
        if isinstance(init_key, tuple):
            self.state = list(init_key)
            if len(self.state) != self.n - 1:
                raise ValueError(
                    f"Cannot generate with starting phrase {self.state}: "
                    f"Number of tokens does not match (N - 1) for this model"
                )
            
        return self
    

    def __iter__(self) -> Self:
        self.vocab_spreadout = []
        for token, count in self.vocab.items():
            self.vocab_spreadout.extend([token] * count)
        return self
    

    def __next__(self) -> str:
        if self.state is None:
            raise StopIteration
        
        keyphrase = ' '.join(self.state)
        
        if keyphrase in self.model:
            candidates = []
            for token, count in self.model[keyphrase].items():
                candidates.extend([token] * count)
            
            choice = random.choice(candidates)
            self.state.pop(0)
            self.state.append(choice)
            
            return choice
        
        else:
            choice = random.choice(self.vocab_spreadout)
            self.state.pop(0)
            self.state.append(choice)
            return choice