from __future__ import annotations
from typing import Self

import io
import json

from collections import defaultdict

class NGramGenerator:

    def __init__(self):
        self.n = None
        self.vocab = defaultdict(int)
        self.model = defaultdict(lambda: defaultdict(int))

    
    def load_model(self, modelfile: str|io.TextIOWrapper) -> None:
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
            tentative_n = len(first_key.split())
            
            for key in model:
                if len(key.split()) != tentative_n:
                    raise ValueError(
                        "Corrupted Model: Ngram length (n) is inconsistent"
                    )
            
            if self.n is None:
                self.n = tentative_n
            elif self.n != tentative_n:
                raise ValueError(
                    "Cannot load model: mismatch in n (Ngram length)"
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
        

    def __iter__(self, init_phrase: str|tuple[str]|list[str]) -> Self:
        if isinstance(init_phrase, str):
            self.state = init_phrase.split()
            if len(self.state != self.n - 1):
                raise ValueError(
                    f"Cannot generate with starting phrase {self.state}: "
                    f"Number of tokens does not match (N - 1) for this model"
                )
            
        if isinstance(init_phrase, list):
            self.state = [*init_phrase]
            if len(self.state != self.n - 1):
                raise ValueError(
                    f"Cannot generate with starting phrase {self.state}: "
                    f"Number of tokens does not match (N - 1) for this model"
                )
        
        if isinstance(init_phrase, tuple):
            self.state = list(init_phrase)
            if len(self.state != self.n - 1):
                raise ValueError(
                    f"Cannot generate with starting phrase {self.state}: "
                    f"Number of tokens does not match (N - 1) for this model"
                )
            
        return self