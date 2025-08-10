import io
import re
import json
import string

from copy import deepcopy
from collections import defaultdict
from unidecode import unidecode

class NGramBuilder:

    def __init__(self, n: int):
        self.n = n
        self.vocab = defaultdict(int)
        self.model = defaultdict(lambda: defaultdict(int))

    
    def add_source(self, text: str|io.TextIOWrapper):
        if isinstance(text, str):
            tokens = NGramBuilder.generate_tokens(text)
        
        elif isinstance(text, io.TextIOWrapper):
            try:
                tokens = NGramBuilder.generate_tokens(text.read())
            except ValueError:
                raise ValueError(f"Cannot add source: '{text.name}' is closed")
            
        else:
            raise ValueError("Argument to add_source must be file or string")
        
        offsetted_sequences = [
            tokens[offset:]
            for offset in range(self.n)
        ]

        for token_tuple in zip(*offsetted_sequences):
            *key_tokens, next_token = token_tuple
            key = " ".join(key_tokens)
            self.model[key][next_token] += 1
        
        for token in tokens:
            self.vocab[token] += 1

    
    def save(self, model_file: str|io.TextIOWrapper):
        if isinstance(model_file, str):
            with open(model_file, "w") as file_object:
                json.dump(self.data, file_object, indent=2)

        elif isinstance(model_file, io.TextIOWrapper):
            try:
                json.dump(self.data, model_file, indent=2)
            except ValueError:
                raise ValueError(f"Cannot save: '{model_file.text}' is closed")
        
        else:
            raise ValueError(
                "Argument to save has to be file object or path to file (str)."
            )


    def copy(self):
        duplicate = NGramBuilder(n=self.n)
        duplicate.vocab = deepcopy(self.vocab)
        duplicate.model = deepcopy(self.model)
        return duplicate
    
    
    def __add__(self, other_builder):
        """ Combines two models """
        if self.n != other_builder.n:
            raise ValueError("Adding Incompatible Models: N not same.")
        
        duplicate = self.copy()
        
        for token, count in other_builder.vocab.items():
            duplicate.vocab[token] += count

        for key, next_tokens in other_builder.model.items():
            for token, count in next_tokens.items():
                duplicate.model[key][token] += count

        return duplicate

    
    @staticmethod
    def generate_tokens(text):
        precleaned = unidecode(text).lower()
        no_spaces = re.sub(r"\s+", " ", precleaned)
        punct_spaced = re.sub(f"([{string.punctuation}])", r" \g<0>", no_spaces)
        ellipsis_restored = punct_spaced.replace(". . .", "...")
        quote_t_restored = ellipsis_restored.replace(" 't", "'t")
        return quote_t_restored.split()


    @property
    def data(self):
        return {
            "vocab": dict(self.vocab),
            "model": {k: dict(v) for k, v in self.model.items()}
        }