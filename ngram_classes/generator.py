""" Defines the NGramGenerator class, which runs autoregression inference """

from __future__ import annotations
from typing import Self

import io
import json
import random

from collections import defaultdict

class NGramGenerator:
    """ N-GRAM GENERATOR
    Class for running prediction and autoregression tasks
    Usage:
        generator = NGramGenerator()
        generator.load_file("path/to/trigram/model.json")
        
        for token in generator(["token_1", "token_2"]):
            print(token, end=" ")
        print()
    """

    def __init__(self):
        """ Initializes the autoregressor with empty fields """
        self.param_n = None
        self.vocab = defaultdict(int)
        self.model = defaultdict(lambda: defaultdict(int))
        self.state = None
        self.vocab_spreadout = None


    def load_file(self, modelfile: str|io.TextIOWrapper) -> None:
        """ LOAD FILE: Loads a file containing an n-gram model
            This will add the model's data into this model.
            Arguments:
                - modelfile (str | io.TextIOWrapper): Either path to the model
                    file or the file itself, as a file (text I/O wrapper) object
            Returns: None
        """
        if isinstance(modelfile, str):
            try:
                with open(modelfile) as sourcefile:
                    data = json.load(sourcefile)

            except FileNotFoundError as nonexistent_file:
                raise FileNotFoundError(
                    f"Cannot load model: No file named {modelfile}'"
                ) from nonexistent_file

            except json.JSONDecodeError as broken_json:
                raise json.JSONDecodeError(
                    f"Cannot load model: broken JSON in file {modelfile}",
                    doc=broken_json.doc,
                    pos=broken_json.pos
                ) from broken_json

        if isinstance(modelfile, io.TextIOWrapper):
            try:
                data = json.load(modelfile)

            except json.JSONDecodeError as broken_json:
                raise ValueError(
                    "Cannot load model from file: Broken JSON",
                    doc=broken_json.doc,
                    pos=broken_json.pos
                ) from broken_json

        self.load_model(data)


    def load_model(self, data: dict[str, dict]) -> None:
        """ LOAD MODEL: Loads an n-gram model from its dictionary format
            This will add the model's data into this model.
            Arguments:
                - data (dict[str, dict]): Model data in its dictionary format
            Returns: None
        """
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

            if self.param_n is None:
                self.param_n = tentative_n
            elif self.param_n != tentative_n:
                raise ValueError(
                    "Cannot load model: mismatch in n (Ngram context length)"
                )

            for token, count in vocab.items():
                self.vocab[token] += count

            for key, next_token in model.items():
                for token, count in next_token.items():
                    self.model[key][token] += count

        except KeyError as broken:
            raise ValueError(
                f"Cannot load model: Model JSON doesn't contain field {broken}"
            ) from broken

        except IndexError as empty_model:
            raise ValueError(
                "Cannot load model: Model appears to be empty"
            ) from empty_model


    def __call__(self, init_key: str|tuple[str]|list[str]) -> Self:
        """ __CALL__ (Overloads parentheses):
            Sets up the initial state of the autoregressor for subsequent
            iteration. Can be used in a for loop.
            Arguments:
                - init_key (str | list | tuple): Either a string containing
                    (N - 1) tokens (where N is the Ngram context length)
                    separated by spaces, or a list or tuple containing the
                    (N - 1) tokens as separate elements.
            Returns:
                - self: Returns this object itself (by reference)
        """
        if isinstance(init_key, str):
            self.state = init_key.split()
            if len(self.state) != self.param_n - 1:
                raise ValueError(
                    f"Cannot generate with starting phrase {self.state}: "
                    f"Number of tokens does not match (N - 1) for this model"
                )

        if isinstance(init_key, list):
            self.state = [*init_key]
            if len(self.state) != self.param_n - 1:
                raise ValueError(
                    f"Cannot generate with starting phrase {self.state}: "
                    f"Number of tokens does not match (N - 1) for this model"
                )

        if isinstance(init_key, tuple):
            self.state = list(init_key)
            if len(self.state) != self.param_n - 1:
                raise ValueError(
                    f"Cannot generate with starting phrase {self.state}: "
                    f"Number of tokens does not match (N - 1) for this model"
                )

        return self


    def __iter__(self) -> Self:
        """ Prepare this object for iteration and return it (by reference) """
        self.vocab_spreadout = []
        for token, count in self.vocab.items():
            self.vocab_spreadout.extend([token] * count)
        return self


    def __next__(self) -> str:
        """ Compute next token, update its state, and then return that token"""
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

        choice = random.choice(self.vocab_spreadout)
        self.state.pop(0)
        self.state.append(choice)
        return choice


    @property
    def data(self) -> dict[str, dict]:
        """ DATA (Property):
            A dictionary containing the vocabulary and parameters of the model
        """
        return {
            "vocab": dict(self.vocab),
            "model": {k: dict(v) for k, v in self.model.items()}
        }
