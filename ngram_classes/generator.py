""" Defines the NGramGenerator class, which runs autoregression inference """

from __future__ import annotations
from typing import Self

import io
import json
import random

class NGramGenerator:
    """ N-GRAM GENERATOR
    Class for running prediction and autoregression tasks
    Usage:
        generator = NGramGenerator()

        # Model loading method 1: from file path
        generator.load_file("path/to/trigram/model.json")

        # Model loading method 2: from file object (Text I/O Wrapper)
        with open("path/to/trigram/model.json") as modelfile:
            generator.load_file(modelfile)

        # Text prediction method 1: predict method (using one of three methods)
        next_token = generator.predict("token_1 token_2") # Using string
        next_token = generator.predict(["token_1", "token_2"]) # List
        next_token = generator.predict(("token_1", "token_2")) # Tuple

        # Text prediction method 2: Autoregression loop
        for token in generator(["token_1", "token_2"]): # use str, list or tuple
            print(token, end=" ")
            if terminating_condition():
                break
        print()
    """

    def __init__(self):
        """ Initializes the autoregressor with empty fields """
        self.param_n = None
        self.vocab = None
        self.model = None
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
            self.vocab = data["vocab"]
            self.model = data["model"]
            self.vocab_spreadout = []
            for token, count in self.vocab.items():
                self.vocab_spreadout.extend([token] * count)

            key_sizes = [
                len(keyphrase.split())
                for keyphrase in self.model.keys()
            ]
            unique_key_sizes = list(set(key_sizes))
            if len(unique_key_sizes) != 1:
                raise ValueError("Broken model file: non-uniform parameter N")
            self.param_n = unique_key_sizes[0] + 1

        except KeyError as missing_field:
            self.param_n, self.vocab, self.model = None, None, None
            raise KeyError(
                f"Broken model file: Missing field {missing_field}"
            ) from missing_field

        except ValueError as broken_model_file_error:
            self.param_n, self.vocab, self.model = None, None, None
            raise broken_model_file_error


    def predict(self, init_key: str|tuple[str]|list[str]) -> str:
        """ PREDICT
        Predicts the next token from a given number of tokens
        Arguments:
            - init_keys (str|list[str]|tuple[str]): (N - 1) starting tokens.
                May be a string of space-separated tokens, or a list or tuple.
        Returns:
            - prediction (str): Nth token
        """
        if any((
            self.vocab is None, self.param_n is None,
            self.model is None, self.vocab_spreadout is None
        )):
            raise ValueError("Cannot predict without loading a model!")

        if isinstance(init_key, str):
            init_key = init_key.split()

        if len(init_key) != self.param_n - 1:
            raise ValueError("Initial Phrase must have (N - 1) tokens")

        keyphrase = ' '.join(init_key)
        if keyphrase in self.model:
            spreadout = []
            for token, count in self.model[keyphrase].items():
                spreadout.extend([token] * count)
            return random.choice(spreadout)

        return random.choice(self.vocab_spreadout)


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

        if isinstance(init_key, list):
            self.state = [*init_key]

        if isinstance(init_key, tuple):
            self.state = list(init_key)

        if len(self.state) != self.param_n - 1:
            raise ValueError(
                f"Starting phrase {self.state} does not have (N -1) tokens"
            )

        return self


    def __iter__(self) -> Self:
        """ Prepare this object for iteration and return it (by reference) """
        return self


    def __next__(self) -> str:
        """ Compute next token, update its state, and then return that token"""
        if self.state is None:
            raise StopIteration

        keyphrase = ' '.join(self.state)
        next_token = self.predict(keyphrase)
        self.state.pop(0)
        self.state.append(next_token)
        return next_token
