from __future__ import annotations

import io
import re
import json
import string

from copy import deepcopy
from collections import defaultdict
from unidecode import unidecode

class NGramBuilder:
    """ Class for training and tuning Ngram models """

    def __init__(self, n: int):
        """ Initializes an Ngram model using a value of n """
        self.n = n
        self.vocab = defaultdict(int)
        self.model = defaultdict(lambda: defaultdict(int))

    
    def add_source(self, text: str) -> None:
        """ ADD SOURCE: Trains the model on a source text.
            Arguments:
                - text (str): A string, acting as a document for training
            Returns: None
        """
        tokens = NGramBuilder.generate_tokens(text)
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


    def add_from_file(self, sourcefile: str|io.TextIOWrapper) -> None:
        """ ADD FROM FILE: Trains the model on the contents of a source file.
            Arguments:
                - sourcefile (str | io.TextIOWraper): File for training, which
                    should contain the text for training the file. Either the 
                    path to the fike should be provided as a string, or the file
                    object should be provided.
            Returns: None
        """
        if isinstance(sourcefile, str):
            with open(sourcefile) as wrapper:
                source = wrapper.read()
        
        elif isinstance(sourcefile, io.TextIOWrapper):
            try:
                source = sourcefile.read()

            except ValueError as ve:
                raise ValueError(f"Cannot add source from file: {str(ve)}")
            
        self.add_source(source)

    
    def save(self, modelfile: str|io.TextIOWrapper) -> None:
        """ SAVE: Saves the model into a JSON file
            Arguments:
                - modelfile (str | io.TextIOWrapper): A file to save the model
                    into; should either be the path to a JSON file, or a file
                    object (in write mode).
            Returns: None
        """
        if isinstance(modelfile, str):
            with open(modelfile, "w") as file_object:
                json.dump(self.data, file_object, indent=2)

        elif isinstance(modelfile, io.TextIOWrapper):
            try:
                json.dump(self.data, modelfile, indent=2)
            except ValueError:
                raise ValueError(
                    f"Cannot save: '{modelfile.name}' is either closed or open"
                    "in the read-only mode."
                )
        
        else:
            raise ValueError(
                "Argument to save has to be file object or path to file (str)."
            )


    def copy(self) -> NGramBuilder:
        """ COPY: Creates a deep copy of this object.
            Arguments: None
            Returns:
                - duplicate (NGramBuilder): Another object of this class, with 
                    the same values of the `n`, `model`, and `vocab` fields.
        """
        duplicate = NGramBuilder(n=self.n)
        duplicate.vocab = deepcopy(self.vocab)
        duplicate.model = deepcopy(self.model)
        return duplicate
    
    
    def __add__(self, other_builder) -> NGramBuilder:
        """ __ADD__ (Operator +)
            Combines two models, generating a new model with the same parameters
            as would have resulted from training a model on the combined 
            training data of both models.
            Usage:
                result = builder_1 + builder_2
            Returns:
                result (NGramBuilder): A new model which has learned the
                combined learning data of both the operand models.
        """
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
    def generate_tokens(text) -> list[str]:
        """ GENERATE TOKENS: Static helper method for tokenizing text
        Arguments:
            - text (str): A document to tokenize
        Returns:
            - tokens (list[str]): A list of token strings (words or punctuation)
        """
        precleaned = unidecode(text).lower()
        punctspace = re.sub(f"([{string.punctuation}])", r" \g<0> ", precleaned)
        ellipsis_restored = punctspace.replace(". . .", "...")
        quote_t_restored = ellipsis_restored.replace(" 't", "'t")
        single_spaces = re.sub(r"\s+", " ", quote_t_restored)
        return single_spaces.split()


    @property
    def data(self) -> dict[str, dict]:
        """ DATA (Property): 
            A dictionary containing the vocabulary and parameters of the model
        """
        return {
            "vocab": dict(self.vocab),
            "model": {k: dict(v) for k, v in self.model.items()}
        }