# N-gram Builder

Large Language Models are all the rave these days. But the best way to see how 
language models work is to look at simpler, smaller models. This repository 
contains simple object-oriented wrappers for building the simplest type of 
language model there is: the N-gram model.

## Objectives

The primary objective of this project is to provide a transparent explanation of
how N-gram models work - and thereby illustrate the principles that help 
language models in general - by using a Python example.

This model comes with a command line tool, which can be used to train models and
generate content using them. It also implements two classes in separate Python
files: one for building and storing models, and one for generating ngram output
using a previously stored model.

### Behaviour

This model is case-insensitive. It treats spaces and newlines as delimiters for 
tokenization, but also treats each punctuation as a separate token. So the 
following sentences:

> Today is a nice day. I feel fine. How do you feel, my friend?

generates the following sequence of tokens:
```json
[
    "today",   "is",      "a",       "nice",    "day",     ".",
    "i",       "feel",    "fine",    ".",       "how",     "do",
    "you",     "feel",    ",",       "my",      "friend",  "?"
]
```

## Usage

### Command Line Tool

The command line tool `ngram` can be used in three ways: to list or delete
existing modes, to train a new model or tune an existing one, or to generate 
text using an existing model.

#### Listing and Deleting Models

To list existing models, invoke ngram with the `list` argument.
```sh
ngram list
```

To delete a model that was trained before, use the `delete` argument.
```sh
ngram delete model-name
```

#### Training a New Model or Tuning an Existing Model

To train a new model, invoke ngram with the `train` command. You will need to 
supply a name for the model, a source material, and the value of N (eg. 3 for a 
trigram model).
```sh
ngram train --name austen --source "Pride and Prejudice.txt" --n 3
```

To tune an existing model, use the `tune` command with the same syntax as above.
```sh
ngram tune --name austen --source Emma.txt
```

__NOTE:__ The source material needs to be a text file. PDF documents don't work.

#### Generating Text using an Existing Model

To generate text by using the model as an auto-regressor, invoke the model using
the `generate` command. You will need to provide three arguments: model name,
generated sequence length in number of tokens (words or punctuations), and a 
sequence of starting tokens.
```sh
ngram generate --name austen --length 100 --start ". the"
```