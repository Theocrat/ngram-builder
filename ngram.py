import os
import sys

from pathlib import Path
from argparse import ArgumentParser, Namespace

from ngram_classes.builder import NGramBuilder
from ngram_classes.generator import NGramGenerator

error_code = Namespace(
    no_command = 1,
    no_target = 2,
    model_does_not_exist = 3
)

data_path = Path("models")

parser = ArgumentParser(
    description="Command Line n-grams training and autoregression"
)

parser.add_argument('command', type=str)
parser.add_argument('--name', type=str, required=False)
parser.add_argument('--source', type=str, required=False)
parser.add_argument('--length', type=int, required=False)
parser.add_argument('--start', type=str, required=False)
parser.add_argument('--path', type=str, required=False)

args = parser.parse_args()
if args.path:
    data_path = Path(args.path)

data_path.mkdir(exist_ok=True)

def model_names():
    return [
        content.name.rstrip(".json") 
        for content in data_path.iterdir()
        if content.suffix == ".json"
    ]

match args.command:
    case 'list':
        print(*model_names(), sep="\n")

    case 'delete':
        if args.name is None:
            print("Usage: python ngram.py delete --name <model-name>", file=sys.stderr)
            exit(error_code.no_target)

        target = data_path / f"{args.name}.json"

        if not target.exists():
            print("No such model:", args.name, file=sys.stderr)
            print(
                "Use the list command to check the available models first", 
                file=sys.stderr
            )
            exit(error_code.model_does_not_exist)

        target.unlink()

    case 'train':
        print("Not yet implemented")

    case 'tune':
        print("Not yet implemented")

    case 'generate':
        print("Not yet implemented")

    case _:
        print("No such command:", args.command, file=sys.stderr)
        print("Available commands are:", file=sys.stderr)
        print("- General Commands:", file=sys.stderr)
        print("  - list", file=sys.stderr)
        print("  - delete", file=sys.stderr)
        print("- Training and Tuning:", file=sys.stderr)
        print("  - train", file=sys.stderr)
        print("  - tune", file=sys.stderr)
        print("- Generate Text:", file=sys.stderr)
        print("  - generate", file=sys.stderr)
        exit(error_code.no_command)