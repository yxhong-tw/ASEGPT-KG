import argparse
import time

from Converter import Converter
from Parser import Parser
from Regularizer import Regularizer
from utils import (
    filter_out_repeated_triplets,
    load_json,
    save_json,
)


def post_process(
    input_path: str,
    output_path: str,
) -> None:
    """ Do whole post-processing.

    Args:
        input_path (str): The input file path.
        output_path (str): The output file path.

    Raises:
        ValueError: The task is not supported.
    """

    data = load_json(file_path=input_path)

    tasks = [
        "Triplets Parsing",
        "Wikipedia Regularizing",
        "OpenCC Translating",
    ]

    for task in tasks:
        start_time = time.time()

        if task == "Triplets Parsing":
            parser = Parser()

            data = parser.parse(data=data)
        elif task == "Wikipedia Regularizing":
            wiki_regularizer = Regularizer()

            data = wiki_regularizer.regularize(data=data)
        elif task == "OpenCC Translating":
            opencc_converter = Converter()

            data = opencc_converter.convert(data=data)
        else:
            raise ValueError(f"The task {task} is not supported.")

        end_time = time.time()

        print(f"Time of {task}: {end_time - start_time}")

    # Filter out the repeated triplets
    data = filter_out_repeated_triplets(data=data)

    save_json(
        file_path=output_path,
        data=data,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-ip",
        "--input_path",
        type=str,
        required=True,
        help="The input file path.",
    )
    parser.add_argument(
        "-op",
        "--output_path",
        type=str,
        required=True,
        help="The output file path.",
    )

    args = parser.parse_args()

    post_process(
        input_path=args.input_path,
        output_path=args.output_path,
    )
