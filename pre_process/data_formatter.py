import argparse

from utils import (
    load_json,
    save_json,
)


def format_data(
    input_file: str,
    output_file: str,
):
    """ Format the data to the required format.

    Args:
        input_file (str): The input file path.
        output_file (str): The output file path.
    """

    data = load_json(file_path=input_file)

    output_data = []

    for one_data in data:
        _one_data = {}

        for key, value in one_data.items():
            if key == "article_content":
                _one_data["merge_content"] = value

            if key != "topic":
                _one_data[key] = [value]
            else:
                _one_data[key] = value

        output_data.append(_one_data)

    save_json(
        file_path=output_file,
        data=output_data,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-if",
        "--input_file",
        type=str,
        required=True,
        help="The input file path.",
    )
    parser.add_argument(
        "-of",
        "--output_file",
        type=str,
        required=True,
        help="The output file path.",
    )

    args = parser.parse_args()

    format_data(
        input_file=args.input_file,
        output_file=args.output_file,
    )
