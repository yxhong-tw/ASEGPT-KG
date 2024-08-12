import argparse
import re

from typing import List

from utils import (
    load_json,
    save_json,
)


def parse_triplets(input_string: str) -> List[List[str]]:
    """ Parse the triplets from the given string.

    Args:
        input_string (str): The input triplet string.

    Returns:
        List[List[str]]: The parsed triplets.
    """

    parts = re.split(
        pattern=r" (?=-\[)|(?<=\]->)|(?=<-\[)|(?<=\]-) ",
        string=input_string,
    )

    triplets = []

    for i in range(
            1,
            len(parts),
            2,
    ):
        relation_match = re.search(
            pattern=r"relationship: (.*?)\}",
            string=parts[i],
        )
        relation = relation_match.group(1) if relation_match else None

        if "->" in parts[i]:
            subject_match = re.search(
                pattern=r"{name: (.*?)}",
                string=parts[i - 1],
            )
            object_match = re.search(
                pattern=r"{name: (.*?)}",
                string=parts[i + 1],
            )
        elif "<-" in parts[i]:
            subject_match = re.search(
                pattern=r"{name: (.*?)}",
                string=parts[i + 1],
            )
            object_match = re.search(
                pattern=r"{name: (.*?)}",
                string=parts[i - 1],
            )

        subject = subject_match.group(1) if subject_match else None
        obj = object_match.group(1) if object_match else None

        if subject and relation and obj:
            triplets.append([
                subject,
                relation,
                obj,
            ])

    return triplets


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

    data = load_json(file_path=args.input_path)

    for one_data in data:
        triplets = parse_triplets(one_data["prediction"])
        one_data["pred_triplets"] = triplets

    save_json(
        file_path=args.output_path,
        data=data,
    )
