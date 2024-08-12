import json

from typing import (
    Dict,
    List,
    Union,
)


def load_json(
    file_path: str
) -> List[Dict[str, Union[int, List[str], List[List[str]], str]]]:
    """ Load json file.

    Args:
        file_path (str): The input file path.

    Returns:
        List[Dict[str, Union[int, List[str], List[List[str]], str]]]: The loaded data.
    """

    data = None

    with open(
            file=file_path,
            mode="r",
            encoding="UTF-8",
    ) as f:
        data = json.load(fp=f)

        f.close()

    return data


def save_json(
    file_path: str,
    data: List[Dict[str, Union[float, int, List[str], str]]],
) -> None:
    """ Save json file.

    Args:
        file_path (str): The output file path.
        data (List[Dict[str, Union[float, int, List[str], str]]]): The data to save.
    """

    with open(
            file=file_path,
            mode="w",
            encoding="UTF-8",
    ) as f:
        json.dump(
            obj=data,
            fp=f,
            ensure_ascii=False,
            indent=4,
        )

        f.close()
