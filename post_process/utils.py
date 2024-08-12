import json

from typing import (
    Dict,
    List,
    Union,
)


def filter_out_repeated_triplets(
    data: List[Dict[str, Union[List[str], str]]],
    key_name: str = "processed_pred_triplets",
) -> List[Dict[str, Union[List[str], str]]]:
    """ Filter out the repeated triplets.

    Args:
        data (List[Dict[str, Union[List[str], str]]]): The data which will be processed.
        key_name (str, optional): The key name of the triplets. Defaults to "processed_pred_triplets".

    Returns:
        List[Dict[str, Union[List[str], str]]]: The data after filtering out the repeated triplets.
    """

    for one_data in data:
        triplets = one_data[key_name]

        triplets = set(triplets)
        triplets = list(triplets)

        one_data[key_name] = triplets

    return data


def load_json(file_path: str) -> List[Dict[str, Union[List[str], str]]]:
    """ Load json file.

    Args:
        file_path (str): The path of the file which will be loaded.

    Returns:
        List[Dict[str, Union[List[str], str]]]: The loaded data.
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
    data: List[Dict[str, Union[List[str], str]]],
) -> None:
    """ Save data to json file.

    Args:
        file_path (str): The path of the file which will be saved.
        data (List[Dict[str, Union[List[str], str]]]): The data which will be saved.
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
