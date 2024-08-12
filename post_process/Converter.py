from opencc import OpenCC
from typing import (
    Dict,
    List,
    Union,
)


class Converter:

    def __init__(self):
        self.s2t_converter = OpenCC(config="s2t")
        self.t2s_converter = OpenCC(config="t2s")

    def _convert(
        self,
        text: str,
        type: str,
    ) -> str:
        """ Convert the given text to the target type.

        Args:
            text (str): The text which will be converted.
            type (str): The target type.

        Returns:
            str: The converted text.
        """

        if type == "s2t":
            return self.s2t_converter.convert(text=text)
        else:
            return self.t2s_converter.convert(text=text)

    def convert(
        self,
        data: Union[List[Dict[str, str]], str],
        type: str = "s2t",
        key_name: str = "pred_triplets",
    ) -> Union[List[Dict[str, str]], str]:
        """ Convert the given data to the given target type.

        Args:
            data (Union[List[Dict[str, str]], str]): The data which will be converted.
            type (str, optional): The target type. Defaults to "s2t".
            key_name (str, optional): The key name of the triplets. Defaults to "pred_triplets".

        Returns:
            Union[List[Dict[str, str]], str]: The converted data.
        """

        if isinstance(
                data,
                str,
        ):
            data = self._convert(
                text=data,
                type=type,
            )
        else:
            for one_data in data:
                try:
                    triplets = one_data[key_name]
                except KeyError:
                    pass

                triplets = [
                    self._convert(
                        text=triplet,
                        type=type,
                    ) for triplet in triplets
                ]

                one_data[key_name] = triplets

        return data
