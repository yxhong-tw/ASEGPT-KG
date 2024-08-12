import re

from typing import (
    Dict,
    List,
    Union,
)


class Parser:

    def parse(
        self,
        data: List[Dict[str, Union[List[str], str]]],
    ) -> List[Dict[str, Union[List[str], str]]]:
        """ Parse the given data.

        Args:
            data (List[Dict[str, Union[List[str], str]]]): The data which will be parsed.

        Returns:
            List[Dict[str, Union[List[str], str]]]: The parsed data.
        """

        for one_data in data:
            triplets = re.findall(
                pattern=r"\"[^,\"]+, [^,\"]+, [^,\"]+\"",
                string=one_data["prediction"],
            )

            pred_triplets = []

            for triplet in triplets:
                pred_triplet = triplet.replace("\"", "").replace("\n", "")

                if pred_triplet != "" and pred_triplet != None:
                    pred_triplets.append(pred_triplet)

            if len(pred_triplets) != 0:
                one_data["pred_triplets"] = pred_triplets

        return data
