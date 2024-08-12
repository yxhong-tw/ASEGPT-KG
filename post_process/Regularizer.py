import json
import re
import wikipedia

import numpy as np

from multiprocessing import (
    cpu_count,
    Pool,
)
from tqdm import tqdm
from typing import (
    Dict,
    List,
    Tuple,
    Union,
)

from Converter import Converter


class Regularizer:

    def __init__(self):
        wikipedia.set_lang(prefix="zh")

        self.cache = {}
        self.threshold_1st = 0.9
        self.threshold_2nd = 0.8

    def _regularize(
        self,
        candidate_entity: str,
    ) -> str:
        """ Regularize the given candidate entity.

        Args:
            candidate_entity (str): The candidate entity which will be regularized.

        Returns:
            str: The regularized candidate entity.
        """

        converter = Converter()

        candidate_entity = converter._convert(
            text=candidate_entity,
            type="s2t",
        )

        if candidate_entity in self.cache:
            return self.cache[candidate_entity]["title"]

        part_ce = candidate_entity[:300] if len(
            candidate_entity) > 300 else candidate_entity

        titles = wikipedia.search(
            query=part_ce,
            results=5,
        )

        if len(titles) == 0:
            return None
        else:
            for title in titles:
                title = converter._convert(
                    text=title,
                    type="s2t",
                )

        if candidate_entity in titles:
            part_ce = candidate_entity[:300] if len(
                candidate_entity) > 300 else candidate_entity

            try:
                page = wikipedia.page(
                    title=part_ce,
                    auto_suggest=False,
                )

                one_entity_data = {
                    "title": converter._convert(
                        text=page.title,
                        type="s2t",
                    ),
                    "url": page.url,
                    "summary": converter._convert(
                        text=page.summary,
                        type="s2t",
                    ),
                }

                self.cache[part_ce] = one_entity_data
                self.cache[one_entity_data["title"]] = one_entity_data

                return one_entity_data["title"]
            except:
                return None

        entity_data = []

        for title in titles:
            try:
                page = wikipedia.page(
                    title=title,
                    auto_suggest=False,
                )

                entity_data.append({
                    "title":
                    converter._convert(
                        text=page.title,
                        type="s2t",
                    ),
                    "url":
                    page.url,
                    "summary":
                    converter._convert(
                        text=page.summary,
                        type="s2t",
                    ),
                })
            except wikipedia.DisambiguationError:
                continue

        candidate_entity_nums = []
        lns = []

        for one_entity_data in entity_data:
            candidate_entity_num = len(
                re.findall(
                    pattern=self.escape_special_characters(
                        pattern=candidate_entity),
                    string=one_entity_data["summary"],
                ))

            ln, _ = self.longest_common_subsequence(
                x=candidate_entity,
                y=one_entity_data["summary"],
            )

            candidate_entity_nums.append(candidate_entity_num)
            lns.append(ln)

        lcs_rate = np.max(a=lns) / len(candidate_entity)
        index = -1

        if lcs_rate >= self.threshold_1st:
            index = np.argmax(a=lns)
        elif lcs_rate >= self.threshold_2nd:
            index = np.argmax(a=candidate_entity_nums)
        else:
            return None

        self.cache[candidate_entity] = entity_data[index]
        self.cache[entity_data[index]["title"]] = entity_data[index]

        return entity_data[index]["title"]

    def regularize(
        self,
        data: List[Dict[str, Union[List[str], str]]],
    ) -> List[Dict[str, Union[List[str], str]]]:
        """ Regularize the given data.

        Args:
            data (List[Dict[str, Union[List[str], str]]]): The data which will be regularized.

        Returns:
            List[Dict[str, Union[List[str], str]]]: The regularized data.
        """

        for (
                index,
                one_data,
        ) in enumerate(tqdm(data)):
            try:
                triplets = one_data["pred_triplets"]

                with Pool(cpu_count()) as pool:
                    regularized_triplets = list(
                        pool.imap(
                            func=self.process_triplet,
                            iterable=triplets,
                        ))

                one_data["processed_pred_triplets"] = regularized_triplets
            except Exception as e:
                one_data["processed_pred_triplets"] = one_data["pred_triplets"]

                print(f"Error at index {index}.")
                print(e)

            if (index % 100 == 0 and index != 0) or index == len(data) - 1:
                with open(
                        file=".regularizer_cache.json",
                        mode="w",
                        encoding="UTF-8",
                ) as f:
                    json.dump(
                        obj=self.cache,
                        fp=f,
                        indent=4,
                        ensure_ascii=False,
                    )

        return data

    def escape_special_characters(self, pattern: str) -> str:
        """ Escape special characters in the given pattern.

        Args:
            pattern (str): The pattern which will be processed.

        Returns:
            str: The processed pattern.
        """

        special_chars = [
            "+",
            "*",
            ".",
            "?",
            "\\",
            "^",
            "$",
            "|",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
        ]
        escaped_pattern = ""

        for char in pattern:
            if char in special_chars:
                escaped_pattern += "\\" + char
            else:
                escaped_pattern += char

        return escaped_pattern

    def longest_common_subsequence(
        self,
        x: str,
        y: str,
    ) -> Tuple[int, str]:
        """ Find the longest common subsequence between two given strings. Also returns the subsequence found.

        Args:
            x (str): The first string.
            y (str): The second string.

        Returns:
            Tuple[int, str]: The length of the longest subsequence and the subsequence found.
        """

        assert x is not None
        assert y is not None

        m = len(x)
        n = len(y)

        # Declare the array for storing the dp values.
        l = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(
                1,
                m + 1,
        ):
            for j in range(
                    1,
                    n + 1,
            ):
                match = 1 if x[i - 1] == y[j - 1] else 0

                l[i][j] = max(
                    l[i - 1][j],
                    l[i][j - 1],
                    l[i - 1][j - 1] + match,
                )

        i = m
        j = n
        seq = ""

        while i > 0 and j > 0:
            match = 1 if x[i - 1] == y[j - 1] else 0

            if l[i][j] == l[i - 1][j - 1] + match:
                if match == 1:
                    seq = x[i - 1] + seq

                i -= 1
                j -= 1
            elif l[i][j] == l[i - 1][j]:
                i -= 1
            else:
                j -= 1

        return (
            l[m][n],
            seq,
        )

    def process_triplet(
        self,
        triplet: str,
    ) -> str:
        """ Process the head and tail entities of every triplet.

        Args:
            triplet (str): The triplet which will be processed.

        Returns:
            str: The processed triplet.
        """

        if len(triplet.split(sep=", ")) != 3:
            return None

        h, r, t = triplet.split(sep=", ")

        regularized_h = self._regularize(candidate_entity=h)
        regularized_t = self._regularize(candidate_entity=t)

        regularized_h = regularized_h if regularized_h is not None else h
        regularized_t = regularized_t if regularized_t is not None else t

        return f"{regularized_h}, {r}, {regularized_t}"
