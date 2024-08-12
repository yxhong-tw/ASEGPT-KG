import torch

import numpy as np
import torch.nn as nn

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import (
    Dict,
    List,
)


class SBERT(nn.Module):

    def __init__(self) -> None:
        super(
            SBERT,
            self,
        ).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SentenceTransformer(
            model_name_or_path="uer/sbert-base-chinese-nli")

        self.model.to(device=device)

    def forward(
        self,
        sentences: List[str],
    ) -> Dict[str, np.ndarray]:
        """ The forward function.

        Args:
            sentences (List[str]): The input sentences.

        Returns:
            Dict[str, np.ndarray]: The output.
        """

        sentences_embedding = self.model.encode(sentences=sentences)

        _cosine_similarity = None

        try:
            _cosine_similarity = cosine_similarity(
                X=[sentences_embedding[0]],
                Y=[sentences_embedding[1]],
            )
        except:
            pass

        return {
            "sentences_embedding": sentences_embedding,
            "cosine_similarity": _cosine_similarity,
        }
