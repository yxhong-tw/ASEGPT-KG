import faiss
import gensim

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from typing import (
    Dict,
    List,
    Union,
)


class Doc2Vec():

    def __init__(self) -> None:
        super(
            Doc2Vec,
            self,
        ).__init__()

        self.model = gensim.models.doc2vec.Doc2Vec.load(
            "/root/ASEGPT-KG/shared_space/models/pre-process/Doc2Vec_65K/checkpoint.model")
        self.simple_preprocess = gensim.utils.simple_preprocess
        self.faiss_index = faiss.IndexFlatL2(self.model.vector_size)

    def forward(
        self,
        documents: List[str],
    ) -> Dict[str, Union[List[np.ndarray], np.ndarray]]:
        """ Forward pass of the Doc2Vec model.

        Args:
            documents (List[str]): A list of documents.

        Returns:
            Dict[str, List[np.ndarray, np.ndarray]]: The output of the forward pass.
        """

        x = self.simple_preprocess(documents[0][0])

        x_vector = self.model.infer_vector(x).reshape(
            1,
            -1,
        )

        documents_embedding = x_vector
        _cosine_similarity = None

        try:
            y = self.simple_preprocess(documents[1][0])

            y_vector = self.model.infer_vector(y).reshape(
                1,
                -1,
            )

            documents_embedding = [
                x_vector,
                y_vector,
            ]

            _cosine_similarity = cosine_similarity(
                X=x_vector,
                Y=y_vector,
            )
        except:
            pass

        return {
            "documents_embedding": documents_embedding,
            "cosine_similarity": _cosine_similarity,
        }
