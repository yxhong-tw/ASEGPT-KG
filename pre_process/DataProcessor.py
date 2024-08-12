import numpy as np

from typing import List


class DataProcessor():

    def __init__(self) -> None:
        super(
            DataProcessor,
            self,
        ).__init__()

    def do_pooling(
        self,
        embedding_chunks: List[np.ndarray],
        pooling_type: str,
    ) -> List[np.ndarray]:
        """ Perform pooling on a list of embedding chunks.

        Args:
            embedding_chunks (List[np.ndarray]): The list of embedding chunks.
            pooling_type (str): The type of pooling to be performed.

        Raises:
            ValueError: The pooling type is not supported.

        Returns:
            List[np.ndarray]: The pooled embedding.
        """

        pooled_embedding = np.stack(
            arrays=[embedding_chunk for embedding_chunk in embedding_chunks],
            axis=0,
        )

        if pooling_type == "max":
            pooled_embedding = np.max(
                a=pooled_embedding,
                axis=0,
            )
        elif pooling_type == "mean":
            pooled_embedding = np.mean(
                a=pooled_embedding,
                axis=0,
            )
        else:
            raise ValueError("Pooling type is not supported.")

        return pooled_embedding

    def get_string_chunks(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 32,
    ) -> List[str]:
        """ Get chunks from the given string.

        Args:
            text (str): The text to be chunked.
            chunk_size (int, optional): The size of each chunk. Defaults to 512.
            overlap (int, optional): The overlap between chunks. Defaults to 32.

        Returns:
            List[str]: A list of chunks.
        """

        current_index = 0
        string_chunks = []

        while True:
            if current_index + chunk_size < len(text):
                string_chunks.append(text[current_index:current_index +
                                          chunk_size])

                current_index += chunk_size - overlap
            else:
                string_chunks.append(text[current_index:])

                break

        return string_chunks
