"""
Paragrath (Chunk) Merge

The code:
1. Split all data into chunks and concatenate them.
2. Compare the similarity between all chunks.
3. Support `Doc2Vec`, `SBERT`, `BGE`, and `BGE-M3` models.
4. Do NOT call `do_pooling()` while calculating the chunk embedding.
5. Calculate the similarity with FAISS.
"""

import argparse
import faiss
import gc
import os
import torch

import numpy as np

from FlagEmbedding import FlagModel
from typing import (
    Dict,
    List,
    Union,
)

from DataMerger import DataMerger
from DataProcessor import DataProcessor
from Doc2Vec import Doc2Vec
from SBERT import SBERT
from utils import (
    load_json,
    save_json,
)


def process(
    data: List[Dict[str, Union[int, List[str], List[List[str]], str]]],
    data_processor: DataProcessor,
    model: Union[Doc2Vec, FlagModel, SBERT],
    index: faiss.IndexFlatIP,
    params: Dict[str, Union[float, int, str]],
    is_topic_data: bool = False,
    use_local_lm: bool = True,
) -> None:
    """ Merge the given data.

    Args:
        data (List[Dict[str, Union[int, List[str], List[List[str]], str]]]): The data need to be processed.
        data_processor (DataProcessor): The data processor.
        model (Union[Doc2Vec, FlagModel, SBERT]): The language model which is used to get the sentence embedding.
        index (faiss.IndexFlatIP): The FAISS index.
        params (Dict[str, Union[float, int, str]]): The parameters.
        is_topic_data (bool, optional): Whether the given data is topic data. Defaults to False.
        use_local_lm (bool, optional): Whether to use the local language model. Defaults to True.
    """

    all_chunks = []

    for i in range(len(data)):
        print(f"Processing {i}")

        data_key = "merge_content" if is_topic_data else "article_content"

        string_chunks = data_processor.get_string_chunks(
            text=data[i][data_key])

        one_data_chunks = []

        for j, string_chunk in enumerate(string_chunks):
            if params["model_name"] == "Doc2Vec":
                embedding_chunk = model.forward(
                    documents=[string_chunk])["documents_embedding"]
            elif params["model_name"] == "SBERT":
                embedding_chunk = model(
                    sentences=[string_chunk])["sentences_embedding"]
            elif params["model_name"] == "BGE":
                embedding_chunk = model.encode(string_chunk)

                embedding_chunk = np.array([embedding_chunk])
            elif params["model_name"] == "BGE-M3":
                embedding_chunk = model.encode(string_chunk)

                embedding_chunk = np.array([embedding_chunk])

            one_data_chunks.append({
                "string_chunk":
                string_chunk,
                "embedding_chunk":
                embedding_chunk.astype("float32"),
                "data_index":
                i,
                "chunk_index":
                j,
            })

        for one_chunk in one_data_chunks:
            one_chunk["all_index"] = len(all_chunks)

            all_chunks.append(one_chunk)

            faiss.normalize_L2(one_chunk["embedding_chunk"])

            index.add(one_chunk["embedding_chunk"])

    over_threshold_chunks = []

    for i in range(len(all_chunks)):
        print(f"Processing {i}")

        faiss.normalize_L2(x=all_chunks[i]["embedding_chunk"])

        distances, indices = index.search(
            all_chunks[i]["embedding_chunk"],
            len(all_chunks),
        )

        search_results = [(
            index,
            float(distance),
        ) for distance, index in zip(
            distances[0],
            indices[0],
        )]

        for one_search_result in search_results:
            if one_search_result[1] > params["similarity_threshold"]:
                if one_search_result[0] != all_chunks[i]["all_index"]:
                    over_threshold_chunks.append({
                        "base_string_chunk": [all_chunks[i]["string_chunk"]],
                        "base_data_index":
                        all_chunks[i]["data_index"],
                        "base_chunk_index":
                        all_chunks[i]["chunk_index"],
                        "base_all_index":
                        all_chunks[i]["all_index"],
                        "match_string_chunk":
                        [all_chunks[one_search_result[0]]["string_chunk"]],
                        "match_data_index":
                        all_chunks[one_search_result[0]]["data_index"],
                        "match_chunk_index":
                        all_chunks[one_search_result[0]]["chunk_index"],
                        "match_all_index":
                        all_chunks[one_search_result[0]]["all_index"],
                        "similarity":
                        one_search_result[1],
                    })

    gc.collect()
    torch.cuda.empty_cache()

    data_merger = DataMerger(
        data=over_threshold_chunks,
        use_local_lm=use_local_lm,
        is_chunk=True,
    )

    merged_data = data_merger.merge()

    save_json(
        file_path=params["output_path"],
        data=merged_data,
    )

    print(
        f"The chunk number of the similarity over threshold ({params['similarity_threshold']}): {len(over_threshold_chunks)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=[
            "Doc2Vec",
            "SBERT",
            "BGE",
            "BGE-M3",
        ],
        required=True,
        help="The model to use.",
    )
    parser.add_argument(
        "-dd",
        "--data_dir",
        type=str,
        required=True,
        help="The data directory.",
    )
    parser.add_argument(
        "-dn",
        "--data_name",
        type=str,
        required=True,
        help="The data name.",
    )
    parser.add_argument(
        "-t",
        "--topic",
        type=int,
        help="The topic number.",
    )
    parser.add_argument(
        "-op",
        "--output_path",
        type=str,
        required=True,
        help="The output path.",
    )
    parser.add_argument(
        "-st",
        "--similarity_threshold",
        type=float,
        default=0.875,
        help="The similarity threshold.",
    )
    parser.add_argument(
        "-ullm",
        "--use_local_lm",
        action="store_true",
        help="Whether to use the local language model.",
    )

    args = parser.parse_args()

    model = None
    index = None

    if args.model == "Doc2Vec":
        model = Doc2Vec()
        index = faiss.IndexFlatIP(100)
    elif args.model == "SBERT":
        model = SBERT()
        index = faiss.IndexFlatIP(768)
    elif args.model == "BGE":
        model = FlagModel(
            model_name_or_path="BAAI/bge-large-zh-v1.5",
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
        index = faiss.IndexFlatIP(1024)
    elif args.model == "BGE-M3":
        model = FlagModel(
            model_name_or_path="BAAI/bge-m3",
            query_instruction_for_retrieval=
            "Generate a representation for this sentence for retrieving related articles:"
        )
        index = faiss.IndexFlatIP(1024)
    else:
        raise ValueError("The model is not supported.")

    data_processor = DataProcessor()

    data_dir = args.data_dir
    data_name = args.data_name

    data_path = os.path.join(
        data_dir,
        f"{data_name}.json",
    )

    data = load_json(file_path=data_path)

    print(f"Data {data_name} loaded successfully!")

    if args.topic is None:
        params = {
            "model_name": args.model,
            "data_name": args.data_name,
            "topic": args.topic,
            "similarity_threshold": args.similarity_threshold,
            "output_path": args.output_path,
        }

        process(
            data=data,
            data_processor=data_processor,
            model=model,
            index=index,
            params=params,
            is_topic_data=False,
            use_local_lm=args.use_local_lm,
        )
    else:
        all_topic_data = {}

        for one_data in data:
            if not one_data["topic"] in all_topic_data.keys():
                all_topic_data[one_data["topic"]] = []

            all_topic_data[one_data["topic"]].append(one_data)

        for topic, topic_data in all_topic_data.items():
            if topic != args.topic:
                continue

            params = {
                "model_name": args.model,
                "data_name": args.data_name,
                "topic": args.topic,
                "similarity_threshold": args.similarity_threshold,
                "output_path": args.output_path,
            }

            process(
                data=topic_data,
                data_processor=data_processor,
                model=model,
                index=index,
                params=params,
                is_topic_data=True,
                use_local_lm=args.use_local_lm,
            )
