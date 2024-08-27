"""
Article (Data) Merge

The code:
1. Split all data into chunks.
2. Get the data embedding using `do_pooling()` with each data chunks.
3. Support `Doc2Vec`, `SBERT`, `BGE`, and `BGE-M3` models.
4. Calculate the similarity with FAISS.
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
    use_local_lm: bool = True,
) -> None:
    """ Merge the given data.

    Args:
        data (List[Dict[str, Union[int, List[str], List[List[str]], str]]]): The data need to be processed.
        data_processor (DataProcessor): The data processor.
        model (Union[Doc2Vec, FlagModel, SBERT]): The language model which is used to get the sentence embedding.
        index (faiss.IndexFlatIP): The FAISS index.
        params (Dict[str, Union[float, int, str]]): The parameters.
        use_local_lm (bool, optional): Whether to use the local language model to merge articles. Defaults to True.
    """

    all_data = []

    for i in range(len(data)):
        print(f"Processing {i}")

        data[i]["raw_article_content"] = []

        string_chunks = data_processor.get_string_chunks(
            text=data[i]["article_content"])

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

        pooled_embedding = data_processor.do_pooling(
            embedding_chunks=[
                item["embedding_chunk"] for item in one_data_chunks
            ],
            pooling_type="mean",
        ).astype("float32")

        all_data.append({
            "article": data[i]["article_content"],
            "pooled_embedding": pooled_embedding,
            "data_index": i,
        })

        faiss.normalize_L2(pooled_embedding)

        index.add(pooled_embedding)

    over_threshold_data = []

    for i in range(len(all_data)):
        print(f"Processing {i}")

        faiss.normalize_L2(x=all_data[i]["pooled_embedding"])

        distances, indices = index.search(
            all_data[i]["pooled_embedding"],
            2,
        )

        search_results = [(
            index,
            float(distance),
        ) for distance, index in zip(
            distances[0],
            indices[0],
        )]

        if search_results[0][0] == i:
            if search_results[1][1] > params["similarity_threshold"]:
                over_threshold_data.append({
                    "base_data": [all_data[i]["article"]],
                    "base_data_index":
                    all_data[i]["data_index"],
                    "match_data": [all_data[search_results[1][0]]["article"]],
                    "match_data_index":
                    all_data[search_results[1][0]]["data_index"],
                    "similarity":
                    search_results[1][1],
                })
        else:
            if search_results[0][1] > params["similarity_threshold"]:
                over_threshold_data.append({
                    "base_data": [all_data[i]["article"]],
                    "base_data_index": all_data[i]["data_index"],
                    "match_data": [all_data[search_results[0][0]]["article"]],
                    "match_data_index": all_data[search_results[0][0]]["data_index"],
                    "similarity": search_results[0][1],
                })

    gc.collect()
    torch.cuda.empty_cache()

    data_merger = DataMerger(
        data=over_threshold_data,
        use_local_lm=use_local_lm,
        is_chunk=False,
    )

    merged_data = data_merger.merge()

    output_data = []

    for data_idx, one_data in enumerate(data):
        break_flag = False

        for one_merged_data in merged_data:
            if data_idx == one_merged_data["base_data_index"] or \
                data_idx == one_merged_data["match_data_index"]:

                break_flag = True

                break

        if not break_flag:
            output_data.append(one_data)

    for one_merged_data in merged_data:
        i = one_merged_data["base_data_index"]
        j = one_merged_data["match_data_index"]

        output_data.append({
            "crawl_datetime": \
                data[i]["crawl_datetime"] + "," + data[j]["crawl_datetime"],
            "source_name": \
                data[i]["source_name"] + "," + data[j]["source_name"],
            "source_category": \
                data[i]["source_category"] + "," + data[j]["source_category"],
            "article_url": \
                data[i]["article_url"] + "," + data[j]["article_url"],
            "article_title": \
                data[i]["article_title"] + "," + data[j]["article_title"],
            "article_author": \
                data[i]["article_author"] + "," + data[j]["article_author"],
            "article_content": one_merged_data["merged_data"],
            "article_creation_date": \
                data[i]["article_creation_date"] + "," + data[j]["article_creation_date"],
            "topic_id": str(data[i]["topic_id"]) + "," + str(data[j]["topic_id"]),
            "raw_article_content": [data[i]["article_content"], data[j]["article_content"]],
        })

    save_json(
        file_path=params["output_path"],
        data=output_data,
    )

    print(
        f"The data number of the similarity over threshold ({params['similarity_threshold']}): {len(over_threshold_data)}"
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
        "-tid",
        "--topic_id",
        type=int,
        help="The topic id.",
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
            model_name_or_path="/root/ASEGPT-KG/shared_space/models/pre-process/bge-large-zh-v1.5",
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
        index = faiss.IndexFlatIP(1024)
    elif args.model == "BGE-M3":
        model = FlagModel(
            model_name_or_path="/root/ASEGPT-KG/shared_space/models/pre-process/bge-m3",
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

    if args.topic_id is None:
        params = {
            "model_name": args.model,
            "data_name": args.data_name,
            "topic_id": args.topic_id,
            "similarity_threshold": args.similarity_threshold,
            "output_path": args.output_path,
        }

        process(
            data=data,
            data_processor=data_processor,
            model=model,
            index=index,
            params=params,
            use_local_lm=args.use_local_lm,
        )
    else:
        all_topic_data = {}

        for one_data in data:
            if not one_data["topic_id"] in all_topic_data.keys():
                all_topic_data[one_data["topic_id"]] = []

            all_topic_data[one_data["topic_id"]].append(one_data)

        for topic_id, topic_data in all_topic_data.items():
            if topic_id != args.topic_id:
                continue

            params = {
                "model_name": args.model,
                "data_name": args.data_name,
                "topic_id": args.topic_id,
                "similarity_threshold": args.similarity_threshold,
                "output_path": args.output_path,
            }

            process(
                data=topic_data,
                data_processor=data_processor,
                model=model,
                index=index,
                params=params,
                use_local_lm=args.use_local_lm,
            )
