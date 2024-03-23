import argparse
import json
from typing import List

import pandas as pd
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMMultiSelector

from llama_index_server import (
    connect_to_nebula_graph,
    convert_to_query_engine_tool,
    load_engine,
    load_multi_doc_index,
    load_multi_docs,
    load_storage,
    load_store,
    set_global_service,
)


def generate_response(questions_df: pd.DataFrame,
                      query_engine: RouterQueryEngine):
    results = []
    for i, row in questions_df.iterrows():
        question = row['question']
        print(f'Question: {question}')

        response = query_engine.query(question)
        print(f'Response: {response}')

        result = {
            'system_prompt': None,
            'question': question,
            'response': response.response
        }
        results.append(result)


def initialize_rag_settings(doc_paths: List[str], space_names: List[str],
                            persist_dirs: List[str]):
    assert len(space_names) == len(
        persist_dirs
    ), 'Length of space_names and persist_dirs should be the same.'

    is_connected = connect_to_nebula_graph()

    documents = load_multi_docs(doc_paths)

    service_context = set_global_service(using_openai_gpt=True,
                                         chunk_size=4096)
    stores = [load_store(space_name) for space_name in space_names]
    storages = [
        load_storage(store=s, persist_dir=p)
        for s, p in zip(stores, persist_dirs)
    ]

    indices = load_multi_doc_index(documents=documents,
                                   storage_contexts=storages,
                                   space_names=space_names)
    engines = [
        load_engine(i,
                    mode='custom',
                    documents=d,
                    service_context=service_context)
        for i, d in zip(indices, documents)
    ]

    engine_tools = convert_to_query_engine_tool(
        engines,
        names=[
            'war_query_engine_tool, silicon_query_engine_tool, ai_query_engine_tool',
            'semiconductor_query_engine_tool'
        ],
        descriptions=[
            'Useful for answering questions about "China–United States trade war" and its impact on the world',
            'Useful for answering questions about "Silicon Photonics development trend" and its impact on the world',
            'Useful for answering questions about "Artificial Intelligence(AI)" and "AI Chips development trend" and its impact on the world',
            'Useful for answering questions about "General Semiconductor development trend" or "Commonsense of Semiconductor".'
        ])

    return RouterQueryEngine(selector=LLMMultiSelector.from_defaults(),
                             query_engine_tools=engine_tools,
                             service_context=service_context)


def main(args: argparse.Namespace):
    questions_df = pd.read_json(args.question_path)
    space_names = [n for arg in args.space_name for n in arg]
    document_data_paths = [n for arg in args.document_path for n in arg]
    persist_dirs = [n for arg in args.persist_dir for n in arg]

    query_engine = initialize_rag_settings(document_data_paths, space_names,
                                           persist_dirs)

    results = generate_response(questions_df, query_engine)
    with open(file=args.output_path, mode='w', encoding='UTF-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Nebula Graph Store
    parser.add_argument('--space_name', type=str, action='append', nargs='+')

    # RAG
    parser.add_argument('-d',
                        '--document_path',
                        type=str,
                        action='append',
                        nargs='+',
                        help='Path to the data to load the documents from.')
    parser.add_argument('-p',
                        '--persist_dir',
                        type=str,
                        help='Path to the directory to store the graph data.')

    # Data
    parser.add_argument('-q',
                        '--question_path',
                        type=str,
                        help='Path to the data to load the questions from.')
    parser.add_argument('-o',
                        '--output_path',
                        type=str,
                        help='Path to the output file.')

    args = parser.parse_args()

    main(args)
