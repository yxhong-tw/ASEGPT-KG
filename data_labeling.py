import os
import time
import argparse
import openai
import pandas as pd
import numpy as np

from typing import Tuple, List
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
from prompts.task_prompt import TRIPLET_LABELING_PROMPT

load_dotenv()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chatcompletion_with_backoff(**kwargs) -> openai.ChatCompletion:
    return openai.ChatCompletion.create(**kwargs)


def call_gpt_api(prompt: str, model='gpt-3.5-turbo', max_tokens=1024) -> str:
    openai.api_key = os.getenv('OPENAI_API_KEY')

    messages = [{'role': 'user', 'content': prompt}]
    completion = chatcompletion_with_backoff(model=model,
                                             max_tokens=max_tokens,
                                             temperature=0,
                                             messages=messages)

    res = completion.choices[0].message.content

    return res


def load_previous_labels(
        output_file_path: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load previous labeled results.

    Args:
        output_file_path (str): Path to the previous labeled results.

    Returns:
        Tuple[pd.DataFrame, List[str], List[str]]: A tuple of previous labeled results, labels list, and labelers list.
    """
    if not os.path.exists(output_file_path):
        return pd.DataFrame(), [], []

    previous_labels_df = pd.read_json(output_file_path)
    labels_list = previous_labels_df['label'].tolist()
    labelers_list = previous_labels_df['labeler'].tolist()

    return previous_labels_df, labels_list, labelers_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        default='./data/articles_20230901-20230921.json')
    parser.add_argument('-s', '--start', type=int, default=0)
    parser.add_argument('-e', '--end', type=int, required=True)
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='gpt-4',
        choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'])
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default='./output/results.json')
    parser.add_argument('--ignore', type=str, default='')
    args = parser.parse_args()

    model = args.model
    start_index = args.start
    end_index = args.end
    output_file_path = args.output
    ignored_data_sources = args.ignore
    ignored_data_sources = ignored_data_sources.split(
        ',') if ignored_data_sources else []
    ignored_data_sources = [src.strip() for src in ignored_data_sources]

    articles_df = (pd.read_json(args.data).replace({
        'article_content': {
            '': np.nan
        }
    }).dropna(subset=['article_content']))
    if ignored_data_sources:
        articles_df = articles_df[~articles_df['source_name'].
                                  isin(ignored_data_sources)]
    articles_df = articles_df.iloc[start_index:end_index]

    previous_labels_df, labels_list, labelers_list = load_previous_labels(
        output_file_path)
    labeled_results = pd.concat([previous_labels_df, articles_df],
                                ignore_index=True)

    prompt_template = TRIPLET_LABELING_PROMPT
    try:
        for i, article in articles_df.iterrows():
            article_content = article['article_content']

            print(i, '=' * 30)
            print(article_content)
            print('->')

            prompt = prompt_template.replace('{INPUT}', article_content)
            triplets_result = call_gpt_api(prompt, model)
            labels_list.append(triplets_result)
            labelers_list.append(model)

            print(triplets_result)
            print()

            time.sleep(0.5)
    except ValueError as e:
        print(e)
    finally:
        labeled_results = labeled_results.iloc[:len(labels_list)]
        labeled_results['label'] = labels_list
        labeled_results['labeler'] = labelers_list

        labeled_results.to_json(output_file_path,
                                orient='records',
                                indent=4,
                                force_ascii=False)
