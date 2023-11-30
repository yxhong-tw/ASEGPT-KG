import os
import time
import argparse
import openai
import pandas as pd
import numpy as np
import json

from typing import Tuple, List, Dict, Any
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
from prompts.task_prompt import FIND_BETTER_SCHEMA_PROMPT
from uie_predictor import UIEPredictor

load_dotenv()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chatcompletion_with_backoff(**kwargs) -> openai.ChatCompletion:
    return openai.ChatCompletion.create(**kwargs)


def call_gpt_api(messages: List[Dict[str, Any]],
                 model='gpt-3.5-turbo',
                 max_tokens=1024) -> str:
    openai.api_key = os.getenv('OPENAI_API_KEY')

    completion = chatcompletion_with_backoff(model=model,
                                             max_tokens=max_tokens,
                                             temperature=0,
                                             messages=messages)

    res = completion.choices[0].message.content

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='./data/news.json')
    parser.add_argument('-s', '--start', type=int, default=0)
    parser.add_argument('-e', '--end', type=int, required=True)
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='gpt-3.5-turbo-16k',
        choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'])
    parser.add_argument('-o', '--output', type=str, default='./output')
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

    schemas = []
    all_messages = []

    schema = {}
    ie = UIEPredictor(model="uie-base", schema=schema, device="gpu")

    prompt_template = FIND_BETTER_SCHEMA_PROMPT
    try:
        for i, article in articles_df.iterrows():
            article_content = article['article_content']
            prompt = prompt_template.replace('{INPUT1}', article_content)

            print(i, '=' * 30)
            print(f"article_content: \n{article_content}")

            counter = 0
            # messages = []
            schema = {'事件': ['公司', '人', '数量']}

            while True:
                messages = []
                triplets = []

                ie.set_schema(schema)
                uie_result = ie(article_content)[0]

                try:
                    for heads in uie_result[0].values():
                        for head in heads:
                            for relation, tails in head["relations"].items():
                                for tail in tails:
                                    triplets.append(
                                        (head["text"], relation, tail["text"]))
                except:
                    pass

                print(f"UIE's triplets: \n{triplets}")
                print(f"UIE's schema: \n{schema}")

                _prompt = prompt.replace('{INPUT2}', str(triplets))
                _prompt = _prompt.replace('{INPUT3}', str(schema))

                message = {'role': 'user', 'content': _prompt}
                print(f"ChatGPT's input: \n{message}")

                messages.append(message)
                all_messages.append(message)

                result = call_gpt_api(messages, model)
                print(f"ChatGPT's origin reponse: \n{result}")

                if result == "任务完成":
                    break

                first_idx = result.find('{')
                last_idx = result.find('}')
                result = result[first_idx:last_idx + 1]
                result = eval(result)
                print(f"ChatGPT's parsed reponse: \n{result}")

                schemas.append(result)

                if schema == result or counter >= 30:
                    break
                else:
                    schema = result
                    message = {'role': 'assistant', 'content': str(schema)}
                    messages.append(message)
                    all_messages.append(message)
                    counter += 1

                # if len(messages) > 6:
                #     messages = messages[2:]

                time.sleep(0.5)
    except ValueError as e:
        print(e)
    finally:
        with open(file=(output_file_path + "/schemas.json"),
                  mode="w",
                  encoding="UTF-8") as f:
            json.dump(obj=schemas, fp=f, ensure_ascii=False)

        with open(file=(output_file_path + "/messages.json"),
                  mode="w",
                  encoding="UTF-8") as f:
            json.dump(obj=all_messages, fp=f, ensure_ascii=False)
