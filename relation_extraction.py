import argparse
import json
import os
from functools import partial
from typing import Callable, List, Literal, Union

import pandas as pd
from vllm import LLM, SamplingParams


def generate(input_text: Union[str, List[str]], model_name_or_path: str,
             get_prompt_func: Callable, args: argparse.Namespace) -> List[str]:
    """Generate predictions for the given input text / list of input text.

    Args:
        input_text (Union[str, List[str]]): input text / list of input text
        model_name_or_path (str): model name or path
        get_prompt_func (Callable): function to generate a prompt
        args (argparse.Namespace): arguments for generation

    Returns:
        List[str]: predictions for the given input text / list of input text
    """

    sampling_params = SamplingParams(max_tokens=args.max_tokens,
                                     temperature=args.temperature,
                                     top_k=args.top_k,
                                     top_p=args.top_p,
                                     frequency_penalty=0.2)

    llm = LLM(model=model_name_or_path, tensor_parallel_size=2)
    if isinstance(input_text, str):
        outputs = llm.generate(get_prompt_func(input_text=input_text),
                               sampling_params)
    else:
        input_prompts = [get_prompt_func(input_text=p) for p in input_text]
        outputs = llm.generate(input_prompts, sampling_params)

    predictions = [output.outputs[0].text for output in outputs]
    return predictions


def get_prompt(strategy: Literal['alpaca', 'chatml'],
               input_text: str,
               rationale: bool = False) -> str:
    """Generate a prompt for the given strategy.

    Args:
        strategy (str): strategy to generate a prompt (alpaca or chatml)
        input_text (str): input text
        rationale (bool, optional): whether to generate a prompt for rationale

    Raises:
        ValueError: if the given strategy is not supported

    Returns:
        str: prompt
    """

    if strategy == 'alpaca':
        return f'''You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

        ### Instruction:
        給定一段新聞段落，請幫我從中找出所有的知識圖譜三元組 (頭實體, 關係, 尾實體)。請幫我過濾掉對於構成新聞段落不重要的三元組，並只給我過濾後的結果。 注意：新聞段落內可能有一個以上的三元組存在，若有多個三元組，格式請以[(頭實體1, 關係1, 尾實體1), (頭實體2, 關係2, 尾實體2)]以此類推呈現。

        ### Input:
        {input_text}

        ### Response:
    '''
    elif strategy == 'chatml':
        if rationale:
            return f'''<|im_start|>system\nYou are a helpful assistant. Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 你是一個樂於助人的助手。請你提供專業、有邏輯、內容真實且有價值的詳細回覆。你的回答對我非常重要，請幫助我完成任務，與解答任何疑惑。\n\n<|im_end|>\n
            <|im_start|>user\n你將執行關係抽取(Relation Extraction)任務。你將識別內容中的命名實體，然後提取它們之間的關係。讓我們一步一步思考，根據我提供的新聞段落，你將傳回格式為"命名實體 A, 關係, 命名實體 B"的三元組(Triplet)，與三元組對應的解釋(Rational)。請多注意新聞段落中的量詞（例如：12%）及代名詞等（例如：他），這些應為組成新聞的重要資訊。\n{input_text}<|im_end|>\n<|im_start|>assistant\n
            '''

        return f'''<|im_start|>system\nBelow is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n<|im_end|>\n
        <|im_start|>user\n給定一段新聞段落，請幫我從中找出所有的知識圖譜三元組 (頭實體, 關係, 尾實體)。請幫我過濾掉對於構成新聞段落不重要的三元組，並只給我過濾後的結果。 注意：新聞段落內可能有一個以上的三元組存在，若有多個三元組，格式請以[(頭實體1, 關係1, 尾實體1), (頭實體2, 關係2, 尾實體2)]以此類推呈現。\n{input_text}<|im_end|>\n<|im_start|>assistant\n
        '''
    else:
        raise ValueError(f'Unsupported strategy: {strategy}')


def main(args: argparse.Namespace):
    articles_df = pd.read_json(args.data)
    if args.sample_size > 0:
        sample_articles = articles_df.sample(args.sample_size,
                                             random_state=args.seed)
    else:
        sample_articles = articles_df.copy()

    predictions = generate(
        input_text=sample_articles['article_content'].tolist(),
        model_name_or_path=args.model,
        get_prompt_func=partial(get_prompt,
                                strategy=args.strategy,
                                rationale=args.rationale),
        args=args)
    try:
        sample_articles['prediction'] = predictions
        if os.path.exists(args.output):
            prev_df = pd.read_json(args.output)
            final_df = pd.concat([prev_df, sample_articles], ignore_index=True)
            final_df.to_json(args.output,
                             orient='records',
                             force_ascii=False,
                             indent=4)
        else:
            sample_articles.to_json(args.output,
                                    orient='records',
                                    force_ascii=False,
                                    indent=4)
    except Exception as e:
        print('An error occurred while saving the predictions to a file.')
        print(e)

        with open(args.output, 'w') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)

    print(f'Predictions saved to {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=True,
                        help='Model name or path')
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        required=True,
                        help='Path to the data file')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        required=True,
                        help='Path to the output file')

    # Data parameters
    parser.add_argument(
        '--sample_size',
        type=int,
        default=-1,
        help=
        'Number of samples to generate predictions for (default: -1, generate predictions for all data)'
    )
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed for sampling')

    # Generation parameters
    parser.add_argument('--temperature',
                        type=float,
                        default=0.1,
                        help='Temperature for sampling')
    parser.add_argument('--top_k',
                        type=int,
                        default=-1,
                        help='Top-k for sampling')
    parser.add_argument('--top_p',
                        type=float,
                        default=0.7,
                        help='Top-p for sampling')
    parser.add_argument('--max_tokens',
                        type=int,
                        default=2048,
                        help='Maximum number of tokens to generate')

    # Prompt parameters
    parser.add_argument('-s',
                        '--strategy',
                        type=str,
                        default='alpaca',
                        choices=['alpaca', 'chatml'],
                        help='Strategy to prompt template')
    parser.add_argument('--rationale',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Whether to generate a prompt for rationale')

    args = parser.parse_args()

    main(args)
