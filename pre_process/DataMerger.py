import os

from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm
from typing import (
    Callable,
    Dict,
    List,
    Union,
)
from vllm import (
    LLM,
    SamplingParams,
)

from prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
)

load_dotenv()


class DataMerger():

    def __init__(
        self,
        data: List[Dict[str, Union[float, int, List[str]]]],
        use_local_lm: bool = True,
        is_chunk: bool = True,
    ) -> None:
        self.data = data
        self.use_local_lm = use_local_lm

        if use_local_lm:
            login(token=os.getenv("HUGGINGFACE_TOKEN"))

            self.sampling_params = SamplingParams(
                max_tokens=2048,
                temperature=0.1,
                top_k=-1,
                top_p=0.7,
                frequency_penalty=0.2,
            )

            self.llm = LLM(
                model="mistralai/Mistral-7B-Instruct-v0.3",
                max_model_len=8192,
                tensor_parallel_size=2,
                max_num_seqs=128,
                max_num_batched_tokens=8192,
            )
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.base = None
        self.match = None
        self.merged = None

        if is_chunk:
            self.base = "base_string_chunk"
            self.match = "match_string_chunk"
            self.merged = "merged_string"
        else:
            self.base = "base_data"
            self.match = "match_data"
            self.merged = "merged_data"

    def generate(
        self,
        input_text: Union[List[str], str],
        get_prompt_func: Callable,
    ) -> List[str]:
        """ Generate predictions for the given (list of) input text.

        Args:
            input_text (Union[List[str], str]): The (list of) input text.
            get_prompt_func (Callable): The function to generate a prompt.

        Returns:
            List[str]: Predictions for the given (list of) input text.
        """

        if isinstance(
                input_text,
                str,
        ):
            outputs = self.llm.generate(
                prompts=get_prompt_func(input_text=input_text),
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
        else:
            input_prompts = [get_prompt_func(input_text=p) for p in input_text]

            outputs = self.llm.generate(
                prompts=input_prompts,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

        predictions = [output.outputs[0].text for output in outputs]

        return predictions

    def get_prompt(
        self,
        input_text: str,
    ) -> str:
        """ Generate a prompt for the given strategy.

        Args:
            input_text (str): The input text.

        Returns:
            str: The prompt.
        """

        # return f'''[SYSTEM]\nYou are a professional clerical worker who can merge and rewrite two given articles based on given objective facts without including conjecture and illusion. Your job is to objectively combine and rewrite the given articles in Traditional Chinese.\n您是一個專業的文書工作者，您能根據給定的客觀事實，合併、改寫給定的兩篇文章，而不包含臆測與幻覺。您的工作是根據給定的文章，客觀地以繁體中文為其做合併與改寫。\n[/SYSTEM]\n[USER]\n{input_text}\n[/USER]
        # '''

        return f'''[INST]\n{input_text}\n[/INST]'''

    def merge(self) -> List[Dict[str, Union[float, int, List[str], str]]]:
        """ Merge the data.

        Returns:
            List[Dict[str, Union[float, int, List[str], str]]]: The merged data.
        """

        output_data = []

        for one_data in tqdm(self.data):
            prompt = USER_PROMPT.replace(
                "{INPUT_1}",
                one_data[self.base][0],
            ).replace(
                "{INPUT_2}",
                one_data[self.match][0],
            )

            if self.use_local_lm:
                predictions = self.generate(
                    input_text=prompt,
                    get_prompt_func=self.get_prompt,
                )

                one_data[self.merged] = predictions[0]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]

                response = call_gpt_api(
                    client=self.client,
                    messages=messages,
                    model="gpt-4o-mini",
                )

                one_data[self.merged] = response

            output_data.append(one_data)

        return output_data


@retry(
    wait=wait_random_exponential(
        min=1,
        max=10,
    ),
    stop=stop_after_attempt(8),
)
def completions_with_backoff(
    client: OpenAI,
    **kwargs,
) -> ChatCompletion:
    """ The function to call OpenAI API with backoff.

    Args:
        client (OpenAI): The OpenAI client.

    Returns:
        ChatCompletion: The response from OpenAI API.
    """

    return client.chat.completions.create(**kwargs)


def call_gpt_api(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int = 2048,
) -> str:
    """ Call OpenAI API.

    Args:
        client (OpenAI): The OpenAI client.
        messages (List[Dict[str, str]]): The messages to send to the API.
        model (str): The model to use.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 2048.

    Returns:
        str: The response from the API.
    """

    response = completions_with_backoff(
        client=client,
        messages=messages,
        max_tokens=max_tokens,
        model=model,
    )

    return response.choices[0].message.content
