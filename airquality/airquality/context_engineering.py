import xml.etree.ElementTree as ET
import re
import inspect
from typing import get_type_hints
import json
import datetime
import sys
import pandas as pd
from airquality.air_quality_data_retrieval import (
    get_historical_data_for_date,
    get_historical_data_in_date_range,
    get_future_data_in_date_range,
    get_future_data_for_date,
)
from typing import Any, Dict, List


def get_type_name(t: Any) -> str:
    """Get the name of the type."""
    name = str(t)
    if "list" in name or "dict" in name:
        return name
    else:
        return t.__name__


def serialize_function_to_json(func: Any) -> str:
    """Serialize a function to JSON."""
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    function_info = {
        "name": func.__name__,
        "description": func.__doc__,
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "returns": type_hints.get('return', 'void').__name__
    }

    for name, _ in signature.parameters.items():
        param_type = get_type_name(type_hints.get(name, type(None)))
        function_info["parameters"]["properties"][name] = {"type": param_type}

    return json.dumps(function_info, indent=2)


def get_function_calling_prompt(user_query):
    fn = """{"name": "function_name", "arguments": {"arg_1": "value_1", "arg_2": value_2, ...}}"""
    example = """{"name": "get_historical_data_in_date_range", "arguments": {"date_start": "2024-01-10", "date_end": "2024-01-14"}}"""

    prompt = f"""<|im_start|>system
You are a helpful assistant with access to the following functions:

{serialize_function_to_json(get_historical_data_for_date)}

{serialize_function_to_json(get_historical_data_in_date_range)}

{serialize_function_to_json(get_future_data_for_date)}

{serialize_function_to_json(get_future_data_in_date_range)}

###INSTRUCTIONS:
- You need to choose one function to use and retrieve paramenters for this function from the user input.
- If the user query contains 'will', and specifies a single day or date, use get_future_data_in_date_range function
- If the user query contains 'will', and specifies a range of days or dates, use get_future_data_in_date_range function.
- If the user query is for future data, but only includes a single day or date, use the get_future_data_in_date_range function,
- If the user query contains 'today' or 'yesterday', use get_historical_data_for_date function.
- If the user query contains 'tomorrow', use get_future_data_in_date_range function.
- If the user query is for historical data, and specifies a range of days or dates, use use get_historical_data_for_date function.
- If the user says a day of the week, assume the date of that day is when that day next arrives.
- Do not include feature_view and model parameters.
- Provide dates STRICTLY in the YYYY-MM-DD format.
- Generate an 'No Function needed' string if the user query does not require function calling.

IMPORTANT: Today is {datetime.date.today().strftime("%A")}, {datetime.date.today()}.

To use one of there functions respond STRICTLY with:
<onefunctioncall>
    <functioncall> {fn} </functioncall>
</onefunctioncall>

###EXAMPLES

EXAMPLE 1:
- User: Hi!
- AI Assiatant: No Function needed.

EXAMPLE 2:
- User: Is this Air Quality level good or bad?
- AI Assiatant: No Function needed.

EXAMPLE 3:
- User: When and what was the minimum air quality from 2024-01-10 till 2024-01-14?
- AI Assistant:
<onefunctioncall>
    <functioncall> {example} </functioncall>
</onefunctioncall>
<|im_end|>

<|im_start|>user
{user_query}
<|im_end|>

<|im_start|>assistant"""

    return prompt


def generate_hermes(user_query: str, model_llm, tokenizer) -> str:
    """Retrieves a function name and extracts function parameters based on the user query."""
    import torch

    prompt = get_function_calling_prompt(user_query)

    tokens = tokenizer(prompt, return_tensors="pt").to(model_llm.device)
    input_size = tokens.input_ids.numel()
    with torch.inference_mode():
        generated_tokens = model_llm.generate(
            **tokens,
            use_cache=True,
            do_sample=True,
            temperature=0.2,
            top_p=1.0,
            top_k=0,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        generated_tokens.squeeze()[input_size:],
        skip_special_tokens=True,
    )


def function_calling_with_openai(user_query: str, client) -> str:
    """
    Generates a response using OpenAI's chat API.

    Args:
        user_query (str): The user's query or prompt.
        client: The OpenAI client instance.

    Returns:
        str: The generated response from the assistant.
    """

    instructions = get_function_calling_prompt(user_query).split('<|im_start|>user')[0]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_query},
        ]
    )

    if completion and completion.choices:
        last_choice = completion.choices[0]
        if last_choice.message:
            return last_choice.message.content.strip()
    return ""


def function_calling_with_ollama(user_query: str, client, model="qwen3:4b") -> str:
    """
    Generates a function-calling response using a local Ollama model.

    Args:
        user_query (str): The user's query or prompt.
        client: The OpenAI-compatible client pointing at Ollama.
        model (str): The model name to use.

    Returns:
        str: The generated response from the assistant.
    """

    instructions = get_function_calling_prompt(user_query).split('<|im_start|>user')[0]

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_query},
        ]
    )

    if completion and completion.choices:
        last_choice = completion.choices[0]
        if last_choice.message:
            return last_choice.message.content.strip()
    return ""


def extract_function_calls(completion: str) -> List[Dict[str, Any]]:
    """Extract function calls from completion."""
    completion = completion.strip()
    pattern = r"(<onefunctioncall>(.*?)</onefunctioncall>)"
    match = re.search(pattern, completion, re.DOTALL)
    if not match:
        return None

    multiplefn = match.group(1)
    root = ET.fromstring(multiplefn)
    functions = root.findall("functioncall")

    return [json.loads(fn.text) for fn in functions]


def invoke_function(function, feature_view, weather_fg, model) -> pd.DataFrame:
    """Invoke a function with given arguments."""
    function_name = function['name']
    arguments = function['arguments']

    function_output = getattr(sys.modules[__name__], function_name)(
        **arguments,
        feature_view=feature_view,
        weather_fg=weather_fg,
        model=model,
    )

    if type(function_output) == str:
        return function_output

    function_output['pm25'] = function_output['pm25'].apply(round, ndigits=2)
    return function_output


def get_context_data(user_query: str, feature_view, weather_fg, model_air_quality, model_llm=None, tokenizer=None, client=None) -> str:
    """
    Retrieve context data based on user query.

    Args:
        user_query (str): The user query.
        feature_view: Feature View for data retrieval.
        weather_fg: Weather feature group.
        model_air_quality: The air quality model.
        model_llm: Local LLM model (optional).
        tokenizer: The tokenizer (optional).
        client: OpenAI-compatible client (optional).

    Returns:
        str: The context data.
    """
    if client:
        completion = function_calling_with_ollama(user_query, client)
    elif model_llm and tokenizer:
        completion = generate_hermes(
            user_query,
            model_llm,
            tokenizer,
        )
    else:
        return ''

    functions = extract_function_calls(completion)

    if functions:
        data = invoke_function(functions[0], feature_view, weather_fg, model_air_quality)

        if isinstance(data, pd.DataFrame):
            return f'Air Quality Measurements:\n' + '\n'.join(
                [f'Date: {row["date"]}; Air Quality: {row["pm25"]}' for _, row in data.iterrows()]
            )
        return data

    return ''
