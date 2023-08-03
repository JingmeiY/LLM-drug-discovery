import os
import numpy as np
import pandas as pd
import random, time
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)
import openai
openai.api_key = "Your Key"
import logging
from datetime import datetime
import csv
import re

def contains_unclear_words(text):
    words = ['unclear', 'insufficient', 'indeterministic', 'impossible','inconclusive']
    pattern = re.compile(r'\b(?:' + '|'.join(words) + r')\b', re.IGNORECASE)
    return pattern.search(text) is not None


def clean_text(text: str) -> str:
    """
    Cleans the input text by removing newline characters, tabs, and multiple spaces,
    and by removing numbers followed by a period at the start of a sentence.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    cleaned_text = re.sub(r'[\n\t]+| {2,}', ' ', text.strip())  # remove newlines, tabs and extra spaces
    cleaned_text = cleaned_text.replace('"', '')  # remove double quotation marks
    cleaned_text = cleaned_text.replace("'", '')  # remove single quotation marks
    # cleaned_text = re.sub(r'\b\d+\.', '', cleaned_text)  # remove numbering at the start of a sentence
    return cleaned_text.strip()  # strip leading and trailing spaces again after the substitutions


def setup_logger(log_file, logger_name=__name__, level=logging.INFO):
    # Create a logger
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create a file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(level)

    # # Create a console handler
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(level)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)
    # logger.addHandler(console_handler)

    return logger


def read_file(FILEPATH):
    with open(FILEPATH,'r') as f:
      file_text = f.read().strip()
      f.close()
    return file_text





def get_output_file(output_folder, output_file):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return os.path.join(output_folder,output_file)


# define a retry decorator
def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
        errors: tuple = (openai.error.RateLimitError, openai.error.Timeout, openai.error.ServiceUnavailableError, ),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def get_answer(model, prompt, TMP = 0):
    if model  == 'text-davinci-003':
        response = openai.Completion.create(
            model= model,
            prompt = prompt,
            temperature = TMP,
            max_tokens = 1,
            top_p =1,
            logprobs = 1)
        answer = response['choices'][0]['text'].strip()
        logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]
        # Print out the log probabilities for each token
        prob = [round(np.exp(logprob)*100, 2) for token, logprob in logprobs.items()][0]
        print(f"probability for {answer} is {prob}")
        return answer, prob


    else:
        response = openai.ChatCompletion.create(
            model= model,
            messages=prompt,
            temperature=TMP,
            max_tokens=1000,
            top_p=1)
        answer = response['choices'][0]['message']['content'].strip()
        return answer


@retry_with_exponential_backoff
def get_embedding(context, model="text-embedding-ada-002", encoding = "cl100k_base", max_tokens = 8000):
    response = openai.Embedding.create(input = context,
                                        model = model,
                                        encoding = encoding,
                                        max_tokens = max_tokens)
    return response["data"][0]["embedding"]


@retry_with_exponential_backoff
def get_fine_tuned_answer(prompt, FT_model, TMP, Max_tokens = 1):
    response = openai.Completion.create(
    model=FT_model,
    prompt=prompt,
    temperature=TMP,
    max_tokens=Max_tokens)
    answer = response['choices'][0]['text'].strip()
    # log_pro = response['choices'][0]['logprobs']['top_logprobs'][0]
    return answer






