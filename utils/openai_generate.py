import openai
import strictfire
import os
import threading
import json
from time import sleep
from typing import List, Dict
from openai.error import (
    RateLimitError,
    APIError,
    APIConnectionError,
    Timeout,
    AuthenticationError,
    ServiceUnavailableError,
    InvalidRequestError,
)
from transformers import GPT2Tokenizer
import logging
import json
import random
import threading
from time import sleep, time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class KeyPool:
    """A pool of API keys for OpenAI's API. Support thread-safe operations"""

    def __init__(self, accounts_list_path, verbose: bool = False):
        self.accounts = json.load(open(accounts_list_path))
        for account in self.accounts:
            account["status"] = "ready"
        self.ready_num = threading.Semaphore(len(self.accounts))
        self.waiting_num = threading.Semaphore(0)
        self.blocked_num = threading.Semaphore(0)
        self.lock = threading.Lock()
        if verbose:
            threading.Thread(target=self.report_stats, daemon=True).start()

    def report_stats(self):
        while True:
            logger.debug(
                f"ready_num: {self.ready_num._value}, waiting_num:"
                f" {self.waiting_num._value}, blocked_num:"
                f" {self.blocked_num._value}"
            )
            sleep(2 * 60)

    def pop(self, get_account=False):
        """Get a key from the pool

        Args:
            get_account (bool, optional): If True, return the account dict. Defaults to False.
        """
        with self.lock:
            random_list = list(range(len(self.accounts)))
            random.shuffle(random_list)
            for i in random_list:
                account = self.accounts[i]
                if account["status"] == "ready":
                    account["status"] = "waiting"
                    self.ready_num.acquire()  # self.ready_num -= 1
                    self.waiting_num.release()  # self.waiting_num += 1
                    if not get_account:
                        return account["key"]
                    else:
                        return account
        return ""

    def free(self, key: str):
        """Free a key"""
        with self.lock:
            for account in self.accounts:
                if account["key"] == key:
                    if account["status"] == "waiting":
                        account["status"] = "ready"
                        self.ready_num.release()  # self.ready_num += 1
                        self.waiting_num.acquire()  # self.waiting_num -= 1
                    else:
                        logger.critical(
                            f"Key found for {key} but it's status is"
                            f" {account['status']} instead of waiting when"
                            " trying to free."
                        )
                    return
        logger.critical(f"Key not found for {key} when trying to free.")

    def unblock(self, key: str):
        """Unblock a key"""
        with self.lock:
            for account in self.accounts:
                if account["key"] == key:
                    if account["status"] == "blocked":
                        account["status"] = "ready"
                        self.blocked_num.acquire()  # self.blocked_num -= 1
                        self.ready_num.release()  # self.ready_num += 1
                    else:
                        logger.critical(
                            f"Key found for {key} but it's status is"
                            f" {account['status']} instead of blocked when"
                            " trying to unblock."
                        )
                    return
        logger.critical(f"Key not found for {key} when trying to unblock.")

    def block(self, key: str, duration_sec=5):
        """Block a key for a while"""
        with self.lock:
            for account in self.accounts:
                if account["key"] == key:
                    if account["status"] == "waiting":
                        account["status"] = "blocked"
                        self.blocked_num.release()  # self.blocked_num += 1
                        self.waiting_num.acquire()  # self.waiting_num -= 1
                        # unblock after duration_sec
                        threading.Timer(
                            duration_sec, self.unblock, args=[key]
                        ).start()
                    else:
                        logger.critical(
                            f"Key found for {key} but it's status is"
                            f" {account['status']} instead of waiting when"
                            " trying to block."
                        )
                    return
        logger.critical(f"Key not found for {key} when tyring to block.")


def create_generate():
    folder = os.path.dirname(os.path.abspath(__file__))
    keypool = KeyPool(
        accounts_list_path=os.path.join(folder, "api_keys.json"),
        verbose=False,
    )

    def generate(
        model_name: str,
        prompt: str = "",
        messages: List[Dict[str, str]] = [],
        print_prompt: bool = False,
        max_tokens: int = 128,
        temperature: float = 0,
        top_p: float = 1,
        seed: int = 111,
        end_tokens: List[str] = [],
    ):
        prompt = prompt.replace("\\n", "\n")
        if print_prompt:
            print("Prompt:", prompt)
            print("=" * 50)
        model = model_name
        trial = 20
        for i in range(trial):
            key = keypool.pop()
            while key == "":
                logger.debug("No key is ready. Retry after 5 seconds.")
                sleep(5)
                key = keypool.pop()

            logger.debug(f"Using {key} as api key.")
            try:
                start_time = time()
                used_time = None
                if  "gpt-3.5" in model or "gpt-4" in model:
                    completion = openai.ChatCompletion.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}]
                        if prompt
                        else messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        seed=seed,
                        top_p=top_p,
                        api_key=key,
                        stop=end_tokens,
                    )
                else:
                    completion = openai.Completion.create(
                        model=model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        api_key=key,
                    )
            except (
                APIConnectionError,
                RateLimitError,
                APIError,
                Timeout,
                AuthenticationError,
                ServiceUnavailableError,
                InvalidRequestError,
            ) as e:
                keypool.block(key, 5)
                logger.exception(
                    f"{key} has error {e.__class__.__name__}. Block it for 5"
                    " seconds."
                )
                sleep(1)
            except Exception as e:
                keypool.free(key)
                logger.debug(f"Free {key}.")
                raise e
            else:
                prompt_len = completion["usage"]["prompt_tokens"]
                num_output_tokens = completion["usage"]["completion_tokens"]
                used_time = time() - start_time
                if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"]:
                    resp = completion["choices"][0]["message"]["content"]
                else:
                    resp = completion["choices"][0]["text"]
                keypool.free(key)
                logger.debug(f"Free {key}.")
                sleep(0.5)
                return resp, prompt_len, num_output_tokens / used_time
        raise Exception(f"Tried for {trial} trials but all failed.")

    return generate


generate = create_generate()


def main(
    model_name: str,
    prompt: str,
    print_prompt: bool = False,
    max_tokens: int = 128,
    temperature: float = 1,
    top_p: float = 1,
):
    print(
        generate(
            model_name=model_name,
            prompt=prompt,
            print_prompt=print_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    )


if __name__ == "__main__":
    strictfire.StrictFire(main)
