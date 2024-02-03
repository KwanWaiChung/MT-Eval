import os
import json
import torch
import numpy as np
import importlib.util

from copy import deepcopy
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
    StoppingCriteria,
)
from datasets import load_dataset
from strictfire import StrictFire
from tqdm import tqdm
from utils.misc import get_logger, config
from utils.openai_generate import generate
from utils.constants import INFERENCE_OUTPUT
from typing import Dict, Any, List, Literal
from time import time

package_name = "flash_attn"
spec = importlib.util.find_spec(package_name)
FLASH_AVAILABLE = True
if spec is None:
    FLASH_AVAILABLE = False

SEED = 111
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, tokenizer, stops=[]):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][-15:])
        return any([stop in tokens[-len(stop) :] for stop in self.stops])


def llama_generate(
    prompt,
    model,
    tokenizer,
    debug: bool = False,
    end_tokens: List[str] = [],
    **kwargs,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if tokenizer.eos_token:
        end_tokens.append(tokenizer.eos_token)
    stopping_criteria = StoppingCriteriaSub(tokenizer, end_tokens)
    if not debug:
        input_ids = input_ids.to(DEVICE)
    if debug:
        output = "some dummy text."
        return output, input_ids.shape[1]
    else:
        with torch.no_grad():
            start_time = time()
            generation_output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                stopping_criteria=StoppingCriteriaList([stopping_criteria]),
                **kwargs,
            )
            used_time = time() - start_time
    s = generation_output.sequences[0]
    output_tokens = s[input_ids.shape[1] :]
    num_output_tokens = len(output_tokens)
    output = tokenizer.decode(output_tokens)
    for stop_token in end_tokens:
        output = output.replace(stop_token, "")
    return output, input_ids.shape[1], num_output_tokens / used_time


def main(
    model_name: str,
    task_name: Literal[
        "refinement_single",
        "refinement_multi",
        "expansion_single",
        "expansion_multi",
        "follow-up_single",
        "follow-up_multi",
        "recollection_single_cls",
        "recollection_multiple_cls",
        "recollection_single_global-inst",
        "recollection_multi_global-inst",
        "cls_ablation_gold",
        "cls_ablation_dgc",
        "cls_ablation_sgc",
        "cls_ablation_rc",
    ],
    conv_key: str = "conv",
    system_message: str = "You are a helpful, respectful and honest assistant.",
    output_key: str = "gen_resp",
    load_8bit: bool = False,
    temperature: float = 1.0,
    top_p: float = 1,
    top_k: int = 50,
    do_sample: bool = False,
    max_new_tokens: int = 1024,
    load_model_args: Dict[str, Any] = {},
    end_tokens: List[str] = [],
    resume: bool = False,
    use_gold_history: bool = False,
    n_forward: int = -1,
):
    task_type, task_subtype = task_name.split("_", 1)
    if "ablation" in task_name:
        task_type = "_".join(task_name.split("_")[:2])
        task_subtype = "_".join(task_name.split("_")[2:])
        out_filename = os.path.join(
            INFERENCE_OUTPUT,
            task_type,
            f"{task_subtype}_{model_name}.jsonl",
        )
    elif use_gold_history and "ablation" not in task_name:
        out_filename = os.path.join(
            INFERENCE_OUTPUT,
            task_type,
            f"{task_subtype}_gold_{model_name}.jsonl",
        )
    else:
        out_filename = os.path.join(
            INFERENCE_OUTPUT,
            task_type,
            f"{task_subtype}_{model_name}.jsonl",
        )
    logger = get_logger(
        name=__name__,
        console_level="info",
        file_level="debug",
        log_path=os.path.join(
            "log",
            f"{task_name}_{model_name}.log",
        ),
        maxBytes=10000000,
    )

    if not end_tokens:
        end_tokens = config[model_name]["end_tokens"]
        logger.info(f"Changed end_tokens to {end_tokens}")
    if "ablation" in task_name:
        data = [
            json.loads(row)
            for row in open(os.path.join("data", f"{task_name}.jsonl"))
        ]
    else:
        data = load_dataset(
            "wckwan/MT-Eval", task_name, split="test"
        ).to_list()
    out_data = []
    if out_filename and os.path.exists(out_filename):
        out_data = [json.loads(line) for line in open(out_filename)]

    print_first_prompt = False
    total_forward = sum(
        turn["do_inference"] for row in data for turn in row[conv_key]
    )
    if resume and out_data:
        matched = 0
        ori_row_map = {
            f"{row['id']}#{turn['id']}": turn
            for row in data
            for turn in row[conv_key]
        }
        for row in out_data:
            for turn in row[conv_key]:
                if not turn["do_inference"]:
                    continue
                _key = f"{row['id']}#{turn['id']}"
                if output_key in turn and _key in ori_row_map:
                    ori_row_map[_key][output_key] = turn[output_key]
                    matched += 1
        if total_forward == matched:
            print(f"{out_filename} has finished.")
            return

        logger.info(f"Resumed {matched} instances from {out_filename}.")

    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    use_openai = False
    model_path = config[model_name]["path"]
    if "gpt-3.5" in model_name or "gpt-4" in model_name:
        use_openai = True
    else:
        use_flash = config[model_name]["use_flash_attn"] and FLASH_AVAILABLE
        if use_flash:
            logger.info("Using flash attention2.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if use_flash else None,
            **load_model_args,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        tokenizer.padding_side = "left"
        try:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        except AttributeError as e:
            logger.info(e)
            logger.info(
                "Can't set the tokenizer.pad_token_id but it's probably"
                " ok if the model is chatglm."
            )
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=do_sample,
        )
    logger.info(f"loaded model `{model_name}`")
    if n_forward > 0:
        logger.info(f"Only inference on the first {n_forward} examples.")
        pbar = tqdm(total=n_forward)
    else:
        pbar = tqdm(total=total_forward)
    pbar.set_description(f"Inferencing {out_filename}")
    token_per_second_list = []
    for i, row in enumerate(data):
        # create prompt
        if use_openai:
            conv = deepcopy(config["gpt-4"]["chat_template"])
        else:
            conv = deepcopy(config[model_name]["chat_template"])
        if system_message:
            conv.set_system_message(system_message)
        for turn in row[conv_key]:
            conv.append_message(conv.roles[0], turn["user"])
            conv.append_message(conv.roles[1], turn["sys"])
            if not turn["do_inference"]:
                pbar.update(1)
                continue
            if resume and output_key in turn:
                pbar.update(1)
                if not use_gold_history:
                    conv.update_last_message(turn[output_key])
                continue
            conv.update_last_message(None)
            prompt = conv.get_prompt()
            error_occured = False
            if use_openai:
                resp, prompt_len, token_per_second = generate(
                    model_name=model_name,
                    prompt="",
                    messages=conv.to_openai_api_messages(),
                    temperature=temperature,
                    top_p=top_p,
                    end_tokens=end_tokens,
                    max_tokens=max_new_tokens,
                )
                token_per_second_list.append(token_per_second)
            else:
                try:
                    resp, prompt_len, token_per_second = llama_generate(
                        prompt=prompt,
                        model=model,
                        tokenizer=tokenizer,
                        generation_config=generation_config,
                        end_tokens=end_tokens,
                    )
                    token_per_second_list.append(token_per_second)
                except Exception as e:
                    error_occured = True
                    logger.exception(f"Error occured at {i}.")
            if not print_first_prompt and not error_occured:
                tqdm.write(prompt)
                tqdm.write(resp)
                print_first_prompt = True
            turn["error"] = error_occured
            pbar.set_postfix({"t/s": f"{np.mean(token_per_second_list):.2f}"})
            pbar.update(1)
            if use_gold_history:
                conv.update_last_message(turn["sys"])
            else:
                conv.update_last_message(resp)
            if not error_occured:
                # I like output_key to be the first key
                turn["prompt"] = prompt
                turn["prompt_len"] = prompt_len
                turn[output_key] = resp
            if i % 10 == 0:
                with open(out_filename, "w", encoding="utf-8") as f:
                    f.write(
                        "\n".join([
                            json.dumps(row, ensure_ascii=False) for row in data
                        ])
                    )
                logger.debug(
                    f"Ran {i+1}/{len(data)}."
                    f" prompt_len={prompt_len if not error_occured else 'ERROR'}."
                    f" Saved to {out_filename}"
                )
    with open(out_filename, "w", encoding="utf-8") as f:
        f.write(
            "\n".join([json.dumps(row, ensure_ascii=False) for row in data])
        )
    logger.info(f"Finished running. Output saved in {out_filename}.")


if __name__ == "__main__":
    StrictFire(main)
