# Turn level 2.0
import json
import os
from tqdm import tqdm
from typing import Literal, List, Union
from utils.misc import get_logger
from utils.constants import INFERENCE_OUTPUT, EVALUATION_OUTPUT
from nltk.tokenize import sent_tokenize
from strictfire import StrictFire

from utils.openai_generate import generate

# from tests.mocks import openai_generate_evaluate_mock as generate

OUTPUT_FOLDER = "output"
TASK_NAMES = [
    "refinement_single",
    "refinement_multi",
    "refinement_multi_gold",
    "expansion_single",
    "expansion_multi",
    "expansion_multi_gold",
    "follow-up_single",
    "follow-up_multi",
    "follow-up_multi_gold",
]
DOCUMENTS = [json.loads(row) for row in open("raw_data/documents.jsonl")]
JUDGE_MODEL = "gpt-4-0613"
TEMPERATURE = 0
MAX_NEW_TOKENS = 1024


# Trim the topic.
for doc in DOCUMENTS:
    doc["gen_resp"] = doc["gen_resp"].split("\n\n", 1)[1]

logger = get_logger(
    name=__name__,
    console_level="info",
    file_level="debug",
    log_path=os.path.join("log", "evaluate.log"),
    maxBytes=10000000,
)

# require document and queries
# queries from predictions
# document from id


def evaluate_refinement_single(filename: str):
    model = filename.split("_")[-1].split(".")[0]
    prompt_template = open("prompts/refinement_single_evaluation.txt").read()
    data = [json.loads(row) for row in open(filename)]
    n_complete = sum(
        "gen_resp" in turn for dial in data for turn in dial["conv"]
    )
    visited_ids = set()
    out_filename = os.path.join(
        EVALUATION_OUTPUT, "refinement", os.path.split(filename)[-1]
    )
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    outputs = []
    if os.path.exists(out_filename):
        outputs = [json.loads(row) for row in open(out_filename)]
        for row in outputs:
            visited_ids.add(row["id"])

    total = 200
    if len(outputs) == total:
        logger.info(f"Evaluated refinement_single for {model}")
        return
    if n_complete != total:
        logger.info(
            f"`{filename}` only has {n_complete} outputs. It should have"
            f" {total}."
        )

    pbar = tqdm(total=total, desc=f"Evaluate {filename}")
    n_evaluate = 0
    for dial_i, dial in enumerate(data):
        for turn_i, turn in enumerate(dial["conv"]):
            doc_i = int(turn["id"].split("_")[0]) - 1
            doc: str = DOCUMENTS[doc_i]["gen_resp"]
            _id = f"{dial['id']}"
            if (
                _id in visited_ids
                or not turn["do_inference"]
                or "gen_resp" not in turn
            ):
                pbar.update(1)
                continue
            query = turn["inst"]
            resp = turn["gen_resp"]
            word_count = len(resp.split())
            sent_count = len(sent_tokenize(resp))
            prompt = (
                prompt_template.replace("{response}", resp)
                .replace("{content}", doc)
                .replace("{num_words}", str(word_count))
                .replace("{num_sent}", str(sent_count))
                .replace("{constraints}", query)
            )
            resp, prompt_len, token_per_second = generate(
                model_name=JUDGE_MODEL,
                prompt=prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_NEW_TOKENS,
            )
            outputs.append({
                "gen_resp": resp,
                "from": filename,
                "prompt": prompt,
                "prompt_len": prompt_len,
                "id": _id,
                "turn": 1,
            })
            pbar.update(1)
            n_evaluate += 1
            if len(outputs) % 10 == 0:
                with open(out_filename, "w", encoding="utf-8") as f:
                    f.write(
                        "\n".join([
                            json.dumps(row, ensure_ascii=False)
                            for row in outputs
                        ])
                    )

    pbar.refresh()
    logger.info(
        f"Evaluated {n_evaluate} instances in refinement_single for"
        f" {model} . Output saved in {out_filename}."
    )


def evaluate_refinement_multi(filename: str):
    prompt_template = open("prompts/refinement_multi_evaluation.txt").read()
    model = filename.split("_")[-1].split(".")[0]
    data = [json.loads(row) for row in open(filename)]
    n_complete = sum(
        "gen_resp" in turn for dial in data for turn in dial["conv"]
    )
    visited_ids = set()
    out_filename = os.path.join(
        EVALUATION_OUTPUT, "refinement", os.path.split(filename)[-1]
    )

    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    outputs = []
    if os.path.exists(out_filename):
        outputs = [json.loads(row) for row in open(out_filename)]
        for row in outputs:
            visited_ids.add(row["id"])

    total = 480
    task_name = "refinement_multi"
    if "gold" in filename:
        task_name += "_gold"
    if len(outputs) == total:
        logger.info(f"Evaluated {task_name} for {model}")
        return
    if n_complete != total:
        logger.info(
            f"`{filename}` only has {n_complete} outputs. It should have 480."
        )

    prev_task_type = ""
    pbar = tqdm(total=total, desc=f"Evaluate {filename}")
    n_evaluate = 0
    for dial_i, dial in enumerate(data):
        doc_i = int(dial["conv"][0]["id"].split("_")[0]) - 1
        doc: str = DOCUMENTS[doc_i]["gen_resp"]
        constraints = []
        prev_task_type: str = dial["conv"][0]["id"].split("_")[1]
        resp_turn_i = 0
        for turn_i, turn in enumerate(dial["conv"]):
            _id = f"{dial['id']}#{turn['id']}"
            resp_turn_i += turn["do_inference"]
            if (
                _id in visited_ids
                or not turn["do_inference"]
                or "gen_resp" not in turn
            ):
                pbar.update(1)
                continue
            cur_task_type = turn["id"].split("_")[1]
            if prev_task_type != cur_task_type:
                constraints = []
                prev_task_type = cur_task_type
            query = turn["inst"]
            constraints.append(query)
            resp = turn["gen_resp"]
            word_count = len(resp.split())
            sent_count = len(sent_tokenize(resp))
            prompt = (
                prompt_template.replace("{response}", resp)
                .replace("{content}", doc)
                .replace("{num_words}", str(word_count))
                .replace("{num_sent}", str(sent_count))
                .replace(
                    "{constraints}",
                    "\n".join(
                        [f"{i}. {c}" for i, c in enumerate(constraints, 1)]
                    ),
                )
            )

            resp, prompt_len, token_per_second = generate(
                model_name=JUDGE_MODEL,
                prompt=prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_NEW_TOKENS,
            )
            outputs.append({
                "gen_resp": resp,
                "from": filename,
                "prompt": prompt,
                "prompt_len": prompt_len,
                "id": _id,
                "turn": resp_turn_i,
            })
            n_evaluate += 1
            pbar.update(1)
            visited_ids.add(_id)
            if len(outputs) % 10 == 0:
                with open(out_filename, "w", encoding="utf-8") as f:
                    f.write(
                        "\n".join([
                            json.dumps(row, ensure_ascii=False)
                            for row in outputs
                        ])
                    )

    with open(out_filename, "w", encoding="utf-8") as f:
        f.write(
            "\n".join([json.dumps(row, ensure_ascii=False) for row in outputs])
        )
    pbar.refresh()
    logger.info(
        f"Evaluated {n_evaluate} instances in {task_name} for"
        f" {model} . Output saved in {out_filename}."
    )


def evaluate_follow_up_multi(filename: str):
    prompt_template = open("prompts/mt-bench_evaluation.txt").read()
    model = filename.split("_")[-1].split(".")[0]
    data = [json.loads(row) for row in open(filename)]
    n_complete = sum(
        "gen_resp" in turn
        for dial in data
        for turn in dial["conv"]
        if turn["do_inference"]
    )
    visited_ids = set()
    out_filename = os.path.join(
        EVALUATION_OUTPUT, "follow-up", os.path.split(filename)[-1]
    )

    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    outputs = []
    if os.path.exists(out_filename):
        outputs = [json.loads(row) for row in open(out_filename)]
        for row in outputs:
            visited_ids.add(row["id"])

    total = 240
    task_name = "follow-up_multi"
    if "gold" in filename:
        task_name += "_gold"
    if len(outputs) == total:
        logger.info(f"Evaluated {task_name} for {model}")
        return
    if n_complete != total:
        logger.info(
            f"`{filename}` only has {n_complete} outputs. It should have"
            f" {total}."
        )
    pbar = tqdm(total=total, desc=f"Evaluate {filename}")
    n_evaluate = 0
    for dial in data:
        resp_turn_i = 0
        for i, turn in enumerate(dial["conv"]):
            resp_turn_i += turn["do_inference"]
            _id = f"{dial['id']}#{turn['id']}"
            if not turn["do_inference"]:
                continue
            if _id in visited_ids or "gen_resp" not in turn:
                pbar.update(1)
                continue
            resp = turn["gen_resp"].strip()
            conversation = [
                f"User: {dial['conv'][i-1]['user'].strip()}",
                f"Assistant: {dial['conv'][i-1]['sys'].strip()}",
                f"User: {turn['user'].strip()}",
                f"Assistant: {resp}",
            ]
            word_count = len(resp.split())
            sent_count = len(sent_tokenize(resp))
            prompt = (
                prompt_template.replace(
                    "{conversation}", "\n".join(conversation)
                )
                .replace("{num_words}", str(word_count))
                .replace("{num_sent}", str(sent_count))
            )
            resp, prompt_len, token_per_second = generate(
                model_name=JUDGE_MODEL,
                prompt=prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_NEW_TOKENS,
            )
            outputs.append({
                "gen_resp": resp,
                "from": filename,
                "prompt": prompt,
                "prompt_len": prompt_len,
                "id": _id,
                "turn": resp_turn_i,
            })
            n_evaluate += 1
            pbar.update(1)
            visited_ids.add(_id)
            if len(outputs) % 10 == 0:
                with open(out_filename, "w", encoding="utf-8") as f:
                    f.write(
                        "\n".join([
                            json.dumps(row, ensure_ascii=False)
                            for row in outputs
                        ])
                    )
    with open(out_filename, "w", encoding="utf-8") as f:
        f.write(
            "\n".join([json.dumps(row, ensure_ascii=False) for row in outputs])
        )
    pbar.refresh()
    logger.info(
        f"Evaluated {n_evaluate} instances in {task_name} for"
        f" {model} . Output saved in {out_filename}."
    )


def evaluate_expansion(filename: str, mode: Literal["single", "multi"]):
    prompt_template = open("prompts/expansion_evaluation.txt").read()
    data = [json.loads(row) for row in open(filename)]
    model = filename.split("_")[-1].split(".")[0]

    n_complete = sum(
        "gen_resp" in turn
        for dial in data
        for turn in dial["conv"]
        if turn["do_inference"]
    )
    visited_ids = set()
    out_filename = os.path.join(
        EVALUATION_OUTPUT, "expansion", os.path.split(filename)[-1]
    )

    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    outputs = []
    if os.path.exists(out_filename):
        outputs = [json.loads(row) for row in open(out_filename)]
        for row in outputs:
            visited_ids.add(row["id"])

    total = 70
    task_name = f"expansion_{mode}"
    if "gold" in filename:
        task_name += "_gold"

    if len(outputs) == total:
        logger.info(f"Evaluated {task_name} for {model}")
        return
    if n_complete != total:
        logger.info(
            f"`{filename}` only has {n_complete} outputs. It should have"
            f" {total}."
        )
    pbar = tqdm(total=total, desc=f"Evaluate {filename}")
    n_evaluate = 0
    for dial in data:
        resp_turn_i = 0
        for i, turn in enumerate(dial["conv"]):
            resp_turn_i += turn["do_inference"]
            if mode == "multi":
                _id = f"{dial['id']}#{turn['id']}"
            else:
                _id = f"{dial['id']}"
            if (
                _id in visited_ids
                or not turn["do_inference"]
                or "gen_resp" not in turn
            ):
                pbar.update(1)
                continue
            if mode == "multi":
                doc_i = int(turn["id"].split("_")[0]) - 1
            else:
                doc_i = int(turn["id"].split("#")[1].split("_")[0]) - 1
            doc: str = DOCUMENTS[doc_i]["gen_resp"]
            inst = turn["inst"]
            resp = turn["gen_resp"].strip()
            prompt = (
                prompt_template.replace("{response}", resp)
                .replace("{content}", doc)
                .replace("{constraints}", inst)
            )
            resp, prompt_len, token_per_second = generate(
                model_name=JUDGE_MODEL,
                prompt=prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_NEW_TOKENS,
            )
            outputs.append({
                "gen_resp": resp,
                "from": filename,
                "prompt": prompt,
                "prompt_len": prompt_len,
                "id": _id,
                "turn": resp_turn_i,
            })
            n_evaluate += 1
            pbar.update(1)
            visited_ids.add(_id)
            if len(outputs) % 10 == 0:
                with open(out_filename, "w", encoding="utf-8") as f:
                    f.write(
                        "\n".join([
                            json.dumps(row, ensure_ascii=False)
                            for row in outputs
                        ])
                    )
    with open(out_filename, "w", encoding="utf-8") as f:
        f.write(
            "\n".join([json.dumps(row, ensure_ascii=False) for row in outputs])
        )
    pbar.refresh()
    logger.info(
        f"Evaluated {n_evaluate} instances in {task_name} for"
        f" {model} . Output saved in {out_filename}."
    )


def evaluate_follow_up_single(filename: str):
    prompt_template = open("prompts/mt-bench_evaluation.txt").read()
    data = [json.loads(row) for row in open(filename)]
    model = filename.split("_")[-1].split(".")[0]

    n_complete = sum(
        "gen_resp" in turn
        for dial in data
        for turn in dial["conv"]
        if turn["do_inference"]
    )
    visited_ids = set()
    out_filename = os.path.join(
        EVALUATION_OUTPUT, "follow-up", os.path.split(filename)[-1]
    )

    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    outputs = []
    if os.path.exists(out_filename):
        outputs = [json.loads(row) for row in open(out_filename)]
        for row in outputs:
            visited_ids.add(row["id"])

    total = 240
    if len(outputs) == total:
        logger.info(f"Evaluated follow-up_single for {model}")
        return
    if n_complete != total:
        logger.info(
            f"`{filename}` only has {n_complete} outputs. It should have"
            f" {total}."
        )
    pbar = tqdm(total=total, desc=f"Evaluate {filename}")
    n_evaluate = 0
    for dial in data:
        for i, turn in enumerate(dial["conv"]):
            _id = f"{dial['id']}"
            if (
                _id in visited_ids
                or not turn["do_inference"]
                or "gen_resp" not in turn
            ):
                pbar.update(1)
                continue
            resp = turn["gen_resp"].strip()
            dial_id, turn_id = dial["id"].split("#")
            conversation = [
                f"User: {turn['user'].strip()}",
                f"Assistant: {resp}",
            ]
            word_count = len(resp.split())
            sent_count = len(sent_tokenize(resp))
            prompt = (
                prompt_template.replace(
                    "{conversation}", "\n".join(conversation)
                )
                .replace("{num_words}", str(word_count))
                .replace("{num_sent}", str(sent_count))
            )
            resp, prompt_len, token_per_second = generate(
                model_name=JUDGE_MODEL,
                prompt=prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_NEW_TOKENS,
            )
            outputs.append({
                "gen_resp": resp,
                "from": filename,
                "prompt": prompt,
                "prompt_len": prompt_len,
                "id": _id,
                "turn": 1,
            })
            pbar.update(1)
            n_evaluate += 1
            visited_ids.add(_id)
            if len(outputs) % 10 == 0:
                with open(out_filename, "w", encoding="utf-8") as f:
                    f.write(
                        "\n".join([
                            json.dumps(row, ensure_ascii=False)
                            for row in outputs
                        ])
                    )
    with open(out_filename, "w", encoding="utf-8") as f:
        f.write(
            "\n".join([json.dumps(row, ensure_ascii=False) for row in outputs])
        )
    pbar.refresh()
    logger.info(
        f"Evaluated {n_evaluate} instances in follow-up_single for"
        f" {model} . Output saved in {out_filename}."
    )


def evaluate_refinement_ablation(filename: str):
    prompt_template = open("prompts/refinement_multi_evaluation.txt").read()
    model = filename.split("_")[-1].split(".")[0]
    data = [json.loads(row) for row in open(filename)]
    n_complete = sum(
        "gen_resp" in turn for dial in data for turn in dial["conv"]
    )
    visited_ids = set()
    out_filename = os.path.join(
        EVALUATION_OUTPUT, "refinement", os.path.split(filename)[-1]
    )

    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    outputs = []
    if os.path.exists(out_filename):
        outputs = [json.loads(row) for row in open(out_filename)]
        for row in outputs:
            visited_ids.add(row["id"])

    total = 120
    task_name = "refinement_ablation"
    if "front" in filename:
        task_name += "_front"
    else:
        task_name += "_middle"

    if len(outputs) == total:
        logger.info(f"Evaluated {task_name} for {model}")
        return
    if n_complete != total:
        logger.info(
            f"`{filename}` only has {n_complete} outputs. It should have 480."
        )

    prev_task_type = ""
    pbar = tqdm(total=total, desc=f"Evaluate {filename}")
    n_evaluate = 0
    for dial_i, dial in enumerate(data):
        doc_i = int(dial["conv"][0]["id"].split("_")[0]) - 1
        doc: str = DOCUMENTS[doc_i]["gen_resp"]
        constraints = []
        prev_task_type: str = dial["conv"][0]["id"].split("_")[1]
        resp_turn_i = 0
        for turn_i, turn in enumerate(dial["conv"]):
            _id = f"{dial['id']}#{turn['id']}"
            resp_turn_i += turn["do_inference"]
            if (
                _id in visited_ids
                or not turn["do_inference"]
                or "gen_resp" not in turn
            ):
                pbar.update(1)
                continue
            cur_task_type = turn["id"].split("_")[1]
            if prev_task_type != cur_task_type:
                constraints = []
                prev_task_type = cur_task_type
            query = turn["inst"]
            constraints.append(query)
            resp = turn["gen_resp"]
            word_count = len(resp.split())
            sent_count = len(sent_tokenize(resp))
            prompt = (
                prompt_template.replace("{response}", resp)
                .replace("{content}", doc)
                .replace("{num_words}", str(word_count))
                .replace("{num_sent}", str(sent_count))
                .replace(
                    "{constraints}",
                    "\n".join(
                        [f"{i}. {c}" for i, c in enumerate(constraints, 1)]
                    ),
                )
            )

            resp, prompt_len, token_per_second = generate(
                model_name=JUDGE_MODEL,
                prompt=prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_NEW_TOKENS,
            )
            outputs.append({
                "gen_resp": resp,
                "from": filename,
                "prompt": prompt,
                "prompt_len": prompt_len,
                "id": _id,
                "turn": resp_turn_i,
                "n_distract_turn": turn["n_distracts"],
            })
            n_evaluate += 1
            pbar.update(1)
            visited_ids.add(_id)
            if len(outputs) % 10 == 0:
                with open(out_filename, "w", encoding="utf-8") as f:
                    f.write(
                        "\n".join([
                            json.dumps(row, ensure_ascii=False)
                            for row in outputs
                        ])
                    )

    with open(out_filename, "w", encoding="utf-8") as f:
        f.write(
            "\n".join([json.dumps(row, ensure_ascii=False) for row in outputs])
        )
    pbar.refresh()
    logger.info(
        f"Evaluated {n_evaluate} instances in {task_name} for"
        f" {model} . Output saved in {out_filename}."
    )


def main(
    model_name: str,
    task_names: Union[
        List[
            Literal[
                "refinement_single",
                "refinement_multi",
                "refinement_multi_gold",
                "expansion_single",
                "expansion_multi",
                "expansion_multi_gold",
                "follow-up_single",
                "follow-up_multi",
                "follow-up_multi_gold",
            ]
        ],
        Literal["all"],
    ] = "all",
):
    if task_names == "all":
        task_names = TASK_NAMES
    elif isinstance(task_names, str):
        task_names = [task_names]
    for task_name in task_names:
        if task_name not in TASK_NAMES:
            raise ValueError(
                f"``{task_name}` does not require GPT-4 evaluation."
            )
        task_type, task_subtype = task_name.split("_", 1)
        filename = os.path.join(
            INFERENCE_OUTPUT, task_type, f"{task_subtype}_{model_name}.jsonl"
        )
        if not os.path.exists(filename):
            continue
        if task_name in ["refinement_multi", "refinement_multi_gold"]:
            # GPT-4 Turn Evaluation Multi Inst
            evaluate_refinement_multi(filename)
        elif task_name == "refinement_single":
            evaluate_refinement_single(
                filename
            )  # GPT-4 Single Turn Evalaution
        elif task_name in ["follow-up_multi", "follow-up_gold"]:
            evaluate_follow_up_multi(filename)  # MT-Bench-Autoregressive
        elif task_name in ["follow-up_single"]:
            evaluate_follow_up_single(filename)  # MT-Bench-Single Evaluation
        elif task_name in ["expansion_multi", "expansion_multi_gold"]:
            evaluate_expansion(filename, "multi")  # MT-Bench-Autoregressive
        elif task_name in ["expansion_single"]:
            evaluate_expansion(
                filename, "single"
            )  # MT-Bench-Single Evaluation
        elif "refinement_ablation_irrelevant" in task_name:
            evaluate_refinement_ablation(filename)


if __name__ == "__main__":
    StrictFire(main)
