import sys


sys.setrecursionlimit(12000 * 12000)
from rouge import Rouge
from strictfire import StrictFire
from typing import Any, List, Literal, Dict, Callable
from utils.bleu import compute
from multiprocessing import Pool
from tabulate import tabulate
from utils.constants import (
    DATASET_MAP,
    MODEL_MAP,
    INFERENCE_OUTPUT,
    EVALUATION_OUTPUT,
    RESULT_OUTPUT,
)
from utils.misc import get_logger, config
from utils.global_inst import INSTRUCTIONS
from utils.parse import parse_rel_output, parse_ner_pos_output
from collections import defaultdict
import pandas as pd
import json
import numpy as np
import os
import dill, multiprocessing
import string

en_rouge = Rouge()
dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
multiprocessing.reduction.ForkingPickler = dill.Pickler
multiprocessing.reduction.dump = dill.dump

ABLATION_MODElS = [
    "GPT-4",
    "Vicuna-13B-v1.5",
    "Qwen-chat-14B",
    "Mistral-Instruct-7B",
    "Mixtral-Instruct-8x7B",
]
LOG_PATH = os.path.join("log", "evaluate")
logger = get_logger(
    name=__name__,
    console_level="info",
    file_level="debug",
    log_path=LOG_PATH,
    maxBytes=10000000,
)


class ProcessPipeline:
    def __init__(self, fns: List[Callable] = []):
        if not isinstance(fns, list):
            fns = [fns]
        self.fns = fns

    def insert(self, index: int, fn: Callable):
        self.fns.insert(index, fn)

    def __call__(self, data: List[Dict]) -> List[Dict]:
        for fn in self.fns:
            data = fn(data)
        return data


def _remove_punctuations_from_str(s: str):
    for punc in string.punctuation:
        s = s.replace(punc, " ")
    return s


def _remove_punctuations_and_lowercase(data):
    # some extra punctuations are found in gold answer.
    for row in data:
        if "gen_resp" not in row:
            continue
        gen_resp = row["gen_resp"]
        answers = []
        for ans in row["answers"]:
            ans = ans.lower()
            for punc in string.punctuation:
                ans = ans.replace(punc, " ")
            answers.append(ans.strip())
        gen_resp = gen_resp.lower()
        gen_resp = _remove_punctuations_from_str(gen_resp)
        gen_resp = gen_resp.strip()
        row["answers"] = answers
        row["gen_resp"] = gen_resp
    return data


def _backup_data(data):
    for row in data:
        if "gen_resp" in row:
            row["raw_gen_resp"] = row["gen_resp"]
        if "answers" in row:
            row["raw_answers"] = row["answers"]
    return data


def _restore_data(data):
    for row in data:
        if "raw_gen_resp" in row:
            row["gen_resp"] = row["raw_gen_resp"]
        if "raw_answers" in row:
            row["answers"] = row["raw_answers"]
    return data


def _filter_newsqa(data):
    # return data
    data = [row for row in data if int(row["resp_turn_i"]) <= 25]
    return data


# Multi-Task
def _process_rel(data: List[Dict]) -> List[Dict]:
    for row in data:
        if "gen_resp" in row:
            row["gen_resp"] = parse_rel_output(row["gen_resp"].lower())
        row["answers"] = json.loads(
            json.dumps(row["answers"], ensure_ascii=False).lower()
        )
        row["answers"] = [tuple(t) for t in row["answers"]]
    return data


def _process_rel_v1_3(data: List[Dict]) -> List[Dict]:
    for row in data:
        if "gen_resp" in row:
            row["gen_resp"] = parse_rel_output(row["gen_resp"].lower())
        row["answers"] = parse_rel_output(row["answers"][0].lower())
    return data


def _process_ner_pos(data: List[Dict]) -> List[Dict]:
    for row in data:
        if "gen_resp" in row:
            row["gen_resp"] = parse_ner_pos_output(row["gen_resp"].lower())
        row["answers"] = json.loads(
            json.dumps(row["answers"], ensure_ascii=False).lower()
        )
    return data


def _process_ner_pos_v1_3(data: List[Dict]) -> List[Dict]:
    for row in data:
        if ";" in row["answers"][0]:
            new_ans = []
            for section in row["answers"][0].split(";"):
                header, words = section.split(":", 1)
                for word in words.split(","):
                    new_ans.append(
                        (
                            header.strip().lower(),
                            _remove_punctuations_from_str(word)
                            .strip()
                            .lower(),
                        )
                    )
            row["answers"] = new_ans
            if "gen_resp" in row:
                new_resp = []
                if ";" in row["gen_resp"]:
                    for section in row["gen_resp"].split(";"):
                        if ":" not in section:
                            continue
                        header, words = section.split(":", 1)
                        for word in words.split(","):
                            new_resp.append(
                                (
                                    header.strip().lower(),
                                    _remove_punctuations_from_str(word)
                                    .strip()
                                    .lower(),
                                )
                            )
                row["gen_resp"] = new_resp
        else:
            row["answers"] = [
                _remove_punctuations_from_str(w).strip()
                for w in row["answers"][0].lower().split(",")
            ]
            if "gen_resp" in row:
                row["gen_resp"] = [
                    _remove_punctuations_from_str(w).strip()
                    for w in row["gen_resp"].lower().split(",")
                ]
    return data


def _process_cls(data: List[Dict]) -> List[Dict]:
    # extract the last word
    for row in data:
        if "gen_resp" in row:
            row["raw_gen_resp"] = row["gen_resp"]
            if row["gen_resp"]:
                gen_resp = row["gen_resp"].split()[-1]
                gen_resp = gen_resp.strip(" \"'" + string.punctuation).lower()
                row["gen_resp"] = gen_resp
    return data


def _process_mnds_new_retrieval(data: List[Dict]) -> List[Dict]:
    for row in data:
        if "gen_resp" in row:
            row["raw_gen_resp"] = row["gen_resp"]
            gen_resp = [r.strip() for r in row["gen_resp"].split(",")]
            row["gen_resp"] = gen_resp
            row["answers"] = row["answers"][0].split(", ")
    return data


def highlight_max_scores_in_latex_table(
    latex_table: str, highlight_cmds: List[str] = ["\\highlight"]
) -> str:
    def _isfloat(s: str) -> bool:
        return s.replace(".", "", 1).isdigit()

    latex_lines = latex_table.split("\n")
    n_cols = max([len(row.split("&")) for row in latex_lines])
    # might contain columns
    valid_row_idx = [
        i for i, row in enumerate(latex_lines) if len(row.split("&")) == n_cols
    ]
    data_row_idx = [
        i
        for i in valid_row_idx
        if any(
            ele.strip("\\").strip().replace(".", "", 1).isdigit()
            for j, ele in enumerate(latex_lines[i].split("&"))
            # if j in col_idx
        )
    ]
    data_row = [
        [
            ele.strip("\\").strip()
            for j, ele in enumerate(latex_lines[i].split("&"))
            # if j in col_idx
        ]
        for i in data_row_idx
    ]
    data_columns = [list(i) for i in zip(*data_row)]
    # get max and highlight
    highlighted_data_columns = []
    for col in data_columns:
        # float_col = [float(ele) for ele in col if ele.replace(".", "", 1).isdigit()]
        # if not float_col:
        #     highlighted_data_columns.append(col)
        #     continue
        # max_float = max(float_col)
        # highlighted_data_columns.append(
        #     [
        #         f"\\highlight{{{ele}}}"
        #         if _isfloat(ele) and float(ele) == max_float
        #         else ele
        #         for ele in col
        #     ]
        # )
        float_col = [
            float(ele) for ele in col if ele.replace(".", "", 1).isdigit()
        ]
        if float_col:
            for cmd in highlight_cmds:
                max_float = max([float(ele) for ele in col if _isfloat(ele)])
                col = [
                    (
                        f"{cmd}{{{ele}}}"
                        if _isfloat(ele) and float(ele) == max_float
                        else ele
                    )
                    for ele in col
                ]
        highlighted_data_columns.append(col)
    highlighted_data_rows = [list(i) for i in zip(*highlighted_data_columns)]
    for i, row in enumerate(highlighted_data_rows):
        latex_lines[data_row_idx[i]] = "&".join(row) + "\\\\"
    # new_latex_table = "\n".join(latex_lines)
    # format
    # might contain columns
    valid_row = [
        [
            ele.rstrip("\\").strip()
            for j, ele in enumerate(latex_lines[i].split("&"))
        ]
        for i in valid_row_idx
    ]
    valid_columns = [list(i) for i in zip(*valid_row)]
    max_width_per_columns = [
        max(len(ele) for ele in col) for col in valid_columns
    ]
    # ...
    formatted_valid_columns = [
        [f"  {ele:>{max_width_per_columns[i]}} " for ele in col]
        for i, col in enumerate(valid_columns)
    ]
    formatted_valid_rows = [list(i) for i in zip(*formatted_valid_columns)]
    for i, row in enumerate(formatted_valid_rows):
        latex_lines[valid_row_idx[i]] = "&".join(row) + "\\\\"
    new_latex_table = "\n".join(latex_lines)
    return new_latex_table


def try_parse_float(value):
    try:
        return float(value)
    except ValueError:
        return value


TASKS = {
    # "tedtalks": {"metric": "bleu", "lang": "zh", "end_tokens": ["Human:"]},
    # "news_commentary": {
    #     "metric": "bleu",
    #     "lang": "zh",
    #     "end_tokens": ["Human:"],
    # },
    # "cnnnews": {"metric": "rouge", "lang": "en", "end_tokens": ["Human:"]},
    "mnds-news": {
        "metric": "qa_acc",
        "lang": "en",
        "preprocess_fn": ProcessPipeline(),
        "suffix": "_v1.4",
    },
    "mnds-news_autoregressive": {
        "metric": "qa_acc",
        "lang": "en",
        "preprocess_fn": ProcessPipeline(),
        "suffix": "_v1.4",
    },
    "mnds-news-retrieval_autoregressive": {
        "metric": "f1",
        "lang": "en",
        "preprocess_fn": ProcessPipeline([_process_mnds_new_retrieval]),
        "suffix": "_v1.1",
    },
    "newsqa": {
        "metric": "qa_acc",
        "lang": "en",
        "preprocess_fn": ProcessPipeline(
            [_remove_punctuations_and_lowercase, _filter_newsqa]
        ),
        "suffix": "_v1.3",
    },
    "newsqa_autoregressive": {
        "metric": "qa_acc",
        "lang": "en",
        "preprocess_fn": ProcessPipeline(
            [_remove_punctuations_and_lowercase, _filter_newsqa]
        ),
        "suffix": "_v1.3",
    },
    "cyrus-cls": {
        "metric": "qa_acc",
        "lang": "en",
        "preprocess_fn": ProcessPipeline([_remove_punctuations_and_lowercase]),
        "suffix": "_v1.0",
    },
    "cyrus-cls_autoregressive": {
        "metric": "qa_acc",
        "lang": "en",
        "preprocess_fn": ProcessPipeline([_remove_punctuations_and_lowercase]),
        "suffix": "_v1.0",
    },
    "cyrus-qa": {
        "metric": "qa_acc",
        "lang": "en",
        "preprocess_fn": ProcessPipeline([_remove_punctuations_and_lowercase]),
        "suffix": "_v1.0",
    },
    "cyrus-cls-ablation-base": {
        "metric": "qa_acc",
        "lang": "en",
        "preprocess_fn": ProcessPipeline([_remove_punctuations_and_lowercase]),
        "suffix": "_v1.0",
    },
    "cyrus-cls-ablation-1": {
        "metric": "qa_acc",
        "lang": "en",
        "preprocess_fn": ProcessPipeline([_remove_punctuations_and_lowercase]),
        "suffix": "_v1.0",
    },
    "cyrus-cls-ablation-2": {
        "metric": "qa_acc",
        "lang": "en",
        "preprocess_fn": ProcessPipeline([_remove_punctuations_and_lowercase]),
        "suffix": "_v1.0",
    },
    "cyrus-cls-ablation-3": {
        "metric": "qa_acc",
        "lang": "en",
        "preprocess_fn": ProcessPipeline([_remove_punctuations_and_lowercase]),
        "suffix": "_v1.0",
    },
    "cyrus-qa_autoregressive": {
        "metric": "qa_acc",
        "lang": "en",
        "preprocess_fn": ProcessPipeline([_remove_punctuations_and_lowercase]),
        "suffix": "_v1.0",
    },
    "multi-task-single-inst_qa": {
        "metric": "qa_acc",
        "lang": "en",
        "preprocess_fn": ProcessPipeline(),
        "suffix": "",
    },
    "multi-task-single-inst_sum": {
        "metric": "rouge",
        "lang": "en",
        "preprocess_fn": ProcessPipeline(),
        "suffix": "",
    },
    "multi-task-single-inst_trans": {
        "metric": "bleu",
        "lang": "zh",
        "preprocess_fn": ProcessPipeline(),
        "suffix": "",
    },
    "multi-task-single-inst_ner": {
        "metric": "f1",
        "lang": "en",
        # "preprocess_fn": ProcessPipeline([_process_ner_pos]),
        "preprocess_fn": ProcessPipeline([_process_ner_pos_v1_3]),  # v1.3
        "suffix": "",
    },
    "multi-task-single-inst_pos": {
        "metric": "f1",
        "lang": "en",
        # "preprocess_fn": ProcessPipeline([_process_ner_pos]),
        "preprocess_fn": ProcessPipeline([_process_ner_pos_v1_3]),  # v1.3
        "suffix": "",
    },
    "multi-task-single-inst_rel": {
        "metric": "f1",
        "lang": "en",
        # "preprocess_fn": ProcessPipeline([_process_rel]),
        "preprocess_fn": ProcessPipeline([_process_rel_v1_3]),  # v1.3
        "suffix": "",
    },
    "multi-task-single-inst_cls": {
        "metric": "acc",
        "lang": "en",
        "preprocess_fn": ProcessPipeline([_process_cls]),
        "suffix": "",
    },
}


def _evaluate_by_dict_mp(
    data: List[Dict],
    end_tokens: List[str] = [],
    metric: Literal["rouge", "qa_acc", "acc", "bleu"] = "acc",
    lang: str = "en",
    model_name: str = "",
    task_name: str = "",
    preprocess_fn=None,
):
    result_dict = evaluate_by_dict(
        data, end_tokens, metric, lang, False, preprocess_fn=preprocess_fn
    )
    return result_dict, model_name, task_name


def evaluate_by_dict(
    data: List[Dict],
    end_tokens: List[str] = [],
    metric: Literal["rouge", "qa_acc", "acc", "bleu", "f1"] = "acc",
    lang: str = "en",
    return_rows: bool = False,
    preprocess_fn=None,
):
    scores = {}
    n_complete = 0
    n_corr = defaultdict(int)
    n_true = defaultdict(int)
    n_pred = defaultdict(int)
    n_sample = defaultdict(int)
    predictions = {}  # for bleu
    references = {}  # for bleu

    def trim_resp(data: List[Dict]) -> List[Dict]:
        for row in data:
            if "gen_resp" in row:
                gen_resp = row["gen_resp"]
                for t in end_tokens:
                    gen_resp = gen_resp.split(t)[0].strip()
                row["gen_resp"] = gen_resp
        return data

    if not preprocess_fn:
        preprocess_fn = ProcessPipeline()
    _backup_data(data)
    preprocess_fn.insert(0, trim_resp)
    data = preprocess_fn(data)
    for row in data:
        if "gen_resp" in row:
            n_complete += 1
            gen_resp = row["gen_resp"]
            # for t in end_tokens:
            #     gen_resp = gen_resp.split(t)[0].strip()
            if metric == "rouge":
                score = 0
                if gen_resp.strip():
                    score = max(
                        [
                            en_rouge.get_scores(hyps=gen_resp, refs=ref)[0][
                                "rouge-l"
                            ]["r"]
                            for ref in row["answers"]
                            if ref
                        ],
                    )
                scores.setdefault(row["resp_turn_i"], []).append(score * 100)
            elif metric == "acc":
                score = int(
                    any(
                        [
                            gen_resp.lower() == ans.lower()
                            for ans in row["answers"]
                        ]
                    )
                )
                row["score"] = score
                scores.setdefault(row["resp_turn_i"], []).append(score * 100)
            elif metric == "qa_acc":
                score = int(
                    any(
                        [
                            ans.lower().strip() in gen_resp.lower()
                            for ans in row["answers"]
                        ]
                    )
                )
                row["score"] = score
                scores.setdefault(row["resp_turn_i"], []).append(score * 100)
            elif metric == "bleu":
                predictions.setdefault(row["resp_turn_i"], []).append(gen_resp)
                references.setdefault(row["resp_turn_i"], []).append(
                    row["answers"]
                )
                predictions.setdefault("all", []).append(gen_resp)
                references.setdefault("all", []).append(row["answers"])
            elif metric == "f1":
                assert type(gen_resp) == list
                assert type(row["answers"]) == list
                gen_resp = set(gen_resp)
                answers = set(row["answers"])
                n_pred[row["resp_turn_i"]] += len(gen_resp)
                n_pred["all"] += len(gen_resp)
                n_true[row["resp_turn_i"]] += len(answers)
                n_true["all"] += len(answers)
                n_corr[row["resp_turn_i"]] += len(gen_resp & answers)
                n_corr["all"] += len(gen_resp & answers)
                n_sample[row["resp_turn_i"]] += 1
                n_sample["all"] += 1
            else:
                raise ValueError(f"Unknown metric: {metric}")

    _restore_data(data)
    results = {}
    if metric == "bleu":
        for turn_i in predictions:
            results[turn_i] = (
                compute(
                    predictions=predictions[turn_i],
                    references=references[turn_i],
                    lang=lang,
                )["bleu"]
                * 100,
                len(predictions[turn_i]),
            )
    elif metric == "f1":
        for turn_i in n_pred:
            prec, rec = 0, 0
            if n_pred[turn_i]:
                prec = n_corr[turn_i] / n_pred[turn_i]
            if n_true[turn_i]:
                rec = n_corr[turn_i] / n_true[turn_i]
            if prec + rec:
                results[turn_i] = (
                    (200 * prec * rec) / (prec + rec),
                    n_sample[turn_i],
                )
            else:
                results[turn_i] = (0, n_sample[turn_i])
    else:
        for turn_i, score in scores.items():
            results[turn_i] = (np.mean(score), len(score))
            # print(f"turn_i: {turn_i}. Score: {np.mean(score):.2f}")
        results["all"] = (
            np.mean(sum(scores.values(), start=[])),
            sum([len(score) for score in scores.values()]),
        )
    if return_rows:
        return results, data
    return results


def _evaluate_mp(
    filename: str,
    end_tokens: List[str] = [],
    metric: Literal["rouge", "qa_acc", "acc", "bleu"] = "acc",
    lang: str = "en",
    preprocess_fn=None,
    model_name: str = "",
    task_name: str = "",
):
    result_dict = evaluate(filename, end_tokens, metric, lang, preprocess_fn)
    return result_dict, model_name, task_name


def evaluate(
    filename: str,
    end_tokens: List[str] = [],
    metric: Literal["rouge", "qa_acc", "acc", "bleu"] = "acc",
    lang: str = "en",
    preprocess_fn=None,
) -> Dict:
    if not os.path.exists(filename):
        return {}
    data = [json.loads(row) for row in open(filename)]
    return evaluate_by_dict(
        data, end_tokens, metric, lang, preprocess_fn=preprocess_fn
    )


def extract_evaluation(
    task_name: Literal[
        "refinement_single",
        "refinement_multi",
        "refinement_multi_gold",
        "expansion_single",
        "expansion_multi",
        "expansion_multi_gold",
        "follow-up_single",
        "follow-up_multi",
        "follow-up_multi_gold",
    ],
):
    raw_results = []
    folder = os.path.join(EVALUATION_OUTPUT, task_name.split("_")[0])
    n_errors = 0
    if not os.path.isdir(folder):
        logger.info(f"GPT-4 evaluations not found in `{folder}`.")
    for f in os.listdir(folder):
        model = f.split("_")[-1].split(".")[0]
        expected_fn = f"{task_name.split('_', 1)[1]}_{model}.jsonl"
        if not f.endswith(".jsonl") or f != expected_fn:
            continue
        # single and multi
        fn = os.path.join(folder, f)
        data = [json.loads(row) for row in open(fn)]
        for row in data:
            if isinstance(row["gen_resp"], str):
                try:
                    row["gen_resp"] = json.loads(
                        row["gen_resp"]
                        .removeprefix("```json")
                        .removesuffix("```")
                    )
                except json.decoder.JSONDecodeError as e:
                    n_errors += 1
                    logger.error(e)
                else:
                    raw_results.append(
                        {
                            "model": model,
                            "task": task_name,
                            "turn": row["turn"],
                            "score": int(row["gen_resp"]["Score"]),
                            "id": row["id"],
                        }
                    )
    if n_errors > 0:
        logger.info(
            f"{n_errors} JSON decode error occurs with {fn}. Check"
            f" {LOG_PATH} for detail."
        )
    return raw_results


def _remove_punctuations_from_str(s: str):
    for punc in string.punctuation:
        s = s.replace(punc, " ")
    return s


def _remove_punctuations_and_lowercase(data):
    # some extra punctuations are found in gold answer.
    for row in data:
        for turn in row["conv"]:
            if "gen_resp" in turn:
                turn["gen_resp"] = _remove_punctuations_from_str(
                    turn["gen_resp"]
                ).lower()
            turn["sys"] = _remove_punctuations_from_str(turn["sys"]).lower()
    return data


def evaluate_recollection_cls():
    folder = os.path.join(INFERENCE_OUTPUT, "recollection")
    raw_results = []
    for f in os.listdir(folder):
        if not f.endswith(".jsonl") or "cls" not in f:
            continue
        model = f.split("_")[-1].split(".")[0]
        task_name = f"recollection_{f.rsplit('_', 1)[0]}"
        f = os.path.join(folder, f)
        data = _remove_punctuations_and_lowercase(
            [json.loads(row) for row in open(f)]
        )
        for dial in data:
            resp_turn_i = 0
            for turn in dial["conv"]:
                resp_turn_i += turn["do_inference"]
                if "gen_resp" not in turn or not turn["do_inference"]:
                    continue

                if "multi" in f:
                    _id = f"{dial['id']}#{turn['id']}"
                else:
                    _id = f"{dial['id']}"
                score = turn["sys"] in turn["gen_resp"]
                raw_results.append(
                    {
                        "model": model,
                        "task": task_name,
                        "turn": resp_turn_i,
                        "score": int(score) * 10,
                        "id": _id,
                    }
                )
    return raw_results


def evaluate_recollection_cls_ablation():
    folder = os.path.join(INFERENCE_OUTPUT, "cls_ablation")
    raw_results = []
    for f in os.listdir(folder):
        if not f.endswith(".jsonl"):
            continue
        model = f.split("_")[-1].split(".")[0]
        task_name = f"cls_ablation_{f.rsplit('_', 1)[0]}"
        f = os.path.join(folder, f)
        data = _remove_punctuations_and_lowercase(
            [json.loads(row) for row in open(f)]
        )
        for dial in data:
            resp_turn_i = 0
            for turn in dial["conv"]:
                resp_turn_i += turn["do_inference"]
                if "gen_resp" not in turn or not turn["do_inference"]:
                    continue

                if "multi" in f:
                    _id = f"{dial['id']}#{turn['id']}"
                else:
                    _id = f"{dial['id']}"
                score = turn["sys"] in turn["gen_resp"]
                raw_results.append(
                    {
                        "model": model,
                        "task": task_name,
                        "turn": resp_turn_i,
                        "score": int(score) * 10,
                        "id": _id,
                    }
                )
    return raw_results


def evaluate_recollection_global_inst():
    folder = os.path.join(INFERENCE_OUTPUT, "recollection")
    raw_results = []
    for f in os.listdir(folder):
        if not f.endswith(".jsonl") or "global-inst" not in f:
            continue
        model = f.split("_")[-1].split(".")[0]
        task_name = f"recollection_{f.rsplit('_', 1)[0]}"
        f = os.path.join(folder, f)
        data = [json.loads(row) for row in open(f)]
        for dial in data:
            inst_args = {k: v for k, v in dial["inst_args"].items() if v}
            inst_obj = INSTRUCTIONS[dial["inst_name"]](**inst_args)
            resp_turn_i = 0
            for turn in dial["conv"]:
                resp_turn_i += turn["do_inference"]
                if "gen_resp" not in turn or not turn["do_inference"]:
                    continue
                if "multi" in f:
                    _id = f"{dial['id']}#{turn['id']}"
                else:
                    _id = f"{dial['id']}"
                score: bool = inst_obj.check_following(turn["gen_resp"])
                raw_results.append(
                    {
                        "model": model,
                        "task": task_name,
                        "turn": resp_turn_i,
                        "score": int(score) * 10,
                        "id": _id,
                    }
                )
    return raw_results


def multi_turn(df: pd.DataFrame, tablefmt: str) -> str:
    # 1. mt_avg, mt results of four tasks.
    sub_df = df[
        (df["task"].str.contains("multi")) & (~df["task"].str.contains("gold"))
    ]
    sub_df = sub_df.pivot_table(
        index="model", columns="group", values="score"
    ).rename_axis(index=None, columns="Model")
    sub_df.insert(0, "Avg.", sub_df.mean(axis=1))
    return tabulate(
        sub_df,
        headers="keys",
        floatfmt=".2f",
        tablefmt=tablefmt,
        numalign="center",
    )


def single_turn_vs_multi_turn(df: pd.DataFrame, tablefmt: str) -> str:
    sub_df = df[
        (df["group"] != "Follow-up") & (~df["task"].str.contains("gold"))
    ]
    sub_df.loc[:, "group"] = sub_df["task"].map(
        lambda x: x.rsplit("_", 1)[0] if len(x.split("_")) > 2 else x
    )
    sub_df = sub_df.pivot_table(
        index="model", columns="group", values="score"
    ).rename_axis(index=None, columns="Model")
    sub_df = sub_df.rename(
        columns=lambda x: " ".join([w.capitalize() for w in x.split("_")])
    )
    # reorder the columns
    sub_df = sub_df[
        [
            "Recollection Single",
            "Recollection Multi",
            "Expansion Single",
            "Expansion Multi",
            "Refinement Single",
            "Refinement Multi",
        ]
    ]
    mt_avg = sub_df.loc[:, sub_df.columns.str.contains("Multi")].mean(axis=1)
    st_avg = sub_df.loc[:, sub_df.columns.str.contains("Single")].mean(axis=1)
    sub_df.insert(0, "Multi Avg.", mt_avg)
    sub_df.insert(0, "Single Avg.", st_avg)
    return tabulate(
        sub_df,
        headers="keys",
        floatfmt=".2f",
        tablefmt=tablefmt,
        numalign="center",
    )


def gold_vs_predicted(df: pd.DataFrame, tablefmt: str) -> str:
    sub_df = df[
        (df["group"] != "Follow-up") & (df["task"].str.contains("multi"))
    ]
    sub_df.loc[:, "group"] = sub_df["task"].map(
        lambda x: "_".join(x.split("_")[:2])
    )
    sub_df.loc[:, "group"] = sub_df.apply(
        lambda x: x["group"] + "_gold" if "gold" in x["task"] else x["group"],
        axis=1,
    )
    sub_df = sub_df.pivot_table(
        index="model", columns="group", values="score"
    ).rename_axis(index=None, columns="Model")
    sub_df = sub_df.rename(
        columns=lambda x: " ".join(
            [w.capitalize() for w in x.split("_") if w != "multi"]
        )
    )
    return tabulate(
        sub_df,
        headers="keys",
        floatfmt=".2f",
        tablefmt=tablefmt,
        numalign="center",
    )


def cls_ablation(df: pd.DataFrame, tablefmt: str) -> str:
    sub_df = df[
        df["task"].str.contains("cls_ablation|recollection_single_cls")
    ]
    sub_df.loc[:, "group"] = df["task"].map(
        {
            "cls_ablation_gold": "Gold",
            "cls_ablation_sgc": "SGC",
            "cls_ablation_dgc": "DGC",
            "cls_ablation_rc": "RC",
            "recollection_single_cls": "Single Turn",
        }
    )
    sub_df = sub_df.pivot_table(
        index="model", columns="group", values="score"
    ).rename_axis(index=None, columns="Model")
    return tabulate(
        sub_df,
        headers="keys",
        floatfmt=".2f",
        tablefmt=tablefmt,
        numalign="center",
    )


def main():
    tasks = [
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
    raw_results = []
    for task in tasks:
        raw_results += extract_evaluation(task)
    raw_results += evaluate_recollection_cls()
    raw_results += evaluate_recollection_global_inst()
    raw_results += evaluate_recollection_cls_ablation()

    df = pd.DataFrame(raw_results)
    df_path = os.path.join(RESULT_OUTPUT, "raw_result.csv")
    os.makedirs(os.path.dirname(df_path), exist_ok=True)
    df.to_csv(df_path, index=False, header=True)
    logger.info(f"Raw result saved to {df_path}")

    # Expected markdown output
    df["group"] = df["task"].map(lambda x: x.split("_")[0].capitalize())
    markdown_output: str = ""

    # 1. Multi-Turn
    markdown_output += "### Multi-Turn Performance\n"
    markdown_output += multi_turn(
        df,
        tablefmt="github",
    )
    markdown_output += "\n\n"

    # 2. Single-Turn vs Multi-TUrn
    if df["task"].str.contains("single").any():
        markdown_output += "### Single-Turn vs Multi-Turn\n"
        markdown_output += single_turn_vs_multi_turn(df=df, tablefmt="github")
        markdown_output += "\n\n"

    # 3. Gold vs Predicted.
    if df["task"].str.contains("gold").any():
        markdown_output += "### Self-generate vs Gold Dialogue History\n"
        markdown_output += gold_vs_predicted(
            df,
            tablefmt="github",
        )
        markdown_output += "\n\n"

    # 4. CLS Abalation
    if df["task"].str.contains("cls_ablation").any():
        markdown_output += "### Recollection CLS Ablation\n"
        markdown_output += cls_ablation(df=df, tablefmt="github")
        markdown_output += "\n\n"

    table_filename = os.path.join(RESULT_OUTPUT, "result.md")
    with open(table_filename, "w") as f:
        f.write(markdown_output)
    logger.info(f"The tables of results are saved to {table_filename}")


if __name__ == "__main__":
    StrictFire(main)
