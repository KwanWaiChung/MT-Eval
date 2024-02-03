import json
import os
import random
from typing import Dict
from copy import deepcopy
from utils.constants import topics
from utils.global_inst import INSTRUCTIONS


def create_refinement_single():
    # n_doc * 5 turns (excluded the first turn.)
    # id: {doc_i}_{task}_{turn_i}
    # turn_i: [1, 5]
    # doc_i: [1, 10]
    # task: [rewrite, ner, qa, sum]

    insts = [
        json.loads(row)
        for row in open("raw_data/refinement_single_inst.jsonl")
    ]

    prompts = []
    for row in insts:
        row = row[0]
        inst = row["user"]
        doc_i = int(row["id"].split("_")[0]) - 1
        content = documents[doc_i]["gen_resp"]
        query = f"Content: {content}\n\nInstruction: {inst}"
        prompts.append({
            "conv": [
                {
                    "user": query,
                    "sys": "",
                    "id": row["id"],
                    "do_inference": True,
                    "inst": inst,
                },
            ],
            "id": f"{row['id']}",
        })
    assert len(prompts) == 200
    p = "data/refinement_single.jsonl"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(
            "\n".join([json.dumps(row, ensure_ascii=False) for row in prompts])
        )
    # id: {session_i}_{doc_i}_{task}_{turn_i}
    print(f"Constructed `Refinement Single` in {p}")


def create_refinement_multi():
    prompts = []
    # n_docs=10, num_turns=12, num_task_combinations = 4
    # num_rows = n_docs * num_turns * num_task_combination

    data = [
        json.loads(row) for row in open("raw_data/refinement_multi_inst.jsonl")
    ]

    instance_tasks = {}
    seeder = random.Random(111)
    num_task_per_session = 2
    for session_i, session in enumerate(data):
        instance_i, task_type, _ = session[0]["id"].split("_")
        if task_type in ["trans", "short-qa", "pos", "rel"]:
            continue
        instance_tasks.setdefault(instance_i, {}).setdefault(task_type, [])
        for turn in session:
            instance_tasks[instance_i][task_type].append({
                "user": turn["user"],
                "sys": turn["sys"],
                "id": turn["id"],
            })
    session_i = 1
    for instance_i, task_types_dict in instance_tasks.items():
        task_order = [t for t in task_types_dict]
        seeder.shuffle(task_order)

        all_task_types = [[t] for t in task_order]
        for _ in range(num_task_per_session - 1):
            task_order.append(task_order.pop(0))
            for tmp_i, candidate_task_types in enumerate(all_task_types):
                candidate_task_types.append(task_order[tmp_i])

        for task_types in all_task_types:
            turns = []
            session_id = str(session_i)
            for task_type in task_types:
                session_id += f"_{task_type}"
                for instance_turn_i, turn in enumerate(
                    task_types_dict[task_type]
                ):
                    query = turn["user"]
                    if instance_turn_i == 0 and turns:
                        query = query.split("Instruction: ")[1]
                    # _id: {session_i}_{doc_i}_{type}_{turn_i}
                    _id = turn["id"]
                    turns.append({
                        "user": query,
                        "sys": turn["sys"],
                        "id": _id,
                        "do_inference": True,
                        "inst": query.split("Instruction: ")[-1],
                    })
            prompts.append({
                "conv": turns,
                "id": session_id,
            })
            session_i += 1
    assert len(prompts) == 40
    assert sum(len(p["conv"]) for p in prompts) == 480

    p = "data/refinement_multi.jsonl"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(
            "\n".join([json.dumps(row, ensure_ascii=False) for row in prompts])
        )
    print(f"Constructed `Refinement Multi` in {p}")


def create_recollection_cls():
    seeder = random.Random(111)
    labels = "\n".join([f"{label}" for i, label in enumerate(topics, 1)])
    instruction = f"""You are given one document at each turn where each of them belongs to one of the following categories: 
{labels}

You task is to classify the document. You should only output the category and nothing else. Reply OK if you understand my instructions."""
    single_turn_instruction = f"""You are given a document below which belongs to one of the following categories: 
{labels}

You task is to classify the document. You should only output the category and nothing else.
Document: {{document}}
Category:"""

    single_prompts = []
    multi_prompts = []
    all_articles = []
    n_session = 10
    n_articles_per_session = 10
    for i, document in enumerate(documents, 1):
        content = document["gen_resp"]
        content = "\n\n".join(content.split("\n\n")[:-3])
        topic = document["topic"]
        all_articles.append({"content": content, "topic": topic, "id": i})
    seeder.shuffle(all_articles)
    for session_i in range(n_session):
        sampled_articles = all_articles[
            session_i
            * n_articles_per_session : (session_i + 1)
            * n_articles_per_session
        ]
        # single-turn
        for turn_i, document in enumerate(sampled_articles):
            content = document["content"]
            while "\n\n" in content:
                content = content.replace("\n\n", "\n")
            _id = f"{session_i+1}#{document['id']}"
            single_prompts.append({
                "conv": [{
                    "user": single_turn_instruction.format(document=content),
                    "sys": document["topic"],
                    "id": _id,
                    "do_inference": True,
                }],
                "id": _id,
            })

        # multi-turn
        turns = [{
            "user": instruction,
            "sys": "OK",
            "id": "instruction",
            "do_inference": False,
        }]
        # for turn_i, row in enumerate(tqdm(sampled_rows)):
        for turn_i, document in enumerate(sampled_articles, 1):
            content = document["content"]
            while "\n\n" in content:
                content = content.replace("\n\n", "\n")
            answer = document["topic"]
            _id = f"{document['id']}"
            turns.append({
                "user": content,
                "sys": answer,
                "id": _id,
                "do_inference": True,
            })
        multi_prompts.append({
            "conv": turns,
            "id": str(session_i + 1),
        })

    p = "data/recollection_single_cls.jsonl"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in single_prompts]
            )
        )
    print(f"Constructed `Recollection Single (CLS)` in {p}")

    p = "data/recollection_multi_cls.jsonl"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in multi_prompts]
            )
        )
    print(f"Constructed `Recollection Multi (CLS)` in {p}")


def create_recollection_global_inst():
    data = [json.loads(row) for row in open("raw_data/global_inst.jsonl")]
    inst_to_row = {}
    for row in data:
        for inst in row["instructions"]:
            if inst not in INSTRUCTIONS:
                continue
            inst_to_row.setdefault(inst, []).append(row)

    num_session_per_inst = 2
    num_turn_per_session = 10
    single_prompts = []
    multi_prompts = []
    seeder = random.Random(111)
    count = 0
    session_i = 1
    for inst_i, (inst, rows) in enumerate(inst_to_row.items(), 1):
        for _ in range(num_session_per_inst):
            inst_obj = INSTRUCTIONS[inst]()
            inst_prompt = inst_obj.get_prompt()
            if inst != "length_constraints:number_words":
                inst_prompt += " Keep all your responses under 200 words."

            turns = [{
                "user": inst_prompt,
                "sys": "ok.",
                "id": "instruction",
                "do_inference": False,
            }]
            for turn_i in range(num_turn_per_session):
                instance = seeder.choice(rows)
                rows.pop(rows.index(instance))
                # _id = f"{inst_i}-{instance['id']}_{turn_i+1}"
                # _id = f"{session_i}_{_id}_{turn_i+1}"
                _id = f"{inst_i}-{instance['id']}"

                turns.append({
                    "user": instance["query"],
                    "sys": "",
                    "id": _id,  # {inst_cat_i}-{inst_i}_{turn_i}
                    "do_inference": True,
                })
                single_id = f"{session_i}_{_id}"
                single_prompts.append({
                    "inst_name": inst,
                    "inst_args": inst_obj.get_args(),
                    "conv": [{
                        "user": (
                            f"Instruction: {inst_prompt}\nQuestion:"
                            f" {instance['query']}"
                        ),
                        "sys": "",
                        "id": single_id,
                        "do_inference": True,
                    }],
                    "id": single_id,
                })
                count += 1
            _id = f"{session_i}_{_id}_{turn_i+1}"
            multi_prompts.append({
                "inst_name": inst,
                "inst_args": inst_obj.get_args(),
                "conv": turns,
                "id": str(session_i),
            })
            session_i += 1

    p = "data/recollection_single_global-inst.jsonl"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in single_prompts]
            )
        )
    print(f"Constructed `Recollection Single (Global Inst)` in {p}")

    p = "data/recollection_multi_global-inst.jsonl"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in multi_prompts]
            )
        )
    print(f"Constructed `Recollection Multi (Global Inst)` in {p}")


def create_follow_up_multi():
    data = [
        json.loads(row) for row in open("raw_data/mt-bench_extended.jsonl")
    ]
    multi_prompts = []
    for row in data:
        turns = []
        # We don't test the first two turns.
        for turn in row["conversation"][:2]:
            inst_id, turn_i = turn["id"].split("_")
            _id = f"{inst_id}_{int(turn_i)+1}"
            turns.append({
                "user": turn["user"],
                "sys": turn["sys"],
                "id": str(int(turn_i) + 1),
                "do_inference": False,
            })
        # Test the last three turns
        for turn in row["conversation"][2:]:
            inst_id, turn_i = turn["id"].split("_")
            turns.append({
                "user": turn["user"],
                "sys": turn["sys"],
                "id": str(int(turn_i) + 1),
                "do_inference": True,
            })
        multi_prompts.append({
            "conv": turns,
            "id": str(inst_id),
        })

    # multi check
    assert sum([len(row["conv"]) for row in multi_prompts]) == 80 * 5
    assert (
        sum([
            len([turn for turn in row["conv"] if turn["do_inference"]])
            for row in multi_prompts
        ])
        == 80 * 3
    )
    p = "data/follow-up_multi.jsonl"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in multi_prompts]
            )
        )
    print(f"Constructed `Follow-up Multi` in {p}")


def create_expansion():
    data = [
        json.loads(row) for row in open("raw_data/refinement_multi_inst.jsonl")
    ]
    cls_instruct = (
        "Instruction: Classify the initially provided content into one of the"
        f" following labels: {', '.join(topics)}. Just provide the correct"
        " label without any further explanations or extra output."
    )
    instance_tasks = {}
    seeder = random.Random(111)
    for session_i, session in enumerate(data):
        instance_i, task_type, _ = session[0]["id"].split("_")
        instance_tasks.setdefault(instance_i, {}).setdefault(task_type, [])
        turn = session[0]
        instance_tasks[instance_i][task_type] = {
            "user": turn["user"],
            "sys": turn["sys"],
            "id": turn["id"],
        }
    # {session_i}_{instance_i}_{type}_{turn_i}
    session_i = 1
    single_prompts = []
    multi_prompts = []
    for instance_i, task_types in instance_tasks.items():
        task_types.pop("qa")
        task_types["qa"] = task_types.pop("short-qa")
        task_types["cls"] = {
            "user": task_types["qa"]["user"].split("Instruction")[0]
            + cls_instruct,
            "sys": documents[int(instance_i) - 1]["topic"],
            "id": f"{instance_i}_cls_1",
        }
        task_order = [k for k in task_types.keys() if k not in ["rewrite"]]
        seeder.shuffle(task_order)

        turns = []
        for turn_i, task_type in enumerate(task_order):
            task: Dict = task_types[task_type]
            query = task["user"]
            if turn_i > 0:
                query = query.split("Instruction: ")[1]

            turns.append({
                "user": query,
                "sys": task["sys"],
                "id": task["id"],
                "do_inference": True,
                "inst": query.split("Instruction: ")[-1],
            })
            single_id = f"{session_i}#{task['id']}"
            single_prompts.append({
                "conv": [{
                    "user": task["user"],
                    "sys": task["sys"],
                    "id": single_id,
                    "do_inference": True,
                    "inst": query.split("Instruction: ")[-1],
                }],
                "id": single_id,
            })
        multi_prompts.append({
            "conv": turns,
            "id": f"{session_i}",
        })
        session_i += 1

    # check
    assert sum([len(row["conv"]) for row in multi_prompts]) == 70
    assert len(single_prompts) == 70

    p = "data/expansion_single.jsonl"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in single_prompts]
            )
        )
    print(f"Constructed `Expansion Single` in {p}")

    p = "data/expansion_multi.jsonl"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in multi_prompts]
            )
        )
    print(f"Constructed `Expansion Multi` in {p}")


def create_recollection_cls_ablation():
    seeder = random.Random(111)
    seeders = [random.Random(seeder.randint(1, 10000)) for _ in range(10)]
    labels = "\n".join([f"{label}" for i, label in enumerate(topics, 1)])
    instruction = f"""You are given one document at each turn where each of them belongs to one of the following categories: 
    {labels}

    You task is to classify the document. You should only output the category and nothing else. Reply OK if you understand my instructions."""

    # For each n_article setting, we create n_session
    n_session = 50
    n_articles = [5, 10]
    all_articles = []
    topic_articles = {}
    for i, article in enumerate(documents, 1):
        doc = "\n".join(article["gen_resp"].split("\n\n")[:-3])
        topic = article["topic"]
        topic_articles.setdefault(topic, []).append(
            {"content": doc, "id": i, "topic": topic}
        )
        all_articles.append({"content": doc, "id": i, "topic": topic})

    def get_prompts(sampled_articles):
        turns = [{
            "user": instruction,
            "sys": "ok.",
            "id": "instruction",
            "do_inference": False,
        }]
        for article in sampled_articles:
            content = article["content"]
            _id = f"{article['id']}"
            turns.append({
                "user": content,
                "sys": article["topic"],
                "id": _id,
                "do_inference": False,
            })
        turns[-1]["do_inference"] = True
        return {
            "conv": turns,
            "id": f"{session_i+1}_{_id}",
        }

    gold_prompts = []
    dgc_prompts = []
    sgc_prompts = []
    rc_prompts = []
    for n_article in n_articles:
        for session_i in range(n_session):
            # sample random article
            article = seeder.choice(all_articles)
            topic = article["topic"]

            # 0. Gold
            all_articles_copy = all_articles.copy()
            all_articles_copy.pop(all_articles_copy.index(article))
            context_articles = seeders[0].choices(
                all_articles_copy, k=n_article - 1
            )
            sampled_articles = context_articles + [article]
            rc_sampled_articles = deepcopy(
                sampled_articles
            )  # also used in Random Class
            gold_prompts.append(get_prompts(sampled_articles))

            # 1. Diverse Gold Class
            all_articles_copy = [
                a for a in all_articles if a["topic"] != article["topic"]
            ]
            context_articles = seeders[2].choices(
                all_articles_copy, k=n_article - 1
            )
            sampled_articles = context_articles + [article]
            dgc_prompts.append(get_prompts(sampled_articles))

            # 2. Same Gold Class
            legit_topics = [
                topic
                for topic in topic_articles
                if len(topic_articles[topic]) >= n_article
            ]
            topic = seeders[1].choice(legit_topics)
            context_articles = seeders[1].choices(
                topic_articles[topic], k=n_article - 1
            )
            sampled_articles = context_articles + [article]
            sgc_prompts.append(get_prompts(sampled_articles))

            # 3. Ramdom Class
            for a in rc_sampled_articles[:-1]:
                cur_topics = topics.copy()
                cur_topics.pop(cur_topics.index(a["topic"]))
                a["topic"] = seeders[3].choice(cur_topics)
            rc_prompts.append(get_prompts(rc_sampled_articles))

    p = "data/cls_ablation_gold.jsonl"
    with open(p, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in gold_prompts]
            )
        )
    print(f"Constructed `Recollection Multi (CLS Gold)` in {p}")

    p = "data/cls_ablation_dgc.jsonl"
    with open(p, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in dgc_prompts]
            )
        )
    print(f"Constructed `Recollection Multi (CLS DGC)` in {p}")

    p = "data/cls_ablation_sgc.jsonl"
    with open(p, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in sgc_prompts]
            )
        )
    print(f"Constructed `Recollection Multi (CLS SGC)` in {p}")

    p = "data/cls_ablation_rc.jsonl"
    with open(p, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in rc_prompts]
            )
        )
    print(f"Constructed `Recollection Multi (CLS RC)` in {p}")


if __name__ == "__main__":
    documents = [json.loads(row) for row in open("raw_data/documents.jsonl")]
    # Trim the topic.
    for doc in documents:
        doc["gen_resp"] = doc["gen_resp"].split("\n\n", 1)[1]
    create_refinement_single()
    create_refinement_multi()
    create_recollection_cls()
    create_recollection_global_inst()
    create_follow_up_multi()
    create_expansion()
    create_recollection_cls_ablation()
