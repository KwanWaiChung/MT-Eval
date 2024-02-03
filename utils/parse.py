from typing import Tuple, List
from enum import Enum


def parse_gpt4_evaluation(s: str) -> Tuple[int, int]:
    """Parse the coherence and helpfulness score.

    Args:
        s (str): The generated response

    Returns:
        Tuple[int,int]: The coherence and helpfulness score.
            If not able to parse, return None for that score.
    """
    scores = [None, None]
    for row in s.split("\n"):
        if "Score:" in row:
            if "coherence" in row.lower():
                index = 0
            elif "helpfulness" in row.lower():
                index = 1
            else:
                print("Unable to parse coherence or helpfulness:", row)
                continue
            row = row.split("Score:")[1]
            if "**" in row:
                row = row.split("**")[1]
            try:
                score = int(row.strip())
            except ValueError:
                print("Unable to parse into int:", row)
                continue
            else:
                scores[index] = score
    return tuple(scores)


def parse_construct_relation_output(s: str) -> List[Tuple[str, str, str]]:
    """Parse the constructed relation from response.

    Args:
        s (str): The response.

    Returns:
        List[Tuple[str,str,str]]: List of relation tuples.
            (entity 1, relation, entity 2)

    """
    relations = []
    for relation in s.split("\n"):
        tuple_str = relation.split(": ")[1].strip()
        tuple_split = tuple_str.strip("()").split(", ")
        assert len(tuple_split) == 3, tuple_split
        relations.append(tuple(tuple_split))
    return relations


def parse_construct_qa_output(s: str) -> List[Tuple[str, str]]:
    """Parse the constructed qa pairs from response.

    Args:
        s (str): The response.

    Returns:
        List[Tuple[str,str,str]]: List of (q, a).

    """
    qa_pairs = s.split("\n\n")
    qas = []
    for p in qa_pairs:
        q, a = p.split("\n")
        q = q.split(": ")[1].strip()
        a = a.split(": ")[1].strip()
        qas.append((q, a))
    return qas


def parse_construct_summary_output(s: str) -> str:
    """Parse the constructed summary from response.

    Args:
        s (str): The response.

    Returns:
        str: The summary.

    """
    summary = s.split(": ")[1]
    return summary


def parse_construct_translation_output(s: str) -> List[Tuple[str, str]]:
    """Parse the constructed translation pairs from response.

    Args:
        s (str): The response.

    Returns:
        List[Tuple[str,str]]: List of (instruction, translation).

    """
    pairs = s.split("\n\n")
    output_pairs = []
    for p in pairs:
        q, a = p.split("\n")
        q = q.split(": ")[1].strip()
        a = a.split(": ")[1].strip()
        output_pairs.append((q, a))
    return output_pairs


def parse_construct_ner_pos_output(s: str) -> List[Tuple[str, str, List[str]]]:
    """Parse the constructed ner/pos pairs from response.

    Args:
        s (str): The response.

    Returns:
        List[Tuple[str,str,str]]: List of (type, instruction, [answers]).

    """
    pairs = s.split("\n\n")
    output_pairs = []
    for p in pairs:
        t, q, a = p.split("\n")
        t = t.split(": ")[1].strip()
        q = q.split(": ")[1].strip()
        a = a.strip(".").split(": ")[1].strip()
        a = a.split(", ")
        output_pairs.append((t, q, a))
    return output_pairs


def parse_construct_paragraph_output(s: str) -> str:
    title, content = s.split("\n\n", maxsplit=1)
    title = title.split(":")[1].strip("* ")
    content = content.strip()
    return title, content


def parse_ner_pos_output(s: str) -> List[str]:
    return s.split(", ")


def parse_rel_output(s: str) -> List[Tuple[str, str, str]]:
    class Status(Enum):
        START = 1
        MIDDLE = 2

    status = Status.START
    output = []
    buffer = ""
    cur_tuple = []
    i = 0
    again = False
    while i < len(s):
        char = s[i]
        if char == "(" and status == Status.START:
            status = Status.MIDDLE
        elif (
            char == "," and status == Status.MIDDLE and buffer.strip()
        ):  # at least a word
            cur_tuple.append(buffer.strip())
            buffer = ""
        elif char == ")" and status == Status.MIDDLE and len(cur_tuple) == 2:
            cur_tuple.append(buffer.strip())
            output.append(tuple(cur_tuple))
            cur_tuple = []
            status = Status.START
        elif char not in ["(", ")"] and status == Status.MIDDLE:
            buffer += char
        elif again:  # second time ilegal
            again = False
        elif not again:  # first time illegal output
            buffer = ""
            cur_tuple = []
            status = Status.START
            again = True
            continue
        i += 1
    return output
