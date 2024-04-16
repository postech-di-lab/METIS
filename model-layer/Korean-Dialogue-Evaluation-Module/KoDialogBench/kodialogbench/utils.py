from typing import Dict, List, Tuple, Union

SPEAKER_PREFIX = "화자"
TURN_SEPERATOR = "\n"


def doc_to_text_direct_prompting(
    doc: Dict[str, Union[List[Tuple[str, str]], str, List[str], int]]
) -> str:
    """
    Convert an example to a query prompt for direct prompting.

    :param doc: example from the dataset
    Structure: {
        "dialogue": list of (speaker_id, utterance),
        "question": str,
        "options": list of str,
        "answer_idx": int
    }
    :return: query prompt
    """
    dialogue_str = TURN_SEPERATOR.join(
        [
            f"{SPEAKER_PREFIX}{speaker_id}: {utterance}"
            for speaker_id, utterance in doc["dialogue"]
        ]
    )
    query_prompt = f"{dialogue_str}\n\n질문: {doc['question']}\n정답:"

    return query_prompt


def doc_to_text_direct_prompting_with_description(
    doc: Dict[str, Union[List[Tuple[str, str]], str, List[str], int]]
) -> str:
    """
    Convert an example to a query prompt for direct prompting with option description.

    :param doc: example from the dataset
    Structure: {
        "dialogue": list of (speaker_id, utterance),
        "option_description": list of (option, description),
        "question": str,
        "options": list of str,
        "answer_idx": int
    }
    :return: query prompt
    """
    dialogue_str = TURN_SEPERATOR.join(
        [
            f"{SPEAKER_PREFIX}{speaker_id}: {utterance}"
            for speaker_id, utterance in doc["dialogue"]
        ]
    )
    description_str = "\n".join(
        [
            f"{option}: {description}"
            for option, description in doc["option_description"]
        ]
    )
    query_prompt = f"[대화]\n{dialogue_str}\n\n[보기]\n{description_str}\n\n질문: {doc['question']}\n정답:"

    return query_prompt


def doc_to_text_option_prompting(
    doc: Dict[str, Union[List[Tuple[str, str]], str, List[str], int]]
) -> str:
    """
    Convert an example to a query prompt for option prompting.

    :param doc: example from the dataset
    Structure: {
        "dialogue": list of (speaker_id, utterance),
        "question": str,
        "options": list of str,
        "answer_idx": int
    }
    :return: query prompt
    """
    dialogue_str = TURN_SEPERATOR.join(
        [
            f"{SPEAKER_PREFIX}{speaker_id}: {utterance}"
            for speaker_id, utterance in doc["dialogue"]
        ]
    )
    description_str = "\n".join(
        [f"{i+1}) {option}" for i, option in enumerate(doc["options"])]
    )
    query_prompt = f"[대화]\n{dialogue_str}\n\n[보기]\n{description_str}\n\n질문: {doc['question']}\n정답:"

    return query_prompt


def doc_to_text_response_selection_prompting(
    doc: Dict[str, Union[List[Tuple[str, str]], str, List[str], int]]
) -> str:
    """
    Convert an example to a query prompt for response selection prompting.

    :param doc: example from the dataset
    Structure: {
        "dialogue": list of (speaker_id, utterance),
        "target_speaker": str,
        "options": list of str,
        "answer_idx": int
    }
    :return: query prompt
    """
    dialogue_str = TURN_SEPERATOR.join(
        [
            f"{SPEAKER_PREFIX}{speaker_id}: {utterance}"
            for speaker_id, utterance in doc["dialogue"]
        ]
    )
    query_prompt = f"{dialogue_str}{TURN_SEPERATOR}{SPEAKER_PREFIX}{doc['target_speaker']}:"

    return query_prompt