import json
import random
import re

# import spacy
import torch

from typing import List, Dict, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from .extractions import extract_all_tagged_phrases

# nlp = spacy.load("en_core_web_sm")


def get_sentences(text: str) -> List[str]:
    # TODO: spacy splitting results in unequal lengths
    # doc = nlp(text)
    # sentences = [sent.text.strip() for sent in doc.sents]
    # sentences = [s for s in sentences if s]
    # return sentences
    # return sentences

    # return sentences

    return text.split(". ")


def format_instance(sentence: str, extraction: Union[str, None]) -> str:
    return "".join(
        [
            f"Sentence: {sentence}\n",
            (
                f"Extractions:\n{extraction}\n"
                if extraction is not None
                else f"Extractions:\n"
            ),
        ]
    )


def generate_instructions(schema: dict, kind: str = "json") -> str:
    instruction_parts = [
        "The following schema is provided to tag the title and abstract of a given scientific paper as shown in the examples:\n"
    ]
    if kind == "json":
        instruction_parts.append(f"{json.dumps(schema, indent=2)}\n\n")
    elif kind == "readable":
        readable_schema = ""
        for tag, description in schema.items():
            readable_schema += f"{tag}: {description}\n"
        instruction_parts.append(f"{readable_schema}\n")
    else:
        raise ValueError(f"Invalid kind: {kind}")

    return "".join(instruction_parts)


def generate_demonstrations(examples: List[dict], kind: str = "json") -> str:
    demonstration_parts = []
    for example in random.sample(examples, k=3):
        sentences = get_sentences(example["abstract"])
        tagged_sentences = get_sentences(example["tagged_abstract"])

        for sentence, tagged_sentence in random.sample(
            list(zip(sentences, tagged_sentences, strict=True)), k=3
        ):
            tag_to_phrase = extract_all_tagged_phrases(tagged_sentence)
            if kind == "json":
                extractions = f"{json.dumps(tag_to_phrase, indent=2)}\n"
            elif kind == "readable":
                extractions = "".join(
                    f"{tag}: {', '.join(phrase)}\n"
                    for tag, phrase in tag_to_phrase.items()
                )
            else:
                raise ValueError(f"Invalid kind: {kind}")

            demonstration_parts.append(format_instance(sentence, extractions))

    return "".join(demonstration_parts)


def generate_prefix(instructions: str, demonstrations: str) -> str:
    return f"{instructions}" f"{demonstrations}"


def generate_prediction(model, tokenizer, prefix: str, input: str, kind: str) -> str:
    prompt = prefix + input
    messages = [
        {
            "role": "system",
            "content": "You are an assistant who tags papers according to given schema and only returns the tagged phrases in the format as provided in the examples without repeating anything else.",
        },
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        # add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=1200,
        eos_token_id=terminators,
        # num_beams=8,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1] :]
    prediction_response = tokenizer.decode(response, skip_special_tokens=True)

    return prediction_response
