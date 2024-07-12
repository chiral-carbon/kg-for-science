import json
import re
from bs4 import BeautifulSoup
from collections import defaultdict
from typing import Dict, List


# TODO: review this function
def extract_all_tagged_phrases(text: str) -> Dict[str, List[str]]:
    soup = BeautifulSoup(text, "html.parser")
    tagged_phrases = defaultdict(list)

    # Recursive function to extract text from nested tags
    def extract_text(tag):
        if tag.name:
            full_text = " ".join(tag.stripped_strings)
            tagged_phrases[tag.name].append(full_text)
            # Recursively process all children tags
            for child in tag.find_all(True):
                extract_text(child)

    for tag in soup.find_all(True):
        extract_text(tag)

    for tag in tagged_phrases:
        tagged_phrases[tag] = list(dict.fromkeys(tagged_phrases[tag]))

    return dict(tagged_phrases)


def extract_prediction(schema: dict, prediction: str, kind: str = "json") -> dict:
    pred = {}
    if kind == "json":
        json_match = re.search(r"\{[^}]+\}", prediction)
        if json_match:
            # TODO: Replace single quotes with double quotes in prompt and remove code below.
            json_str = json_match.group(0)
            json_str = re.sub(r"(?<![\w'])'|'(?![\w'])", '"', json_str)
            json_str = re.sub(r'}\s*"', '}, "', json_str)
            json_str = re.sub(r']\s*"', '], "', json_str)
            try:
                pred = json.loads(json_str)
            except json.JSONDecodeError as e:
                # TODO: Use the warning module here.
                print(f"Failed to parse JSON: {json_str}")
                print(f"Error: {str(e)}")
    elif kind == "readable":
        match = re.findall(
            rf'^({"|".join(list(schema.keys()))}): (.+)$',
            prediction,
            flags=re.MULTILINE,
        )
        pred = {tag: values.split(", ") for tag, values in match}
    else:
        raise ValueError(f"Invalid kind: {kind}")

    return pred
