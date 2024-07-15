import json
import logging
import re
from bs4 import BeautifulSoup
from collections import defaultdict
from typing import Dict, List


# TODO: review the functions here
def extract_all_tagged_phrases(text: str) -> Dict[str, List[str]]:
    soup = BeautifulSoup(text, "html.parser")
    tagged_phrases = defaultdict(list)

    for tag in soup.find_all(True):
        if tag.name:
            # Clean and process the text
            full_text = " ".join(tag.stripped_strings)
            full_text = re.sub(r"\s+", " ", full_text.strip())
            full_text = re.sub(r'(?<!\\)\\(?!["\\])', r"\\\\", full_text)
            full_text = full_text.replace('"', '\\"')

            if full_text:  # Only add non-empty strings
                tagged_phrases[tag.name].append(full_text)

    # Remove duplicates while preserving order
    return {
        tag: list(dict.fromkeys(phrases)) for tag, phrases in tagged_phrases.items()
    }


def extract_prediction(schema: dict, prediction: str, kind: str = "json") -> dict:
    pred = {}
    if kind == "json":
        json_match = re.search(r"\{[\s\S]+\}", prediction)
        if json_match:
            json_str = json_match.group(0)
            json_str = re.sub(r"(\w+)-\$?\\?(\w+)\$?", r"\1-\2", json_str)
            json_str = json_str.replace('\\"', '"')
            json_str = re.sub(r'}\s*"', '}, "', json_str)
            json_str = re.sub(r']\s*"', '], "', json_str)
            try:
                pred = json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse JSON: {json_str}")
                logging.warning(f"Error: {str(e)}")

                try:
                    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
                    json_str = re.sub(r"(?<![\w'])'|'(?![\w'])", '"', json_str)
                    pred = json.loads(json_str)
                except json.JSONDecodeError:
                    logging.error(
                        f"Failed to parse JSON even after attempted fixes: {json_str}"
                    )
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
