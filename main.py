import json
import numpy as np
import os
import pprint
import random
import re
import torch
import string
import torch.nn.functional as F
import warnings

from time import time
from tqdm import tqdm

from src.processing.generate import (
    format_instance,
    get_sentences,
    generate_prefix,
    generate_instructions,
    generate_demonstrations,
    generate_prediction,
)
from src.processing.extractions import extract_all_tagged_phrases, extract_prediction
from src.eval.metrics import classify_predictions

from accelerate import infer_auto_device_map, init_empty_weights, Accelerator
from sklearn.metrics import accuracy_score, f1_score
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


# constants
kind = "readable"


# helper functions
def set_env_vars(fname="../access_keys.json"):
    with open(fname) as f:
        keys = json.load(f)
        for key in keys:
            if key not in os.environ.keys():
                os.environ[key.upper()] = keys[key]


def load_model_and_tokenizer(model_id: str):
    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    # device_map = infer_auto_device_map(model, max_memory=max_memory)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.getenv("HF_TOKEN"),
    )

    model = accelerator.prepare(model)

    return model, tokenizer


def main():
    # set random seeds
    warnings.filterwarnings("ignore")
    random.seed(0)
    np.random.seed(1)
    torch.manual_seed(2)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(3)
        torch.backends.cudnn.deterministic = True

    set_env_vars()

    # load the data
    examples = []
    with open("data/examples.jsonl", "r") as f:
        for line in f:
            examples.append(json.loads(line))

    with open("schema.json", "r") as f:
        schema = json.load(f)

    # load model and tokenizer
    model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    model, tokenizer = load_model_and_tokenizer(model_id)

    # split the data
    train = examples[:3]
    valid = examples[3:]

    # generate the prefix
    prefix = generate_prefix(
        instructions=generate_instructions(schema, kind),
        demonstrations=generate_demonstrations(train, kind),
    )

    # evaluate the model
    n_tp = 0
    n_fp = 0
    n_fn = 0
    n_tp_union = 0
    n_fp_union = 0
    n_fn_union = 0
    running_time = 0
    pred_times = []
    all_inputs = []
    predicted_responses = []
    gold_tags = []
    predicted_tags = []
    for example in tqdm(valid):
        abstract = example["abstract"]
        tagged_abstract = example["tagged_abstract"]
        for sentence, tagged_sentence in tqdm(
            zip(get_sentences(abstract), get_sentences(tagged_abstract), strict=True)
        ):
            input = format_instance(sentence, extraction=None)
            s_time = time()
            predicted_response = generate_prediction(
                model, tokenizer, prefix, input, kind
            )
            e_time = time()
            pred = extract_prediction(schema, predicted_response, kind=kind)
            gold = extract_all_tagged_phrases(tagged_sentence)

            tp, fp, fn = classify_predictions(gold, pred)
            n_tp += tp
            n_fp += fp
            n_fn += fn
            utp, ufp, ufn = classify_predictions(gold, pred, union=True)
            n_tp_union += utp
            n_fp_union += ufp
            n_fn_union += ufn

            running_time += time() - s_time
            pred_times.append(e_time - s_time)

            all_inputs.append(input)
            predicted_responses.append(predicted_response)
            predicted_tags.append(pred)
            gold_tags.append(gold)

    # calculate metrics
    precision = round(n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0, 4)
    recall = round(n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0, 4)
    f1 = round(
        (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        ),
        4,
    )
    union_precision = round(
        n_tp_union / (n_tp_union + n_fp_union) if (n_tp_union + n_fp_union) > 0 else 0,
        4,
    )
    union_recall = round(
        n_tp_union / (n_tp_union + n_fn_union) if (n_tp_union + n_fn_union) > 0 else 0,
        4,
    )
    union_f1 = round(
        (
            2 * (union_precision * union_recall) / (union_precision + union_recall)
            if (union_precision + union_recall) > 0
            else 0
        ),
        4,
    )
    avg_time = round(sum(pred_times) / len(pred_times), 4)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "union_precision": union_precision,
        "union_recall": union_recall,
        "union_f1": union_f1,
        "avg_time_per_sentence": avg_time,
        "total_time": round(running_time, 4),
    }

    pprint.pprint(metrics)

    # save the results
    uuid = "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(8)
    )
    res_dir = "results"
    few_shot = "random"
    fname = f"prompt_{few_shot}_{kind}_{uuid}.txt"

    with open(os.path.join(res_dir, fname), "w", encoding="utf-8") as f:
        for input, predicted_response, gold_tag, pred_tag in zip(
            all_inputs, predicted_responses, gold_tags, predicted_tags
        ):
            f.write(f"Prompt:\n{prefix + input}\n")
            f.write(f"True Tag:\n{gold_tag}\n")
            f.write(f"Predicted Response:\n{predicted_response}\n")
            f.write(f"Predicted Tag:\n{pred_tag}\n")
            f.write("#" * 50 + "\n")

    ground_truth_file = f"ground_truth_{few_shot}_{kind}_{uuid}.json"
    with open(os.path.join(res_dir, ground_truth_file), "w") as f:
        json.dump({"gold_tags": gold_tags}, f, indent=4)

    pred_file = f"predictions_{few_shot}_{kind}_{uuid}.json"
    with open(os.path.join(res_dir, pred_file), "w") as f:
        json.dump({"predicted_tags": predicted_tags}, f, indent=4)

    pred_responses_file = f"predicted_responses_{few_shot}_{kind}_{uuid}.txt"
    with open(os.path.join(res_dir, pred_responses_file), "w", encoding="utf-8") as f:
        for response in predicted_responses:
            f.write(f"{response}\n")
            f.write("#" * 50 + "\n")

    mname = f"metrics_{few_shot}_{kind}_{uuid}.json"
    with open(os.path.join(res_dir, mname), "w") as f:
        json.dump({"metrics": metrics, "prompt_file": fname}, f, indent=4)


if __name__ == "__main__":
    main()
