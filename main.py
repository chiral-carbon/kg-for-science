import click
import json
import logging
import numpy as np
import os
import pprint
import random
import re
import torch
import string
import sys
import torch.nn.functional as F
import warnings

from datetime import datetime
from time import time
from tqdm import tqdm

from accelerate import infer_auto_device_map, init_empty_weights, Accelerator
from sklearn.metrics import accuracy_score, f1_score
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from src.processing.generate import (
    format_instance,
    get_sentences,
    generate_prefix,
    generate_instructions,
    generate_demonstrations,
    generate_prediction,
    # generate_prediction_forward_pass,
)
from src.processing.extractions import extract_all_tagged_phrases, extract_prediction
from src.eval.metrics import classify_predictions


# TODO: add batching to run data parallely
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

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.getenv("HF_TOKEN"),
    )

    model, tokenizer = accelerator.prepare(model, tokenizer)

    return model, tokenizer


@click.command()
@click.option(
    "--kind",
    default="json",
    help="Specify the kind of prompt input: json (default) or readable",
)
@click.option(
    "--runtype",
    type=click.Choice(["new", "eval"], case_sensitive=False),
    default="eval",
    help="Specify the type of run: new or eval (default)",
)
@click.option(
    "--data",
    default=None,
    help="Specify the directory of the data if running on new data",
)
def main(kind, runtype, data):

    # set up logging and save directories
    logger = logging.getLogger(__name__)

    uuid = "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(8)
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = "results"
    log_dir = "logs"
    few_shot = "random"
    out_dir_path = f"{runtype}_{few_shot}_{kind}_{uuid}_{timestamp}"
    os.makedirs(os.path.join(res_dir, out_dir_path), exist_ok=True)
    os.makedirs(os.path.join(res_dir, out_dir_path, log_dir), exist_ok=True)

    log_file = os.path.join(res_dir, out_dir_path, log_dir, f"log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    # set random seeds and environment variables
    logging.info("setting random seeds and environment variables...")
    warnings.filterwarnings("ignore")
    random.seed(0)
    np.random.seed(1)
    torch.manual_seed(2)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(3)
        torch.backends.cudnn.deterministic = True

    set_env_vars()

    logging.info("loading schema and data...")
    # load the schema
    with open("src/schema.json", "r") as f:
        schema = json.load(f)

    # load the data
    examples = []
    with open("data/human_annotations.jsonl", "r") as f:
        for line in f:
            examples.append(json.loads(line))

    train = examples[:3]
    valid = []
    if runtype == "new":
        valid = []
        for file in os.listdir(data):
            with open(os.path.join(data, file), "r") as f:
                for line in f:
                    valid.append(json.loads(line))
    else:
        valid = examples[3:]

    logging.info(f"Number of training examples: {len(train)}")
    logging.info(f"Number of validation examples: {len(valid)}")

    logging.info("loading model and tokenizer...")
    # load model and tokenizer
    model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    model, tokenizer = load_model_and_tokenizer(model_id)

    logging.info("generating base prompt...")
    # generate the prefix
    prefix = generate_prefix(
        instructions=generate_instructions(schema, kind),
        demonstrations=generate_demonstrations(train, kind),
    )

    logging.info("running the model...")
    logging.info(f"Run type: {runtype}")
    logging.info(f"Kind: {kind}")
    logging.info(f"Data: {data}")
    logging.info(f"Model: {model_id}")

    # run/evaluate the model
    if runtype == "eval":
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
        logging.info(f"#" * 50)
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

            if runtype == "eval":
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
        gold_tags.append(gold)
        predicted_responses.append(predicted_response)
        predicted_tags.append(pred)

        logging.info(f"Prompt:\n{prefix + input}\n")
        logging.info(f"True Tag:\n{gold}\n")
        logging.info(f"Predicted Response:\n{predicted_response}\n")
        logging.info(f"Predicted Tag:\n{pred}\n")

    # calculate metrics
    metrics = {}
    metrics["avg_pred_response_time_per_sentence"] = round(
        sum(pred_times) / len(pred_times), 4
    )
    metrics["total_time"] = round(running_time, 4)

    if runtype == "eval":
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
            (
                n_tp_union / (n_tp_union + n_fp_union)
                if (n_tp_union + n_fp_union) > 0
                else 0
            ),
            4,
        )
        union_recall = round(
            (
                n_tp_union / (n_tp_union + n_fn_union)
                if (n_tp_union + n_fn_union) > 0
                else 0
            ),
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

        metrics.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "union_precision": union_precision,
                "union_recall": union_recall,
                "union_f1": union_f1,
            }
        )

    pprint.pprint(metrics)

    # save the results
    if runtype == "eval":
        ground_truth_file = f"ground_truth.json"
        with open(os.path.join(res_dir, out_dir_path, ground_truth_file), "w") as f:
            json.dump({"gold_tags": gold_tags}, f, indent=4)
    else:
        gold_tags = [None for _ in predicted_tags]

    fname = "prompts.txt"
    with open(os.path.join(res_dir, out_dir_path, fname), "w") as f:
        for input, pred_response, pred_tag, gold_tag in zip(
            all_inputs, predicted_responses, predicted_tags, gold_tags
        ):
            f.write(f"{input}\n")
            f.write(f"Predicted Response: {pred_response}\n")
            f.write(f"Predicted Tag: {pred_tag}\n")
            f.write(f"True Tag: {gold_tag}\n")
            f.write("#" * 50 + "\n")

    pred_file = "predictions.json"
    with open(os.path.join(res_dir, out_dir_path, pred_file), "w") as f:
        json.dump({"predicted_tags": predicted_tags}, f, indent=4)

    pred_responses_file = "predicted_responses.txt"
    with open(
        os.path.join(res_dir, out_dir_path, pred_responses_file), "w", encoding="utf-8"
    ) as f:
        for response in predicted_responses:
            f.write(f"{response}\n")
            f.write("#" * 50 + "\n")

    mname = "metrics.json"
    with open(os.path.join(res_dir, out_dir_path, mname), "w") as f:
        json.dump({"metrics": metrics, "prompt_file": fname}, f, indent=4)

    logger.info(f"Results saved in: {os.path.join(res_dir, out_dir_path)}")


if __name__ == "__main__":
    main()
