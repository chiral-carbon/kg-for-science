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
import torch.nn.functional as F

# import wandb
import warnings

from collections import defaultdict
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
)
from src.processing.extractions import extract_all_tagged_phrases, extract_prediction
from src.eval.metrics import classify_predictions, compute_metrics
from src.utils.utils import load_model_and_tokenizer, save_results, set_env_vars


# TODO: add sweep configuration and run sweeps
# TODO: add batching to run data parallely using DDP after committing single data run
SAVE_INTERVAL = 10
RES_DIR = "results"
LOG_DIR = "logs"


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

    few_shot = "random"
    uuid = "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(8)
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_path = f"{runtype}_{few_shot}_{kind}_{uuid}_{timestamp}"
    os.makedirs(os.path.join(RES_DIR, out_dir_path), exist_ok=True)
    os.makedirs(os.path.join(RES_DIR, out_dir_path, LOG_DIR), exist_ok=True)

    log_file = os.path.join(RES_DIR, out_dir_path, LOG_DIR, f"log.txt")
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

    # load the schema
    logging.info("loading schema and data...")
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
        seen = set()
        for file in os.listdir(data):
            with open(os.path.join(data, file), "r") as f:
                for line in f:
                    dict_line = json.loads(line)
                    if dict_line["title"] not in seen:
                        seen.add(dict_line["title"])
                        valid.append(dict_line)
                    else:
                        logging.info(f"Duplicate found: {dict_line}")
    else:
        valid = examples[3:]

    logging.info(f"Number of training examples: {len(train)}")
    logging.info(f"Number of validation examples: {len(valid)}")

    # load model and tokenizer
    logging.info("loading model and tokenizer...")
    model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    model, tokenizer = load_model_and_tokenizer(model_id)

    # generate the prefix
    logging.info("generating base prompt...")
    prefix = generate_prefix(
        instructions=generate_instructions(schema, kind),
        demonstrations=generate_demonstrations(train, kind),
    )

    # run/evaluate the model
    logging.info("running the model...")
    logging.info(f"Run type: {runtype}")
    logging.info(f"Kind: {kind}")
    logging.info(f"Data: {data}")
    logging.info(f"Model: {model_id}")

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
    for i, example in enumerate(tqdm(valid)):
        logging.info(f"#" * 50)
        abstract = example["title"] + ". " + example["abstract"]
        sentences = get_sentences(abstract)
        if runtype == "eval":
            tagged_abstract = (
                example["tagged_title"] + ". " + example["tagged_abstract"]
            )
            tagged_sentences = get_sentences(tagged_abstract)
            zipped = zip(sentences, tagged_sentences, strict=True)
        else:
            zipped = zip(sentences, [None for _ in sentences], strict=True)

        for sentence, tagged_sentence in tqdm(zipped):
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
            else:
                gold = None

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

        if (i + 1) % SAVE_INTERVAL == 0:
            if runtype == "eval":
                metrics = compute_metrics(
                    running_time,
                    pred_times,
                    runtype,
                    eval_metrics=(n_tp, n_fp, n_fn, n_tp_union, n_fp_union, n_fn_union),
                )
            else:
                metrics = compute_metrics(running_time, pred_times, runtype)

            save_results(
                out_dir_path,
                all_inputs,
                gold_tags,
                predicted_responses,
                predicted_tags,
                metrics,
                append=(i > 0),
            )

    if i == len(valid) - 1:
        if runtype == "eval":
            metrics = compute_metrics(
                running_time,
                pred_times,
                runtype,
                eval_metrics=(n_tp, n_fp, n_fn, n_tp_union, n_fp_union, n_fn_union),
            )
        else:
            metrics = compute_metrics(running_time, pred_times, runtype)

        save_results(
            out_dir_path,
            all_inputs,
            gold_tags,
            predicted_responses,
            predicted_tags,
            metrics,
            append=True,
        )

    pprint.pprint(metrics)

    logger.info(f"Results saved in: {os.path.join(RES_DIR, out_dir_path)}")


if __name__ == "__main__":
    main()
