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
import torch
import wandb
import warnings

warnings.filterwarnings("ignore")

from collections import defaultdict
from datetime import datetime
from time import time
from tqdm import tqdm

from accelerate import infer_auto_device_map, init_empty_weights, Accelerator
from sklearn.metrics import accuracy_score, f1_score
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from config import *
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
from src.utils.utils import (
    load_model_and_tokenizer,
    save_results,
    set_env_vars,
    load_sweep_config,
    save_best_config,
)


SAVE_INTERVAL = DEFAULT_SAVE_INTERVAL
RES_DIR = DEFAULT_RES_DIR
LOG_DIR = DEFAULT_LOG_DIR


@click.command()
@click.option(
    "--kind",
    default=DEFAULT_KIND,
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
@click.option(
    "--sweep",
    is_flag=True,
    help="Run sweeps",
)
@click.option(
    "--sweep_config",
    default="sweep_config.json",
    help="Sweep configuration file",
)
@click.option(
    "--load_best_config",
    default=None,
    help="Load the best configuration from a file",
)
def main(kind, runtype, data, sweep, sweep_config, load_best_config):
    # set up wandb
    run = wandb.init(project="kg-runs")
    config = wandb.config

    run_flag = "run"
    if sweep:
        if runtype != "eval":
            raise ValueError("Sweeps can only be run in eval mode")
        run_flag = "sweep"
        kind = config.kind
        temperature = config.temperature
        top_p = config.top_p
        few_shot_num = config.few_shot_num
        few_shot_selection = config.few_shot_selection
    # few_shot_type = config.few_shot_type
    elif load_best_config:
        with open(load_best_config, "r") as f:
            best_config = json.load(f)
        kind = best_config["kind"]
        temperature = best_config["temperature"]
        top_p = best_config["top_p"]
        few_shot_num = best_config["few_shot_num"]
        few_shot_selection = best_config["few_shot_selection"]
    else:
        temperature = DEFAULT_TEMPERATURE
        top_p = DEFAULT_TOP_P
        few_shot_num = DEFAULT_FEW_SHOT_NUM
        few_shot_selection = DEFAULT_FEW_SHOT_SELECTION

        config.update(
            {
                "kind": kind,
                "temperature": temperature,
                "top_p": top_p,
                "few_shot_num": few_shot_num,
                "few_shot_selection": few_shot_selection,
            }
        )

        wandb.config.update(config)

    wandb.run.name = f"{run_flag}_{kind}_t{temperature:.2f}_p{top_p:.2f}_fs{few_shot_num}_{few_shot_selection}"

    logger = logging.getLogger(__name__)

    # set up logging and save directories
    uuid = "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(8)
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_path = f"{runtype}_{few_shot_selection}_{kind}_{uuid}_{timestamp}"
    os.makedirs(os.path.join(RES_DIR, out_dir_path), exist_ok=True)
    os.makedirs(os.path.join(RES_DIR, out_dir_path, LOG_DIR), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(RES_DIR, out_dir_path, LOG_DIR, f"log.txt")
            ),
            logging.StreamHandler(),
        ],
    )

    # set random seeds and environment variables
    logging.info("setting random seeds and environment variables...")
    random.seed(0)
    np.random.seed(1)
    torch.manual_seed(2)
    if torch.cuda.is_available():
        logging.info(
            "Using {} {} GPUs".format(
                torch.cuda.device_count(), torch.cuda.get_device_name()
            )
        )
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
                        logging.info(f"Duplicate found in {file}:\n{dict_line}\n\n")
    else:
        valid = examples[3:]

    logging.info(f"Number of training examples: {len(train)}")
    logging.info(f"Number of validation examples: {len(valid)}")

    # load model and tokenizer
    logging.info("loading model and tokenizer...")
    model_id = DEFAULT_MODEL_ID
    model, tokenizer = load_model_and_tokenizer(model_id)

    # generate the prefix
    logging.info("generating base prompt...")
    prefix = generate_prefix(
        instructions=generate_instructions(schema, kind),
        demonstrations=generate_demonstrations(
            train, kind, num_examples=few_shot_num, selection=few_shot_selection
        ),
    )

    # run/evaluate the model
    logging.info("running the model...")
    logging.info(f"Run type: {runtype}")
    logging.info(f"Data: {data}")
    logging.info(f"Model: {model_id}")
    logging.info(
        f"Run parameters: kind={kind}, temperature={temperature}, top_p={top_p}, few_shot_num={few_shot_num}, few_shot_selection={few_shot_selection}"
    )

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
                model,
                tokenizer,
                prefix,
                input,
                kind,
                temperature=temperature,
                top_p=top_p,
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

            all_inputs.append(prefix + input)
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
                wandb.log(metrics)
            else:
                metrics = compute_metrics(running_time, pred_times, runtype)

            save_results(
                out_dir_path,
                all_inputs,
                gold_tags,
                predicted_responses,
                predicted_tags,
                metrics,
                runtype,
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
            runtype,
            append=True,
        )
        all_inputs.clear()
        gold_tags.clear()
        predicted_responses.clear()
        predicted_tags.clear()

    pprint.pprint(metrics)
    if runtype == "eval" and sweep:
        wandb.log(
            {
                "prediction_time": e_time - s_time,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "union_true_positives": utp,
                "union_false_positives": ufp,
                "union_false_negatives": ufn,
            }
        )
        save_best_config(metrics, config)

    logger.info(f"Results saved in: {os.path.join(RES_DIR, out_dir_path)}")


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep_config = load_sweep_config()
        wandb.agent(wandb.sweep(sweep_config, project="kg-runs"), function=main)
    else:
        main()
