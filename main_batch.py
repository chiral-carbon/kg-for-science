import click
import gc
import json
import logging
import numpy as np
import os
import pprint
import random
import re
import torch
import torch.distributed as dist
import string
import sys
import wandb
import warnings

warnings.filterwarnings("ignore")

from collections import defaultdict
from datetime import datetime
from time import time
from tqdm import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from config import *
from src.processing.generate import (
    format_instance,
    get_sentences,
    generate_prefix,
    generate_instructions,
    generate_demonstrations,
    batch_generate_prediction,
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


class LazyArxivDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        abstract = example["title"] + ". " + example["abstract"]
        sentences = get_sentences(abstract)
        inputs = [format_instance(sentence, extraction=None) for sentence in sentences]
        return inputs, example


def collate_fn(batch):
    inputs, examples = zip(*batch)
    flat_inputs = [item for sublist in inputs for item in sublist]
    encodings = tokenizer(
        flat_inputs,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    return encodings, examples


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
@click.option("--batch_size", default=1, help="Batch size for processing")
def main(kind, runtype, data, sweep, sweep_config, load_best_config, batch_size):
    # Initialize the distributed environment
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Set up the device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
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
    random.seed(0)
    np.random.seed(1)
    torch.manual_seed(2)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(3)
        torch.backends.cudnn.deterministic = True

    set_env_vars()

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
        seen = set()
        for file in os.listdir(data):
            with open(os.path.join(data, file), "r") as f:
                for line in f:
                    dict_line = json.loads(line)
                    if dict_line["title"] not in seen:
                        seen.add(dict_line["title"])
                        valid.append(dict_line)
    else:
        valid = examples[3:]

    if global_rank == 0:
        logging.info(f"Number of training examples: {len(train)}")
        logging.info(f"Number of validation examples: {len(valid)}")

    # load model and tokenizer
    model_id = DEFAULT_MODEL_ID
    model, tokenizer = load_model_and_tokenizer(model_id)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    # generate the prefix
    prefix = generate_prefix(
        instructions=generate_instructions(schema, kind),
        demonstrations=generate_demonstrations(
            train, kind, num_examples=few_shot_num, selection=few_shot_selection
        ),
    )

    # Set up dataset and dataloader
    dataset = LazyArxivDataset(valid, tokenizer, max_length=512)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=global_rank, shuffle=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Inference loop
    all_results = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, disable=global_rank != 0):
            inputs, examples = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = batch_generate_prediction(
                model,
                tokenizer,
                prefix,
                inputs["input_ids"],
                kind,
                temperature=temperature,
                top_p=top_p,
                batch_size=len(examples),
                device=device,
            )

            for example, output in zip(examples, outputs):
                pred = extract_prediction(schema, output, kind=kind)
                result = {
                    "title": example["title"],
                    "abstract": example["abstract"],
                    "predicted_tags": pred,
                }
                all_results.append(result)

    # Gather results from all processes
    all_gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(all_gathered_results, all_results)

    # Combine and save results (only on rank 0)
    if global_rank == 0:
        combined_results = [
            item for sublist in all_gathered_results for item in sublist
        ]
        save_results(out_dir_path, combined_results, None, None, None, None, runtype)

        logging.info(f"Results saved in: {os.path.join(RES_DIR, out_dir_path)}")

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
