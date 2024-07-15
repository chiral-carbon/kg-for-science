import json
import logging
import os
import torch

from accelerate import infer_auto_device_map, init_empty_weights, Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

RES_DIR = "results"


def save_results(
    out_dir_path,
    gold_tags,
    all_inputs,
    predicted_responses,
    predicted_tags,
    metrics,
    runtype,
    append=False,
):
    mode = "a" if append else "w"

    with open(os.path.join(RES_DIR, out_dir_path, "prompts.txt"), mode) as f:
        for input, pred_response, pred_tag, gold_tag in zip(
            all_inputs, predicted_responses, predicted_tags, gold_tags
        ):
            f.write(f"{input}\n")
            f.write(f"True Tag: {gold_tag}\n")
            f.write(f"Predicted Response: {pred_response}\n")
            f.write(f"Predicted Tag: {pred_tag}\n")
            f.write("#" * 50 + "\n")

    with open(
        os.path.join(RES_DIR, out_dir_path, "predicted_responses.txt"),
        mode,
        encoding="utf-8",
    ) as f:
        for response in predicted_responses:
            f.write(f"{response}\n")
            f.write("#" * 50 + "\n")

    if append:
        with open(os.path.join(RES_DIR, out_dir_path, "predictions.json"), "r+") as f:
            data = json.load(f)
            data["predicted_tags"].extend(predicted_tags)
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
    else:
        with open(os.path.join(RES_DIR, out_dir_path, "predictions.json"), "w") as f:
            json.dump({"predicted_tags": predicted_tags}, f, indent=4)

    if runtype == "eval":
        if append:
            with open(
                os.path.join(RES_DIR, out_dir_path, "ground_truth.json"), "r+"
            ) as f:
                data = json.load(f)
                data["gold_tags"].extend(gold_tag)
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
        else:
            with open(
                os.path.join(RES_DIR, out_dir_path, "ground_truth.json"), "w"
            ) as f:
                json.dump({"gold_tags": gold_tags}, f, indent=4)

    with open(os.path.join(RES_DIR, out_dir_path, "metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "prompt_file": "prompts.txt"}, f, indent=4)

    logging.info(f"Results saved in: {os.path.join(RES_DIR, out_dir_path)}")


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
