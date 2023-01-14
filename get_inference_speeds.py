import time
import json
from pathlib import Path
import pandas as pd
import numpy as np

import torch
import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification
import warnings

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()  # suppress bigbird warning on seq<256 where it uses standard attention


def get_inference_speed(model, seq_len=256, bs=1, n_trials=1, device="cpu"):
    config = AutoConfig.from_pretrained(model, num_labels=4)
    if "gpt" in model:  # we don't use the tokenizer but this prevents errors:
        config.pad_token_id = config.eos_token_id
    model = AutoModelForSequenceClassification.from_config(config)
    model.eval()
    model.to(device)
    input_ids = torch.randint(0, 10000, (bs, seq_len))
    with torch.no_grad():
        start = time.time()
        for _ in range(n_trials):
            model(input_ids)
        end = time.time()
        return (end - start) / n_trials


def gather_all_speeds(
    models,
    batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
    seq_lens=[32, 64, 128, 256, 512],
    n_trials=1,
    device="cpu",
    threads=24,
):
    torch.set_num_interop_threads(threads)  # Inter-op parallelism
    torch.set_num_threads(threads)
    results = []
    for bs in batch_sizes:
        for seq_len in seq_lens:
            for model in models:
                speed = get_inference_speed(model, seq_len, bs, n_trials, device)
                results.append({"model": model, "bs": bs, "seq_len": seq_len, "speed": speed})
    results = pd.DataFrame(results, columns=["model", "bs", "seq_len", "speed"])
    return results


models = [
    "albert-base-v2",
    "bert-base-uncased",
    "distilgpt2",
    "distilroberta-base",
    "EleutherAI/gpt-neo-125M",
    "facebook/muppet-roberta-base",
    "funnel-transformer/medium-base",
    "funnel-transformer/small-base",
    "google/bigbird-roberta-base",
    "google/electra-base-discriminator",
    "google/mobilebert-uncased",
    "gpt2",
    "microsoft/deberta-v3-base",
    "roberta-base",
    "squeezebert/squeezebert-uncased",
]

device = "cpu"
threads = 24

results = gather_all_speeds(
    models,
    batch_sizes=[1, 2, 4, 8, 16, 32],
    seq_lens=[16, 32, 64, 128, 256, 512],
    n_trials=5,
    device=device,
    threads=threads,
)

results.to_csv(f"results/inference_speed_{device}_{threads}_while_train.csv")
