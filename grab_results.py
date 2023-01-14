import json
import numpy as np
import pandas as pd

# hans is weird with acc=1. multirc is in superglue? use eraser_multirc not multirc whatever that is.
# do a single trial because the issues with small dataset sizes won't come up here. (small datasets are their own issue, generally we try to collect more data if this is the problem in real life)

datasets = [
    "sst2",
    "qqp",
    "mnli",
    "qnli",
    "boolq",
    "ag_news",
    "imdb",
    "snli",
    "rotten_tomatoes",
    "yelp_polarity",
    "eraser_multi_rc",
    "wiki_qa",
    "scitail",
    "emotion",
    "tweet_eval_hate",
]

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

results = pd.DataFrame()
for i, dataset in enumerate(datasets):
    for model in models:
        try:
            with open(f"./output/{model}/{dataset}/eval_results.json", "r") as f:
                eval_results = json.load(f)
            results.loc[model, dataset] = eval_results["eval_accuracy"]
        except FileNotFoundError:
            results.loc[model, dataset] = np.NaN

results["avg"] = results.mean(axis=1)
results = results.sort_values("avg", ascending=False)
results.to_csv("results/accuracies.csv")
