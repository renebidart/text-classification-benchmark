#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Based on https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py """

import os
import random
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import logging

import numpy as np
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    data_info: Optional[str] = field(default="all_classification_datasets.json",)
    default_classification_hyperparams: bool = field(default=True,)
    task_name: Optional[str] = field(
        default=None, metadata={"help": "Name of the task to train on"},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    dataset_cache_dir: Optional[str] = field(
        default="hf_datasets",
        metadata={
            "help": "where to save the huggingface datasets, !!! maybe useful for batch jobs"
        },
    )
    keep_in_memory: Optional[bool] = field(
        default=True, metadata={"help": "Must be false for datasets that are too big for RAM"},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8, metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test data."},
    )


@dataclass
class ModelArguments:
    keep_first: Optional[int] = field(
        default=0, metadata={"help": "How many tokens to keep from the start of the sequence."},
    )
    classformer_downsample: Optional[str] = field(
        default="38", metadata={"help": "How to downsample the objective for MLM"},
    )
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script it's the path to a json file
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    # if training_args.should_log:
    #     transformers.utils.logging.set_verbosity_info()
    #     transformers.utils.logging.enable_default_handler()
    #     transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set random seed each time so can run multiple trials
    random.seed(datetime.now())
    training_args.seed = random.randint(0, 100)
    set_seed(training_args.seed)
    print(f"random_seed: {training_args.seed}")

    # Get info on the dataset we're using:
    with open(data_args.data_info, encoding="UTF-8") as json_data:
        all_info = json.load(json_data)
    info = all_info[data_args.task_name]

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=info["n_labels"],
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # If model_name_or_path is not specified then training from scratch
    if model_args.model_name_or_path is None:
        logger.info("Training new model from scratch")
        model = AutoModelForSequenceClassification.from_config(config)
    # elif 'neo' in model_args.model_name_or_path:
    #     from transformers import GPTNeoForSequenceClassification
    #     model = GPTNeoForSequenceClassification(config)
    else:
        # hacky way to load weights of pre-trained classifier without error from
        # different number of classes so differently sized output layer
        model = AutoModelForSequenceClassification.from_config(config)
        pre_model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # select only the parameters with matching size
        params = {
            pre_k: pre_v if v.size() == pre_v.size() else v
            for ((k, v), (pre_k, pre_v)) in zip(
                model.state_dict().items(), pre_model.state_dict().items()
            )
        }
        logger.info(f"Drop {len(pre_model.state_dict())-len(params)} weights from pretrained")
        model.load_state_dict(params, strict=False)

    # Randomize linear layer parameters for multiple trials:
    if hasattr(model, "classifier"):
        for layer in model.classifier.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    # Preprocessing Set Up
    if not hasattr(tokenizer, "pad_token"):
        tokenizer.pad_token = tokenizer.eos_token
    if "gpt" in model_args.model_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    padding = "max_length" if data_args.pad_to_max_length else False
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"max_seq_length passed ({data_args.max_seq_length}) larger than max length"
            f"for model ({tokenizer.model_max_length})."
            f"Using max_seq_length={tokenizer.model_max_length}."
        )
    max_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        """Tokenize texts and change labels to int

        Uses data labels supplied in config (info) read from outside function
        We ordered sentences (for len>2) so last is least useful and truncated first
        """
        if info["type"] in ["classification", "nli", "qa"]:  # classification objective
            if isinstance(info["x_name"], str):
                texts = (examples[info["x_name"]],)
            elif len(info["x_name"]) == 2:
                texts = (examples[info["x_name"][0]], examples[info["x_name"][1]])
            elif len(info["x_name"]) == 3:
                examples1 = examples[info["x_name"][1]]
                examples2 = examples[info["x_name"][2]]
                combined = [e1 + tokenizer.sep_token + e2 for e1, e2 in zip(examples1, examples2)]
                texts = (examples[info["x_name"][0]], combined)
            elif len(info["x_name"]) == 4:
                examples1 = examples[info["x_name"][1]]
                examples2 = examples[info["x_name"][2]]
                examples3 = examples[info["x_name"][3]]
                combined = [
                    e1 + tokenizer.sep_token + e2 + tokenizer.sep_token + e3
                    for e1, e2, e3 in zip(examples1, examples2, examples3)
                ]
                texts = (examples[info["x_name"][0]], combined)
            result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)
        else:
            raise ValueError(f"Unknown dataset type {data_args.task_name}")

        # Convert labels from strings to integers
        if label_to_int is not None:
            result["label"] = [
                (label_to_int[x] if x != -1 else -1) for x in examples[info["y_name"]]
            ]
        else:
            result["label"] = examples[info["y_name"]]
        return result

    # Do Preprocessing
    with training_args.main_process_first(desc="dataset map pre-processing"):
        print(f"Loading dataset: {data_args.task_name} from {data_args.dataset_cache_dir}")
        raw_dataset = load_dataset(
            info["dataset_name"],
            info["config"],
            cache_dir=data_args.dataset_cache_dir,
            keep_in_memory=data_args.keep_in_memory,
            ignore_verifications=True,
        )
        # drop all missing observations (label is -1)
        raw_dataset = raw_dataset.filter(lambda example: example[info["y_name"]] != -1)
        # get label to id (int) mapping from train dataset and apply to all
        if isinstance(raw_dataset["train"][info["y_name"]][0], str):
            label_list = sorted(list(set(raw_dataset["train"][info["y_name"]])))
            label_to_int = {label: i for i, label in enumerate(label_list)}
        else:
            label_to_int = None

        datasets = raw_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=raw_dataset["train"].column_names,
            num_proc=data_args.preprocessing_num_workers,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if info["val_name"] is None:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets[info["val_name"]]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function. Add acc if using cola
    if data_args.task_name in ["cola"]:
        acc_metric = load_metric("accuracy", keep_in_memory=True)
    try:
        metric = load_metric(info["dataset_name"], info["config"], keep_in_memory=True)
    except FileNotFoundError:
        metric = load_metric("accuracy", keep_in_memory=True)  # we have lots of RAMs

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            if data_args.task_name in ["cola"]:  # Add an extra for cola
                result["eval_accuracy"] = acc_metric.compute(
                    predictions=preds, references=p.label_ids
                )["accuracy"]
            return result
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator defaults to DataCollatorWithPadding so change if already did padding
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Set of hyperparams that tend to work well accross datasets and sensible settings:
    if data_args.default_classification_hyperparams:
        # Following roberta do ~10 epochs (less with big dataset), 10% warmup, bs=32
        total_max_steps = int(10 * min(info["n_train"], 300000) / 32)
        training_args.save_total_limit = 1  # more is almost always unnecessary
        training_args.save_strategy = "epoch"
        training_args.metric_for_best_model = "eval_loss"
        training_args.load_best_model_at_end = True
        training_args.per_device_train_batch_size = 32
        training_args.per_device_eval_batch_size = 32
        training_args.learning_rate = 1e-05
        training_args.adam_beta1 = 0.9
        training_args.adam_beta2 = 0.98
        training_args.adam_epsilon = 1e-06
        training_args.lr_scheduler_type = "polynomial"
        training_args.warmup_steps = total_max_steps // 10
        training_args.max_steps = total_max_steps

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "tags": "text-classification",
        }
        if data_args.task_name is not None:
            kwargs["language"] = "en"
            kwargs["dataset_tags"] = "glue"
            kwargs["dataset_args"] = data_args.task_name
            kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"
        trainer.push_to_hub(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
