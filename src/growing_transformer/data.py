import random
from itertools import chain
from typing import Union

from functools import partial

import torch

from datasets import Dataset, DatasetDict, load_dataset, load_metric
from torch.utils.data import Subset


def tokenize_dataset(
    tokenizer,
    dataset: Union[Dataset, DatasetDict],
    num_workers: int = None,
    ignore_cache: bool = False,
    add_special_tokens: bool = False,
):
    """Tokenize the complete dataset."""
    if isinstance(dataset, DatasetDict):
        column_names = dataset["train"].column_names
    else:
        column_names = dataset.column_names

    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(batch):
        return tokenizer(
            batch[text_column_name],
            return_special_tokens_mask=add_special_tokens,
            add_special_tokens=add_special_tokens,
        )

    return dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=column_names,
        load_from_cache_file=(not ignore_cache),
        desc="Running tokenizer on every text in dataset",
    )


def group_dataset(
    dataset: Union[Dataset, DatasetDict], num_workers: int = None,
    ignore_cache: bool = False, max_seq_length: int = 512
):
    """
    Group sentences according to FULL-SENTENCES scheme introduced by RoBERTa.
    """

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    return dataset.map(
        group_texts,
        batched=True,
        num_proc=num_workers,
        load_from_cache_file=(not ignore_cache),
        desc=f"Grouping texts in chunks of {max_seq_length}",
    )


def prepare_mlm_dataset(
    tokenizer, dataset, num_workers: int = None, ignore_cache: bool = False,
    max_seq_length: int = 512
):
    dataset = tokenize_dataset(
        tokenizer=tokenizer,
        dataset=dataset,
        num_workers=num_workers,
        ignore_cache=ignore_cache,
        add_special_tokens=False,
    )

    dataset = group_dataset(
        dataset=dataset, num_workers=num_workers, ignore_cache=ignore_cache,
        max_seq_length=max_seq_length
    )

    return dataset


def downsample_dataset(data, proportion):
    """Subsample dataset."""

    return split_dataset(data, proportion)[0]


def split_dataset(data, proportion):
    """Split datset into two parts."""

    n_samples = len(data)
    indices = list(range(n_samples))
    random.shuffle(indices)

    # calculate new number of samples
    n_samples = round(proportion * n_samples)
    return Subset(data, indices[:n_samples]), Subset(data, indices[n_samples:])


glue_task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def load_glue_task(task, tokenizer, ignore_cache=False, max_seq_length=512):
    dataset = load_dataset("glue", task)

    sentence1_key, sentence2_key = glue_task_to_keys[task]

    padding = "max_length"

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None else
            (examples[sentence1_key], examples[sentence2_key])
        )
        return {
            "labels": examples["label"],
            **tokenizer(
                *args, padding=padding, max_length=max_seq_length,
                truncation=True),
        }

    if not task == "stsb":
        label_list = dataset["train"].features["label"].names
        num_labels = len(label_list)
    else:
        label_list = None
        num_labels = 1

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=(not ignore_cache),
        desc="Running tokenizer on dataset",
    )

    metric = load_metric("glue", task)

    return dataset, metric, num_labels, label_list


def mask_tokens(batch, tokenizer, mlm_prob):
    """
    Prepare masked tokens inputs/labels for masked language modeling.

    This function closely follows the functionality of the
    DataCollatorForLanguageModeling (used in the dynamic case).
    """

    # If special token mask has been preprocessed, pop it from the dict
    special_tokens_mask = batch.pop("special_tokens_mask", None)

    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(
                val, already_has_special_tokens=True)
            for val in batch["input_ids"]
        ]

    special_tokens_mask = torch.tensor(
        special_tokens_mask, dtype=torch.bool)

    inputs = torch.tensor(batch["input_ids"], dtype=torch.long)
    labels = inputs.clone()

    # We sample a few tokens in each sequence for MLM training
    probability_matrix = torch.full(labels.shape, mlm_prob)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    # We only compute loss on masked tokens
    labels[~masked_indices] = -100

    # 80% of the time, we replace masked input tokens with [MASK]
    indices_replaced = torch.bernoulli(
        torch.full(labels.shape, 0.8)).bool() & masked_indices

    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices & ~indices_replaced
    )

    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long)

    inputs[indices_random] = random_words[indices_random]

    # The rest of the time we keep the masked input tokens unchanged

    batch["labels"] = labels.tolist()
    batch["input_ids"] = inputs.tolist()

    return batch


def apply_static_masking(dataset, tokenizer, mlm_prob, num_workers: int = None,
                         ignore_cache=False):
    return dataset.map(
        partial(mask_tokens, tokenizer=tokenizer, mlm_prob=mlm_prob),
        batched=True,
        num_proc=num_workers,
        load_from_cache_file=ignore_cache,
        desc="Apply language masking",
    )
