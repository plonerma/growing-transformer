import random
from itertools import chain
from typing import Union

from datasets import Dataset, DatasetDict, load_dataset, load_metric
from torch.utils.data import Subset


def tokenize_dataset(
    tokenizer, dataset: Union[Dataset, DatasetDict], num_workers: int = None, ignore_cache: bool = False
):
    """Tokenize the complete dataset."""
    if isinstance(dataset, DatasetDict):
        column_names = dataset["train"].column_names
    else:
        column_names = dataset.column_names

    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(batch):
        return tokenizer(batch[text_column_name], return_special_tokens_mask=True)

    return dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=column_names,
        load_from_cache_file=(not ignore_cache),
        desc="Running tokenizer on every text in dataset",
    )


def group_dataset(
    dataset: Union[Dataset, DatasetDict], num_workers: int = None, ignore_cache: bool = False, max_seq_length: int = 512
):
    """Group sentences according to FULL-SENTENCES scheme introduced by RoBERTa."""

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


def load_glue_task(task, tokenizer, ignore_cache=False):
    dataset = load_dataset("glue", task)

    sentence1_key, sentence2_key = glue_task_to_keys[task]

    padding = "max_length"
    max_seq_length = 512

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        return {
            "labels": examples["label"],
            **tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True),
        }

    if not task == "stsb":
        label_list = dataset["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=(not ignore_cache),
        desc="Running tokenizer on dataset",
    )

    metric = load_metric("glue", task)

    return dataset, metric, num_labels
