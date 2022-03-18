import random

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from tqdm import tqdm

from .. import logger as log


class SegmentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.segments = list(self.create_token_segments(data))

    def __getitem__(self, i):
        return self.segments[i]

    def __len__(self):
        return len(self.segments)

    def downsampled(self, proportion):
        """Subsample dataset."""
        n_samples = len(self)
        indices = list(range(n_samples))
        random.shuffle(indices)

        # calculate new number of samples
        n_samples = round(proportion * n_samples)
        # create new dataset
        return Subset(self, indices[:n_samples])

    def prepare_tokens(self, tokens):
        return self.tokenizer.prepare_for_model(
            ids=self.tokenizer.convert_tokens_to_ids(tokens),
            max_length=512,
            padding="max_length",
            return_tensors="pt",
            return_special_tokens_mask=True,
            truncation=True,
        )

    def create_token_segments(
        self,
        data,
    ):

        log.info("Preparing SegmentDataset")

        # reserve two tokens for start- and end-of-sequence tokens
        max_tokens = self.max_length - 2
        remaining_length = max_tokens
        tokens = []

        for segment in tqdm(data):
            new_tokens = self.tokenizer.tokenize(segment["text"])

            # if not enough remaining budget
            if len(new_tokens) > remaining_length:
                # yield full segment
                yield self.prepare_tokens(tokens)

                # reset new segment
                remaining_length = max_tokens
                tokens = []

            # add tokens
            tokens += new_tokens

            # reduce budget by number of tokens
            remaining_length -= len(new_tokens)

        if len(tokens) > 0:  # a new segment was started
            yield self.prepare_tokens(tokens)


class MLMSegmenetDataset(SegmentDataset):
    def __init__(self, *args, mask_rate=0.15, **kwargs):
        super().__init__(*args, **kwargs)

        self.mask_rate = mask_rate

    def __getitem__(self, i):
        encoded_input = super().__getitem__(i)

        input_masked = encoded_input["input_ids"].clone()

        # randomly mask
        mask = torch.rand(input_masked.shape) < self.mask_rate

        # unmask special tokens (bos, eos, pad)
        mask = mask & (~encoded_input["special_tokens_mask"].bool())

        # With 80% prob, mask token
        mask_tokens = (torch.rand(input_masked.shape) < 0.8) & mask
        input_masked[mask_tokens] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # With 10% (50% of remaining), replace token
        replace_tokens = (torch.rand(input_masked.shape) < 0.5) & mask & ~mask_tokens

        random_tokens = torch.randint(len(self.tokenizer), size=input_masked.shape, dtype=input_masked.dtype)

        input_masked[replace_tokens] = random_tokens[replace_tokens]

        return {"input_masked": input_masked, "mlm_mask": mask, **encoded_input}
