import torch
from torch.utils.data import Dataset


class SegmentDataset(Dataset):
    def __init__(self, corpus, tokenizer, max_length=512):
        self.corpus = corpus
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, i):
        # reserve two tokens for start- and end-of-sequence tokens
        remaining_length = self.max_length - 2
        tokens = []

        # add new sentences until maximum length has been reached
        while remaining_length > 0:
            new_tokens = self.tokenizer.tokenize(self.corpus[i]["text"])

            # if new tokens within budget
            if len(new_tokens) <= remaining_length:

                # add tokens
                tokens += new_tokens

                # remove tokens from budget
                remaining_length -= len(new_tokens)

                # move to next sentence (if at end, start at beginning)
                i = (i + 1) % len(self.corpus)
            else:
                break

        return self.tokenizer.prepare_for_model(
            ids=self.tokenizer.convert_tokens_to_ids(tokens),
            max_length=512,
            padding="max_length",
            return_tensors="pt",
            return_special_tokens_mask=True,
        )


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
