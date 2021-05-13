import random
from typing import List
from nltk import word_tokenize
import torch
from transformers import PreTrainedTokenizerFast as PTTF
from transformers import AutoModelForMaskedLM


class Text:
    def __init__(self, tokenizer: PTTF, text: str):
        self.tokenizer = tokenizer
        self.text = text
        self.tokens = word_tokenize(text)

    def mask(self, p):
        self.masked_tokens = [
            self.tokenizer.mask_token if random.uniform(0, 1) < p else tkn
            for tkn in self.tokens]

        self.token_ids = self.tokenizer(self.masked_tokens, is_split_into_words=True)['input_ids']

    def augment(self, token_ids: List[int]):
        augmented = [
            old if old != self.tokenizer.mask_token_id else new
            for old, new in zip(self.token_ids, token_ids)]

        return self.tokenizer.decode(augmented[1:-1])


class BERTAugment:
    def __init__(
            self,
            pretrained='bert-base-cased',
            mask_token='[MASK]'):

        self.tokenizer = PTTF.from_pretrained(
            pretrained,
            mask_token=mask_token)

        self.model = AutoModelForMaskedLM.from_pretrained(pretrained)
        self.model.eval()

    def predict(self, token_ids: List[int]):
        logits = self.model(torch.tensor(token_ids).view(1, -1)).logits.squeeze()
        w, token_ids = torch.topk(logits, k=5)
        return [
            random.choices(token_ids[i], k=1, weights=w[i])[0].item()
            for i in range(len(logits))]

    def augment(self, text: str, n=10, p=0.3):
        text = Text(self.tokenizer, text)
        res = []
        for _ in range(n):
            text.mask(p)
            res.append(text.augment(self.predict(text.token_ids)))

        return res
