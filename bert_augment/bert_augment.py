import random
from typing import List
from nltk import word_tokenize
import torch
from transformers import PreTrainedTokenizerFast as PTTF
from transformers import AutoModelForMaskedLM
from fire import Fire


class Text:
    '''
    A class for handling text.
    '''

    def __init__(self, tokenizer: PTTF, text: str):
        self.tokenizer = tokenizer
        self.text = text
        self.tokens = word_tokenize(text)

    def mask(self, p: float):
        '''
        Mask the text.

        Parameters
        ----------
        p : float
            The probability to mask a token.

        '''

        self.masked_tokens = self.tokens[:1] + [
            self.tokenizer.mask_token if random.uniform(0, 1) < p else tkn
            for tkn in self.tokens[1:]]

        self.token_ids = self.tokenizer(self.masked_tokens, is_split_into_words=True)['input_ids']

    def augment(self, token_ids: List[int]) -> str:
        '''
        Replaces original tokens with new ones and decode the result to string.

        Parameters
        ----------
        token_ids : List[int]
            New tokens to be use, the tokens are used only where the original token was masked.
        '''
        augmented = [
            old if old != self.tokenizer.mask_token_id else new
            for old, new in zip(self.token_ids, token_ids)]

        return self.tokenizer.decode(augmented[1:-1])


class BERTAugment:
    '''
    A class for text augmentation.

    The augmented data is generated using masked language model.
    '''

    def __init__(self, pretrained):
        self.tokenizer = PTTF.from_pretrained(
            pretrained,
            mask_token='[MASK]')

        self.model = AutoModelForMaskedLM.from_pretrained(pretrained)
        self.model.eval()

    def predict(self, token_ids: List[int]) -> List[int]:
        '''
        Predicting masked words.

        Parameters
        ----------
        token_ids : List[int]
            List of token ids containing one or more mask_token_id.

        Returns
        -------
        list
            a list of predicted token ids.
        '''
        logits = self.model(torch.tensor(token_ids).view(1, -1)).logits.squeeze()
        w, token_ids = torch.topk(logits, k=5)
        return [
            random.choices(token_ids[i], k=1, weights=w[i])[0].item()
            for i in range(len(logits))]

    def augment(self, text: str, n=10, p=0.3) -> List[str]:
        '''
        Augment the given text.

        Parameters
        ----------
        text : str
            The text to be augmented.
        n : int
            Number of samples to generate.
        p : float
            Masking probability.

        Returns
        -------
        list
            a list of generated samples.
        '''
        text = Text(self.tokenizer, text)
        res = []
        for _ in range(n):
            text.mask(p)
            res.append(text.augment(self.predict(text.token_ids)))

        return res


def main():
    from transformers import logging
    logging.set_verbosity_error()

    def augment(
            text,
            n=10,
            p=0.3,
            pretrained_model='bert-base-cased',
            seed=2021):

        random.seed(seed)
        ba = BERTAugment(pretrained=pretrained_model)

        print(text, '\n')
        for a in ba.augment(text, n=n, p=p):
            print(a)

    Fire(augment)
