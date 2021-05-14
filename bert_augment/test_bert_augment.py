import random
from bert_augment import BERTAugment, Text


ba = BERTAugment('bert-base-cased')
text = Text(ba.tokenizer, 'The dog was the first species to be domesticated.')
random.seed(0)
text.mask(0.4)

tokens = ['The', 'dog', 'was', 'the', 'first', 'species', 'to', 'be', 'domesticated', '.']
masked_tokens = ['The', 'dog', 'was', 'the', '[MASK]', 'species', 'to', 'be', '[MASK]', '.']
token_ids = [101, 1109, 3676, 1108, 1103, 103, 1530, 1106, 1129, 103, 119, 102]


def test_tokens():
    assert text.tokens == tokens


def test_masked_tokens():
    assert text.masked_tokens == masked_tokens


def test_token_ids():
    assert text.token_ids == token_ids


def test_predict():
    prediction = ba.predict(token_ids)
    print(prediction)
    assert ba.tokenizer.convert_ids_to_tokens(prediction[5]) == 'last'
    assert ba.tokenizer.convert_ids_to_tokens(prediction[9]) == 'killed'


def test_augment():
    s = 'The picture quality is great and the sound is amazing.'
    augmented = ba.augment(s, n=5)
    assert len(augmented) == 5
