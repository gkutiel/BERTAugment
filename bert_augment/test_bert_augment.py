import random
from bert_augment import BERTAugment, Text


ba = BERTAugment()
text = Text(ba.tokenizer, 'The dog was the first species to be domesticated.')
random.seed(0)
text.mask(0.4)

tokens = ['The', 'dog', 'was', 'the', 'first', 'species', 'to', 'be', 'domesticated', '.']
masked_tokens = ['The', 'dog', 'was', '[MASK]', 'first', 'species', 'to', '[MASK]', 'domesticated', '.']
token_ids = [101, 1109, 3676, 1108, 103, 1148, 1530, 1106, 103, 4500, 2913, 119, 102]


def test_tokens():
    assert text.tokens == tokens


def test_masked_tokens():
    assert text.masked_tokens == masked_tokens


def test_token_ids():
    assert text.token_ids == token_ids


def test_predict():
    prediction = ba.predict(token_ids)
    print(prediction)
    assert ba.tokenizer.convert_ids_to_tokens(prediction[4]) == 'this'
    assert ba.tokenizer.convert_ids_to_tokens(prediction[8]) == 'get'


def test_augment():
    s = 'The picture quality is great and the sound is amazing.'
    augmented = ba.augment(s, n=5)
    print(s)
    print()
    print()
    print(*augmented, sep='\n\n')
    assert len(augmented) == 5
    assert False
