# BERTAugment
BERTAugment is a Python 3 library for augmenting text for NLP applications. 
It utilizes pre-trained masked language model to mask a given text and augment it by predicting the masked words.

## Installation
```shell
$ pip install bert_augment
```

## Usage
BERTAugment can be used from the command line or as a package.

### Command Line
```shell
bertaugment [-n number] [-p mask_probability] [--pretrained_model pretrained_model] [--seed seed] text
```

**Basic Usage**
```shell
>>> bertaugment 'The picture quality is great and the sound is amazing.'
The picture quality is great and the sound is amazing. 

The picture quality is high and the sound is satisfactory.
The picture here is great and the it is amazing ;
The picture looks is up and the sound is amazing.
The picture quality is good and the quality not good.
The picture quality is good and the sound is amazing.
The ; - is.. | which is not?
The picture quality is great and the sound system amazing.
The picture quality is great and the sound sounds amazing.
The picture quality is great and the sound quality amazing.
The picture quality is poor and the sound is beautiful.
```

### Python
```python
from bert_augment import BERTAugment

ba = BERTAugment(pretrained='bert-base-cased')
samples = ba.augment('The picture quality is great and the sound is amazing.')
```