import setuptools
from bert_augment import version

with open("README.md") as f:
    long_description = f.read()

setuptools.setup(
    name="bert_augment",
    version=version,
    author="Gilad Kutiel",
    author_email="gilad.kutiel@gmail.com",
    description="BERT based text augmentation",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="TODO",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'torch',
        'transformers',
        'nltk']
)
