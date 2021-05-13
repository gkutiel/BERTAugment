from pathlib import Path
from bert_augment import version

pys = list(Path('.').rglob('*.py'))
mds = list(Path('.').rglob('*.md'))
sdist = Path('dist') / f'plmbr-{version}.tar.gz'


def task_test():
    return {
        'actions': ['pytest -v'],
        'file_dep': pys,
    }


def task_build():
    return {
        'actions': ['python setup.py sdist'],
        'targets': [sdist],
        'file_dep': pys,
        'task_dep': ['test']
    }


def task_upload():
    return {
        'actions': [f'twine upload {sdist}'],
        'file_dep': [sdist],
        'verbosity': 2,
    }
