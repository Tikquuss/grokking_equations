from setuptools import find_packages, setup

setup(
    name="grok",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        "pytorch_lightning==1.5",
        #"torch==1.11.0",
        #"torchtext<0.12.0", # to avoid `No module named 'torchtext.legacy'` with `import pytorch_lightning as plt`
        "torchtext==0.11.2",
        #"torch==1.10.2", will be automatically installed by torchtext
        "blobfile",
        "numpy",
        "tqdm",
        "scipy",
        "mod",
        "matplotlib",
        "intrinsics-dimension",
        "wandb",
        "sympy"
    ],
)