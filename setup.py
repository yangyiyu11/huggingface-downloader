from setuptools import setup, find_packages

setup(
    name="hf-downloader",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "huggingface-hub",
        "tqdm"
    ],
    entry_points={
        "console_scripts": [
            "hf-downloader=src.hf_downloader:main"
        ]
    }
)