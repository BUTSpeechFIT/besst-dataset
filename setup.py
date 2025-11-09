#!/usr/bin/env python3
"""
Setup configuration for besst-dataset package.

For modern installations, use pyproject.toml.
This setup.py is provided for compatibility with older pip versions.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="besst-dataset",
    version="0.6.0",
    description="BESST multimodal stress/cognitive load dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jean Pešan",
    author_email="ipesan@fit.vutbr.cz",
    url="https://github.com/yourusername/besst-dataset",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "dataset": ["metadata/**/*", "**/*.py"],
    },
    python_requires=">=3.8",
    install_requires=[
        "datasets>=2.0.0",
        "numpy>=1.20.0",
        "librosa>=0.9.0",
        "soundfile>=0.10.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="dataset multimodal stress cognitive-load affective-computing",
    license="CC BY-NC-SA 4.0",
)
