from setuptools import setup, find_packages

setup(
    name="mindeepvariant",
    version="0.1.0",
    description=(
        "A minimal, from-scratch deep learning variant caller. "
        "Treats variant calling as image classification, inspired by "
        "Google's DeepVariant and Karpathy's minGPT."
    ),
    author="Hemant Goyal",
    license="MIT",
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=1.10",
        "pysam>=0.19",
        "biopython>=1.79",
        "matplotlib>=3.5",
    ],
    entry_points={
        "console_scripts": [
            "mindv=mindeepvariant.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
