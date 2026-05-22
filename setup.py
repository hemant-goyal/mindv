from setuptools import setup, find_packages

setup(
    name="mindeepvariant",
    version="1.1.0",
    author="hemant-goyal",
    description="Minimal deep learning AMR variant caller for haploid organisms",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/hemant-goyal/mindv",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.2",
        "pysam>=0.21",
        "biopython>=1.83",
        "numpy>=1.26",
    ],
    entry_points={
        "console_scripts": [
            "mindv=mindeepvariant.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
