from setuptools import setup, find_packages

setup(
    name="sparseopt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "networkx>=3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "jupyter>=1.0.0",
            "numpy>=1.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sparseopt=sparseopt.cli:app",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for optimizing sparse and irregular PyTorch models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sparseopt",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 