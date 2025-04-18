from setuptools import setup, find_packages

setup(
    name="sparseopt",
    version="0.1.0",
    description="AI model graph optimizer built on PyTorch FX",
    author="Mohamed Farouk",
    author_email="mohamed.farouk@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "transformers>=4.20.0",
        "rich>=10.0.0",
        "numpy>=1.20.0",
        "tqdm>=4.60.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 