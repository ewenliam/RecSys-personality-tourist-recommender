"""Setup script for the Tourist Recommendation System."""
from setuptools import setup, find_packages

setup(
    name="tourist-recommender",
    version="0.1.0",
    description="Personalized Tourist Recommendation System with MBTI, BERTopic, and GNN",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "bertopic>=0.15.0",
        "umap-learn>=0.5.0",
        "hdbscan>=0.8.0",
        "torch-geometric>=2.3.0",
        "xgboost>=1.7.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.0.270",
            "ipykernel>=6.0.0",
        ],
    },
)
