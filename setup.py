from setuptools import setup, find_packages

setup(
    name="f1_race_predictor",
    version="1.0.0",
    author="Preetham V R",
    description="Predict Formula 1 podium finishes, race winners, and exact finishing positions using Gradient Boosting models.",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "streamlit>=1.28.0",
        "requests>=2.31.0",
    ],
)
