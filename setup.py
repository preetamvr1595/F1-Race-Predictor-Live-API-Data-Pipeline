from setuptools import setup, find_packages

setup(
    name="f1_race_predictor",
    version="1.0.0",
    author="Preetham V R",
    description="Formula 1 podium, winner, and finish position prediction system using Gradient Boosting models.",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "pydantic>=2.0.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.7.0", "flake8>=6.1.0"],
    },
)
