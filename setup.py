from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="amr-pattern-recognition",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine Learning-based Antimicrobial Resistance Pattern Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Reyn4ldo/final-thesis-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.11.0",
            "mypy>=0.991",
        ],
    },
    entry_points={
        "console_scripts": [
            "amr-app=app.streamlit_app:main",
        ],
    },
)
