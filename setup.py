from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="scope3-gnn-mapper",
    version="0.1.0",
    author="virbahu",
    description=(
        "Graph Neural Network framework for mapping and predicting "
        "Scope 3 supply chain emissions"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/virbahu/scope3-gnn-mapper",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.2.0",
            "pre-commit>=3.2.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.22.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "scope3-build-graph=scripts.build_graph:main",
            "scope3-train=scripts.train:main",
            "scope3-evaluate=scripts.evaluate:main",
            "scope3-scenario=scripts.run_scenario:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
