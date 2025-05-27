from setuptools import setup, find_namespace_packages

version = "0.1.1"

setup(
    name="metaflow_ollama",
    version=version,
    description="An EXPERIMENTAL Ollama decorator for Metaflow",
    author="Outerbounds",
    author_email="hello@outerbounds.co",
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=["metaflow_extensions.*"]),
    # Remove py_modules - it conflicts with packages
    install_requires=[],
    python_requires=">=3.6",  # Add this for clarity
    classifiers=[  # Add these for better PyPI presentation
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
