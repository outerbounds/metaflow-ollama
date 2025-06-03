from setuptools import setup, find_namespace_packages

version = "0.2.0"

setup(
    name="metaflow_ollama",
    version=version,
    description="An ollama decorator for Metaflow",
    author="Outerbounds",
    author_email="hello@outerbounds.co",
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=["metaflow_extensions.*"]),
    install_requires=[],
    python_requires=">=3.6",
    classifiers=[
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
