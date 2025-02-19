from setuptools import setup, find_namespace_packages

version = "0.0.1"

setup(
    name="metaflow-ollama",
    version=version,
    description="An EXPERIMENTAL Ollama decorator for Metaflow",
    author="Outerbounds",
    author_email="hello@outerbounds.co",
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=["metaflow_extensions.*"]),
    py_modules=[
        "metaflow_extensions",
    ],
    install_requires=[],
)