## Ollama with Metaflow

<video width="100%" controls>
    <source src="./static/metaflow-ollama.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>

[Ollama](https://ollama.com/) is an open-source framework for running a wide variety of generative AI models. 
It includes an inference server, and tools for running models locally and on production servers.
This repository demonstrates how to run any Ollama model for the lifecycle of a Metaflow task, so it can be queried from the task. 

### Benefits
The package handles:
- Ensuring Ollama server is installed and running across Mac and Linux.
- Pulling models and running them as background processes accessible from workstations/tasks.
- Cleaning up all processes created by the manager, e.g., at the end of task_decorate.

### Use cases
It can be used in two modes.

#### Mode 1: annotate a Metaflow `@step` with `@ollama` 

To simply pull and run models so they are available for the lifecycle of a Metaflow task, use the `@ollama` step decorator.

```python
    @pypi(packages={'ollama': ''})
    @ollama(models=['qwen:0.5b'])
    @step
    def start(self):
        from ollama import chat
        from ollama import ChatResponse

        response_qwen: ChatResponse = chat(
            model='qwen:0.5b', 
            messages=[
                {
                    'role': 'user',
                    'content': 'What are the leading Chinese tech companies?',
                },
            ]
        )
        self.next(self.end)
```

#### Mode 2: `from metaflow.plugins.ollama import OllamaManager`

Sometimes, you may prefer managing ollama processes more directly. 
For this, use the `OllamaManager` from within or outside a Metaflow task.
```python
    @pypi(packages={'ollama': ''})
    @step
    def prompt(self):
        from ollama import chat 
        from ollama import ChatResponse
        from metaflow.plugins.ollama import OllamaManager 

        ollama_manager = OllamaManager(models=[self.input], debug=True)
        self.response: ChatResponse = chat(model=self.input, messages=[self.config.message])
        ollama_manager.terminate_models()
        self.next(self.end)
```

## Setup

### Setup on Outerbounds

If you are an Outerbounds customer, the ollama tools are bundled in the `outerbounds` package for versions ≥ 0.3.143. 

After running
```bash
pip install outerbounds
```
you will be able to use the abstractions
```python
from metaflow import ollama
from metaflow.plugins.ollama import OllamaManager 
```

### Setup for open-source Metaflow

If you are an open-source Metaflow user, you can install Metaflow normally, and then install the `metaflow-ollama` extension
```bash
pip install metaflow-ollama
```
then, you will be able to use the abstractions
```python
from metaflow import ollama
from metaflow.plugins.ollama import OllamaManager 
```

## Limitations and improvements

- For remote processes (e.g., `@kubernetes` tasks), the extension installs ollama binary dynamically. This could be cached.
    - On Outerbounds, this is handled automatically via `fast-bakery`. 
- Models are pulled via `ollama pull <>`. Similar to previous line, these models could be cached in S3 to control/optimize network throughput.
    - [Where are (ollama) models stored?](https://github.com/ollama/ollama/blob/main/docs/faq.md#where-are-models-stored)
- Support backends beyond the "local" runtime. For example, so a `@kubernetes` task could submit requests to other pods, previously running servers, etc.