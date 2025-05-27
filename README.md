## Ollama with Metaflow

[![metaflow-ollama](https://i.ytimg.com/vi/G0z_yNk5AkE/maxresdefault.jpg)](https://www.youtube.com/watch?v=G0z_yNk5AkE)


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

To pull and run models so they are available for the lifecycle of a Metaflow task, use the `@ollama` step decorator.

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

The arguments to the decorator are:
```
models: list[str]
    List of Ollama model names.
backend: str
    Determines where and how to run the Ollama process.
force_pull: bool
    Whether to run `ollama pull` no matter what, or first check the remote cache in Metaflow datastore for this model key.
skip_push_check: bool
    Whether to skip the check that populates/overwrites remote cache on terminating an ollama model.
debug: bool
    Whether to turn on verbose debugging logs.
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

        ollama_manager = OllamaManager(
            models=[self.input], 
            flow_datastore_backend=self._datastore.parent_datastore._storage_impl,
        )
        self.response: ChatResponse = chat(model=self.input, messages=[self.config.message])
        ollama_manager.terminate_models()
        self.next(self.end)
```

The `OllamaManager` constructor takes the same arguments as the decorator:
```
models: list[str]
    List of Ollama model names.
backend: str
    Determines where and how to run the Ollama process.
force_pull: bool
    Whether to run `ollama pull` no matter what, or first check the remote cache in Metaflow datastore for this model key.
skip_push_check: bool
    Whether to skip the check that populates/overwrites remote cache on terminating an ollama model.
debug: bool
    Whether to turn on verbose debugging logs.
```

It also takes two arguments that specify the location of the model cache. 

```
flow_datastore_backend
    When running from inside a flow, set as shown above using self._datastore.parent_datastore._storage_impl.

remote_storage_root
    When running outside a flow, or to hard code the remote data store location, pass an S3 path.
    Giving permission to an S3 bucket from your metaflow-ollama runtime is not in scope of this library.
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
    - The dynamic installation requires that `curl` exist in the Metaflow task image.
- Custom models are not currently supported. It is not impossible they can work, but at the moment this package does not natively integrate with Ollama's custom model registration functions. If desired, for example to connect with upstream @model artifacts produced in flows, please contact Outerbounds.
- Support backends beyond the "local" runtime. For example, so a `@kubernetes` task could submit requests to other pods, previously running servers, etc.