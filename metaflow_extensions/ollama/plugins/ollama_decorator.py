from metaflow.decorators import StepDecorator
from metaflow import current
import functools
import os

from .ollama import OllamaManager


class OllamaDecorator(StepDecorator):
    """
    This decorator is used to run Ollama APIs as Metaflow task sidecars.

    User code call
    --------------
    @ollama(
        models=[...],
        ...
    )

    Valid backend options
    ---------------------
    - 'local': Run as a separate process on the local task machine.
    - (TODO) 'managed': Outerbounds hosts and selects compute provider.
    - (TODO) 'remote': Spin up separate instance to serve Ollama models.

    Valid model options
    -------------------
    Any model here https://ollama.com/search, e.g. 'llama3.2', 'llama3.3'

    Parameters
    ----------
    models: list[Ollama]
        List of Ollama containers running models in sidecars.
    backend: str
        Determines where and how to run the Ollama process.
    force_pull: bool
        Whether to run `ollama pull` no matter what, or first check the remote cache in Metaflow datastore for this model key.
    skip_push_check: bool
        Whether to skip the check that populates/overwrites remote cache on terminating an ollama model.
    """

    name = "ollama"
    defaults = {
        "models": [],
        "backend": "local",
        "force_pull": False,
        "skip_push_check": False,
        "debug": False,
    }

    def step_init(
        self, flow, graph, step_name, decorators, environment, flow_datastore, logger
    ):
        super().step_init(
            flow, graph, step_name, decorators, environment, flow_datastore, logger
        )
        self.flow_datastore_backend = flow_datastore._storage_impl

    def task_decorate(
        self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
    ):
        @functools.wraps(step_func)
        def ollama_wrapper():
            try:
                self.ollama_manager = OllamaManager(
                    models=self.attributes["models"],
                    backend=self.attributes["backend"],
                    flow_datastore_backend=self.flow_datastore_backend,
                    force_pull=self.attributes["force_pull"],
                    skip_push_check=self.attributes["skip_push_check"],
                    debug=self.attributes["debug"],
                )
            except Exception as e:
                print(f"[@ollama] Error initializing OllamaManager: {e}")
                raise
            try:
                step_func()
            finally:
                self.ollama_manager.terminate_models()
            if self.attributes["debug"]:
                print(f"[@ollama] process statuses: {self.ollama_manager.processes}")
                print(f"[@ollama] process runtime stats: {self.ollama_manager.stats}")

        return ollama_wrapper
