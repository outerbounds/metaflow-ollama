from metaflow.decorators import StepDecorator
from metaflow import current
import functools

from .ollama import OllamaManager


class OllamaDecorator(StepDecorator):
    """
    This decorator is used to run Ollama APIs as Metaflow task sidecars.

    User code call
    -----------
    @ollama(
        models=['meta/llama3-8b-instruct', 'meta/llama3-70b-instruct'],
        backend='local'
    )

    Valid backend options
    ---------------------
    - 'local': Run as a separate process on the local task machine.
    - (TODO) 'managed': Outerbounds hosts and selects compute provider.
    - (TODO) 'remote': Spin up separate instance to serve Ollama models.

    Valid model options
    ----------------
        - 'llama3.2'
        - 'llama3.3'
        - any model here https://ollama.com/search

    Parameters
    ----------
    models: list[Ollama]
        List of Ollama containers running models in sidecars.
    backend: str
        Determines where and how to run the Ollama process.
    """

    name = "ollama"
    defaults = {"models": [], "backend": "local", "debug": False}

    def task_decorate(
        self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
    ):
        @functools.wraps(step_func)
        def ollama_wrapper():
            try:
                self.ollama_manager = OllamaManager(
                    models=self.attributes["models"],
                    backend=self.attributes["backend"],
                    debug=self.attributes["debug"],
                )
            except Exception as e:
                print(f"[@ollama] Error initializing OllamaManager: {e}")
                raise
            try:
                step_func()
            finally:
                try:
                    self.ollama_manager.terminate_models()
                except Exception as term_e:
                    print(f"[@ollama] Error during sidecar termination: {term_e}")
            if self.attributes["debug"]:
                print(f"[@ollama] process statuses: {self.ollama_manager.processes}")
                print(f"[@ollama] process runtime stats: {self.ollama_manager.stats}")

        return ollama_wrapper