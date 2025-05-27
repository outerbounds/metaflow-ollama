from metaflow import FlowSpec, step, Config, pypi, card, current, resources, retry
from metaflow.cards import Markdown


class OllamaBenchmarkStarter(FlowSpec):

    config = Config("config", default="config.json")

    @step
    def start(self):
        print(type(self._datastore.parent_datastore._storage_impl))
        print(dir(self._datastore.parent_datastore._storage_impl))
        self.models = self.config.models
        self.next(self.prompt, foreach="models")

    @retry
    @resources(cpu=2, memory=8000)
    @pypi(packages={"ollama": ""})
    @step
    def prompt(self):
        import time
        from ollama import chat  # type: ignore
        from ollama import ChatResponse
        from metaflow.plugins.ollama import OllamaManager  # type: ignore

        # Start server inline instead of automating in the decorator.
        ollama_manager = OllamaManager(
            models=[self.input],
            flow_datastore_backend=self._datastore.parent_datastore._storage_impl,
        )
        # unique model per foreach task, in this case.

        # Run a workload.
        t0 = time.time()
        self.response: ChatResponse = chat(
            model=self.input, messages=[self.config.message]
        )
        tf = time.time()

        print(f"Client-side runtime {round(tf-t0, 3)} seconds.")

        # Clean up processes.
        ollama_manager.terminate_models()

        self.next(self.join)

    @pypi(packages={"ollama": ""})
    @card
    @step
    def join(self, inputs):
        current.card.append(
            Markdown(
                f"Each model will be listed, including its response to the following message: {self.config.message}"
            )
        )
        for inp_task in inputs:
            current.card.append(Markdown(f"## {inp_task.input}"))
            current.card.append(Markdown(inp_task.response["message"]["content"]))
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    OllamaBenchmarkStarter()
