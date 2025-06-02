from metaflow import FlowSpec, step, pypi, ollama, kubernetes
from metaflow.profilers import gpu_profile


class HelloOllama(FlowSpec):

    @gpu_profile(interval=1)
    @kubernetes(gpu=1)
    @pypi(packages={"ollama": ""})
    @ollama(models=["llama3.2:1b", "qwen:0.5b"], debug=True)
    @step
    def start(self):
        """
        An introduction to using the @ollama decorator.

        Notice that the @kubernetes decorator uses default base image.
        We install the ollama python client using normal @pypi usage.
        This dependency management approach contrasts that of the end step.

        This step downloads two models.
        It also turns on debugging for verbose logs.
        """
        from ollama import chat
        from ollama import ChatResponse

        response_llama: ChatResponse = chat(
            model="llama3.2:1b",
            messages=[
                {
                    "role": "user",
                    "content": "What are the leading Chinese tech companies?",
                },
            ],
        )
        response_qwen: ChatResponse = chat(
            model="qwen:0.5b",
            messages=[
                {
                    "role": "user",
                    "content": "What are the leading Chinese tech companies?",
                },
            ],
        )
        print(
            f"\n\n Response from llama2:1b {response_llama['message']['content']} \n\n"
        )
        print(
            f"\n\n Response from qwen:0.5b {response_qwen['message']['content']} \n\n"
        )
        self.next(self.end)

    @pypi(
        disabled=True
    )  # Tell --environment=fast-bakery to ignore building for this step.
    @kubernetes(gpu=1, image="docker.io/eddieob/ollama-metaflow-task:gpu")
    @ollama(models=["qwen:0.5b"])
    @step
    def end(self):
        """
        As noted in the start step docstrings,
        this step is different in its use of a custom image,
        which is defined in this repo's examples/dependencies/Dockerfile.cpu.
        """
        from ollama import chat
        from ollama import ChatResponse

        response_qwen: ChatResponse = chat(
            model="qwen:0.5b",
            messages=[
                {
                    "role": "user",
                    "content": "What are the leading Chinese tech companies?",
                },
            ],
        )
        print(
            f"\n\n Response from qwen:0.5b {response_qwen['message']['content']} \n\n"
        )


if __name__ == "__main__":
    HelloOllama()
