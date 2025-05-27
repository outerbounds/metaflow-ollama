from metaflow import FlowSpec, step, ollama, pypi, resources, card, current
from metaflow.profilers import gpu_profile
from metaflow.cards import ProgressBar


class OllamaGPU(FlowSpec):

    @card
    @pypi(packages={"ollama": ""})
    @ollama(models=["llama3.2:3b"], debug=True)
    @resources(gpu=1)
    @gpu_profile(interval=1)
    @step
    def start(self):
        from ollama import chat
        from ollama import ChatResponse

        pbar = ProgressBar(max=250, label="iter")
        current.card.append(pbar)
        for i in range(250):
            response_llama: ChatResponse = chat(
                model="llama3.2:3b",
                messages=[
                    {
                        "role": "user",
                        "content": "What are the leading Chinese tech companies?",
                    },
                ],
            )
            pbar.update(i)
            current.card.refresh()
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    OllamaGPU()
