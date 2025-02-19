from metaflow import FlowSpec, step, ollama, pypi

class HelloOllama(FlowSpec):

    @pypi(packages={'ollama': ''})
    @ollama(models=['llama3.2:1b', 'qwen:0.5b'])
    @step
    def start(self):
        from ollama import chat
        from ollama import ChatResponse

        response_llama: ChatResponse = chat(
            model='llama3.2:1b', 
            messages=[
                {
                    'role': 'user',
                    'content': 'What are the leading Chinese tech companies?',
                },
            ]
        )
        response_qwen: ChatResponse = chat(
            model='qwen:0.5b', 
            messages=[
                {
                    'role': 'user',
                    'content': 'What are the leading Chinese tech companies?',
                },
            ]
        )
        print(f"\n\n Response from llama2:1b {response_llama['message']['content']} \n\n")
        print(f"\n\n Response from qwen:0.5b {response_qwen['message']['content']} \n\n")

        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    HelloOllama()