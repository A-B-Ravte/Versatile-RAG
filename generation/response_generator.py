from llm.base_llm import BaseLLM


class ResponseGenerator:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate(self, prompt: str, **kwargs) -> str:
        return self.llm.generate(prompt, **kwargs)