from llama_cpp import Llama
from llm.base_llm import BaseLLM


class LlammaLocal(BaseLLM):
    _instance = None

    @classmethod
    def get_instance(cls, model_path: str):
        if cls._instance is None:
            print(f"Loading model from {model_path}... Please wait.")
            cls._instance = cls(model_path)
            print("Model loaded successfully!")
        return cls._instance

    def __init__(self, model_path: str):
        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )

    def generate(self,prompt: str,max_tokens: int = 200,temperature: float = 0.01,top_p: float = 0.9,repeat_penalty: float = 1.1) -> str:
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty
        )
        return response["choices"][0]["text"].strip()