import os
from dotenv import load_dotenv
from google import genai
from llm.base_llm import BaseLLM


class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        load_dotenv()

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text.strip()