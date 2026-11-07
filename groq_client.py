from pathlib import Path

from openai import OpenAI
from llm_client import LLMClient


class GroqClient(LLMClient):

    def __init__(
        self,
        api_key_file: str = "groq_api.txt",
        model: str = "openai/gpt-oss-20b"
    ):

        project_dir = Path(__file__).resolve().parent
        api_key_path = project_dir / api_key_file

        if not api_key_path.exists():
            raise FileNotFoundError(
                f"API key file not found: {api_key_path}"
            )

        api_key = api_key_path.read_text(
            encoding="utf-8"
        ).strip()

        if not api_key:
            raise ValueError("API key file is empty.")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=30.0,
            max_retries=0
        )

        self.model = model

    def generate(self, prompt: str) -> str:

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content or ""