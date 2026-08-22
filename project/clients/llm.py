from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-3-flash-preview"


class LLMClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content
