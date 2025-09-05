import os, truststore
truststore.inject_into_ssl()
from goldmansachs.openai import OpenAI

DEFAULT_CHAT_MODEL = os.environ.get("CHAT_MODEL","meta-llama/Llama-3.3-70B-Instruct")

class LLMClient:
    def __init__(self, model: str = DEFAULT_CHAT_MODEL):
        self.client = OpenAI()
        self.model = model
    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
        )
        return resp.choices[0].message.content or ""
