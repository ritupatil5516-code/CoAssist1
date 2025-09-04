from __future__ import annotations
from typing import Optional
import truststore
from openai import OpenAI

truststore.inject_into_ssl()

DEFAULT_CHAT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

class LLMClient:
    def __init__(self, model: str = DEFAULT_CHAT_MODEL, client: Optional[OpenAI] = None):
        self.model = model
        self.client = client or OpenAI()

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
