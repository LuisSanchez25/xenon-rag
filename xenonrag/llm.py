"""Talking to a language model.
 
Everything sits behind LLMBackend so the rest of the project never imports a
vendor SDK. Swapping Gemini for a local model later means adding one class and
changing one line of configuration -- nothing in the prompt building, the
retrieval, or the CLI has to know which model answered.
 
That indirection is not architecture for its own sake. The plan is explicitly
to move to a local model running on the group's GPU once retrieval is tuned,
and to compare the two on the same benchmark. Having both behind one interface
is what makes that comparison a config change rather than a rewrite.
"""
 
from __future__ import annotations
 
import os
import time
from abc import ABC, abstractmethod
 
 
class LLMError(RuntimeError):
    """Raised when a backend cannot produce an answer."""
 
 
class LLMBackend(ABC):
    """Interface every model backend implements."""
 
    model: str
 
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the model's completion for `prompt`."""
 
 
def _retry(fn, attempts: int = 5, base_delay: float = 1.0):
    """Call `fn`, backing off on rate limits.
 
    The free Gemini tier allows a limited number of requests per minute, and an
    evaluation run fires questions in a tight loop, so hitting the limit is
    expected rather than exceptional. Waiting 1s, 2s, 4s, 8s clears it.
    Anything that is not a rate limit is raised immediately -- retrying a bad
    API key just wastes time.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:                      # noqa: BLE001
            msg = str(exc).lower()
            rate_limited = ("429" in msg or "resource_exhausted" in msg
                            or "rate limit" in msg or "quota" in msg)
            if not rate_limited or i == attempts - 1:
                raise
            last = exc
            time.sleep(base_delay * (2 ** i))
    raise LLMError(f"gave up after {attempts} attempts: {last}")
 
 
class GeminiBackend(LLMBackend):
    """Google's hosted models, via the free tier.
 
    Reads GEMINI_API_KEY from the environment. Never hard-code the key or
    commit it -- on a shared server, keep it in a file with mode 600 and
    source it, or export it in your shell profile.
    """
 
    def __init__(self, model: str = "gemini-3.6-flash",
                 temperature: float = 0.2, max_output_tokens: int = 1500):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:                    # pragma: no cover
            raise LLMError(
                "google-genai is not installed. pip install google-genai"
            ) from exc
 
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Get a key from Google AI Studio and "
                "export it, e.g. `export GEMINI_API_KEY=$(grep -oP '(?<==).*' "
                "~/.config/xenonrag/env)`"
            )
 
        self.model = model
        self._types = types
        self.client = genai.Client(api_key=key)
        # Low temperature: this is a question-answering tool over a fixed
        # corpus, not a creative one. We want the same question to give the
        # same answer, especially while evaluating.
        self.config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
 
    def generate(self, prompt: str) -> str:
        def call():
            resp = self.client.models.generate_content(
                model=self.model, contents=prompt, config=self.config
            )
            return resp.text or ""
 
        text = _retry(call)
        if not text.strip():
            raise LLMError("model returned an empty response")
        return text
 
 
class OllamaBackend(LLMBackend):
    """A local model served by Ollama.
 
    Used once retrieval is tuned, so the whole system can run on the group's
    hardware with nothing leaving the machine. `num_ctx` matters here: Ollama
    defaults to 4096 tokens, and a prompt carrying eight expanded excerpts can
    exceed that. Anything past the window is silently dropped -- the same class
    of quiet failure as the embedding model's 512-token limit.
    """
 
    def __init__(self, model: str = "qwen2.5-coder:7b",
                 host: str = "http://localhost:11434",
                 num_ctx: int = 8192, temperature: float = 0.2,
                 timeout: int = 300):
        self.model = f"ollama/{model}"
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout
 
    def generate(self, prompt: str) -> str:
        import requests
 
        def call():
            r = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_ctx": self.num_ctx,
                                "temperature": self.temperature},
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json().get("response", "")
 
        text = _retry(call, attempts=2)
        if not text.strip():
            raise LLMError("model returned an empty response")
        return text
 
 
def get_backend(model: str = "gemini", **kwargs) -> LLMBackend:
    """Look up a backend by short name."""
    if model == "gemini":
        return GeminiBackend(**kwargs)
    if model == "ollama":
        return OllamaBackend(**kwargs)
    raise ValueError(f"unknown backend: {model!r} (try 'gemini' or 'ollama')")