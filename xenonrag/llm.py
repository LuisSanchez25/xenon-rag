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

    name: str

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
                 temperature: float = 0.2, max_output_tokens: int = 8000,
                 thinking_budget: int = 0):
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
 
        self.name = model
        self.model = model
        self._types = types
        self.client = genai.Client(api_key=key)
 
        # Low temperature: this is a question-answering tool over a fixed
        # corpus, not a creative one. The same question should give the same
        # answer, which matters most while evaluating.
        #
        # thinking_budget=0 and a generous max_output_tokens together avoid a
        # specific failure. On Gemini 3 models max_output_tokens is a combined
        # budget for reasoning *and* output, and the reasoning expands to fill
        # almost all of it -- so a modest limit produced answers that were a
        # fragment of a sentence, sometimes with the model's own draft notes
        # leaking into the text.
        self._base_cfg = {"temperature": temperature,
                          "max_output_tokens": max_output_tokens}
        self._thinking_budget = thinking_budget
        self._options = [o for o in self._thinking_options()
                         if self._make_config(o) is not None]
        self._option_i = 0
        self.config = self._make_config(self._options[0])
 
    def _thinking_options(self) -> list[dict | None]:
        """Reasoning controls to try, best first, ending with none.
 
        Which parameter caps reasoning changed between model generations, and
        the wrong one is rejected with an opaque 400 rather than ignored:
 
            gemini-3.x   thinking_level="low"   (cannot be disabled outright;
                                                 "none" is refused)
            gemini-2.5   thinking_budget=0
            older        no reasoning to control
 
        Rather than hard-code a mapping that will go stale again, the ladder is
        walked at runtime and the first accepted option is kept for the session.
        """
        return [{"thinking_level": "low"},
                {"thinking_budget": self._thinking_budget},
                None]
 
    def _make_config(self, option: dict | None):
        """Build a request config with the given reasoning control, if any."""
        cfg = dict(self._base_cfg)
        if option:
            try:
                cfg["thinking_config"] = self._types.ThinkingConfig(**option)
            except Exception:                     # field absent in this SDK
                return None
        return self._types.GenerateContentConfig(**cfg)
 
    @staticmethod
    def _extract(resp) -> str:
        """Text from the response, excluding reasoning parts.
 
        `resp.text` can include the model's internal draft when reasoning is
        active, so parts flagged as thoughts are dropped explicitly.
        """
        try:
            parts = resp.candidates[0].content.parts or []
        except (AttributeError, IndexError, TypeError):
            return resp.text or ""
        out = [p.text for p in parts
               if getattr(p, "text", None) and not getattr(p, "thought", False)]
        return "".join(out) if out else (resp.text or "")
 
    def generate(self, prompt: str) -> str:
        def call():
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=self.config
                )
            except Exception as exc:                        # noqa: BLE001
                msg = str(exc)
                rejected = ("INVALID_ARGUMENT" in msg or "400" in msg)
                if not rejected or self._option_i + 1 >= len(self._options):
                    raise
                # This model does not accept the reasoning control we sent.
                # Step down the ladder and keep the working one for the session.
                self._option_i += 1
                nxt = self._options[self._option_i]
                print(f"  note: {self.model} rejected that reasoning setting; "
                      f"falling back to {nxt or 'none'}")
                self.config = self._make_config(nxt)
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=self.config
                )
            reason = ""
            try:
                reason = str(resp.candidates[0].finish_reason or "")
            except (AttributeError, IndexError, TypeError):
                pass
 
            text = self._extract(resp)
 
            if "MAX_TOKENS" in reason:
                # Silently returning a fragment is worse than failing: it looks
                # like a real answer and gets graded as one.
                thoughts = getattr(getattr(resp, "usage_metadata", None),
                                   "thoughts_token_count", None)
                raise LLMError(
                    f"response hit the token ceiling"
                    + (f" after {thoughts} reasoning tokens" if thoughts else "")
                    + "; raise max_output_tokens or lower thinking_budget"
                )
            if reason and "STOP" not in reason and text.strip():
                print(f"  note: finish_reason={reason}")
            return text
 
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

    def __init__(self, model: str = "qwen3:8b",
                 host: str = "http://localhost:11434",
                 num_ctx: int = 8192, temperature: float = 0.2,
                 timeout: int = 300, think: bool = False):
        self.name = f"ollama/{model}"
        self.model = model
        # Qwen3 and other recent models reason before answering. That costs
        # latency and, worse, the reasoning can leak into the reply -- the
        # same failure that produced fragmentary answers from Gemini until
        # its thinking budget was set to zero.
        self.think = think
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
                    "think": self.think,
                    "keep_alive": self.keep_alive,
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
        # Belt and braces: some builds emit <think>...</think> regardless of
        # the flag. Strip it rather than grading the model's notes.
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        return text.strip()


def get_backend(name: str = "gemini", **kwargs) -> LLMBackend:
    """Look up a backend by short name."""
    if name == "gemini":
        return GeminiBackend(**kwargs)
    if name == "ollama":
        return OllamaBackend(**kwargs)
    raise ValueError(f"unknown backend: {name!r} (try 'gemini' or 'ollama')")