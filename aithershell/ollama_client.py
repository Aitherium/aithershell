"""
Ollama client for AitherShell — local LLM inference, no Genesis required.

Talks directly to Ollama's HTTP API at localhost:11434 (or wherever).
Supports streaming, model selection, and a healthcheck for `aither init`.

This is the pillar of local-only mode: with this module, AitherShell can
generate responses without any AitherOS containers running.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class OllamaModel:
    """One model installed in Ollama."""
    name: str         # e.g. "nemotron-orchestrator:8b"
    size_bytes: int   # 4_500_000_000 = 4.5 GB
    modified: str     # ISO timestamp

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)


class OllamaError(Exception):
    """Raised when Ollama can't fulfill a request."""
    pass


class OllamaClient:
    """Direct HTTP client for Ollama at localhost:11434.

    Usage:
        client = OllamaClient("http://localhost:11434")
        if await client.is_available():
            async for chunk in client.chat_stream("nemotron-orchestrator:8b", "hello"):
                print(chunk, end="", flush=True)
    """

    def __init__(self, url: str = "http://localhost:11434", timeout: float = 120.0):
        # Strip trailing slash
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ---------- Discovery ----------

    async def is_available(self) -> bool:
        """Quick check: is Ollama running?"""
        try:
            client = await self._get_client()
            r = await client.get(f"{self.url}/api/tags", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[OllamaModel]:
        """List installed models. Returns [] if Ollama unreachable."""
        try:
            client = await self._get_client()
            r = await client.get(f"{self.url}/api/tags", timeout=5.0)
            r.raise_for_status()
            data = r.json()
            return [
                OllamaModel(
                    name=m["name"],
                    size_bytes=m.get("size", 0),
                    modified=m.get("modified_at", ""),
                )
                for m in data.get("models", [])
            ]
        except Exception as e:
            logger.debug(f"list_models failed: {e}")
            return []

    async def has_model(self, name: str) -> bool:
        """Check if a model is installed. Matches exact name or family prefix."""
        models = await self.list_models()
        names = {m.name for m in models}
        if name in names:
            return True
        # Allow "nemotron-orchestrator:8b" to match "nemotron-orchestrator:8b-q4_K_M" etc
        family = name.split(":")[0]
        return any(n.split(":")[0] == family for n in names)

    # ---------- Pull (download) ----------

    async def pull_model(self, name: str) -> AsyncIterator[Dict]:
        """Pull a model from Ollama registry. Streams progress events.

        Each yielded dict has keys like:
          {'status': 'pulling manifest'}
          {'status': 'downloading', 'digest': 'sha256:...', 'total': 4500000000, 'completed': 12345}
          {'status': 'success'}
        """
        client = await self._get_client()
        try:
            async with client.stream(
                "POST",
                f"{self.url}/api/pull",
                json={"name": name, "stream": True},
                timeout=None,  # downloads can take 10+ min
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except httpx.HTTPError as e:
            raise OllamaError(f"Failed to pull {name}: {e}") from e

    # ---------- Generation ----------

    async def chat_stream(
        self,
        model: str,
        message: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        history: Optional[List[Dict]] = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens as plain text strings.

        Args:
            model: Ollama model name (e.g. "nemotron-orchestrator:8b")
            message: User message
            system: Optional system prompt
            temperature: 0.0-2.0, sampling temperature
            max_tokens: max tokens to generate (None = model default)
            history: Optional [{role, content}, ...] of prior turns

        Yields:
            Text chunks as they arrive from the model.

        Raises:
            OllamaError: on connection/protocol errors. Caller should handle
                         and optionally fall back to cloud.
        """
        messages: List[Dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        options: Dict = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options,
        }

        client = await self._get_client()
        try:
            async with client.stream(
                "POST",
                f"{self.url}/api/chat",
                json=body,
                timeout=self.timeout,
            ) as response:
                if response.status_code == 404:
                    raise OllamaError(
                        f"Model '{model}' not installed. Run: ollama pull {model}"
                    )
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract message content
                    msg = chunk.get("message", {})
                    content = msg.get("content", "")
                    if content:
                        yield content

                    # Stop on done flag
                    if chunk.get("done"):
                        break
        except httpx.HTTPError as e:
            raise OllamaError(f"Ollama request failed: {e}") from e

    async def generate(
        self,
        model: str,
        message: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Non-streaming convenience: collect all chunks into one string."""
        chunks: List[str] = []
        async for chunk in self.chat_stream(model, message, system, temperature, max_tokens):
            chunks.append(chunk)
        return "".join(chunks)


# ---------- Effort → model mapping ----------

# Maps effort levels (1-10) to ollama model names. Lower effort = smaller, faster model.
# Users override via config.backends.local.model
DEFAULT_EFFORT_MODELS = {
    # Effort 1-2: reflex (fastest, ~3B params)
    1: "llama3.2:3b",
    2: "llama3.2:3b",
    # Effort 3-6: balanced orchestrator (8B Nemotron)
    3: "nemotron-orchestrator:8b",
    4: "nemotron-orchestrator:8b",
    5: "nemotron-orchestrator:8b",
    6: "nemotron-orchestrator:8b",
    # Effort 7+: would normally route to cloud; if forced local, use what we have
    7: "nemotron-orchestrator:8b",
    8: "nemotron-orchestrator:8b",
    9: "nemotron-orchestrator:8b",
    10: "nemotron-orchestrator:8b",
}


def model_for_effort(effort: int, override: Optional[str] = None) -> str:
    """Select an ollama model name for the given effort level.

    If `override` is set, return it unchanged (user explicitly picked a model).
    """
    if override:
        return override
    return DEFAULT_EFFORT_MODELS.get(effort, "nemotron-orchestrator:8b")


# ---------- Sync helpers for non-async callers ----------

def quick_check(url: str = "http://localhost:11434") -> bool:
    """Synchronous Ollama healthcheck. Useful for `aither init` setup wizard."""
    async def _check() -> bool:
        client = OllamaClient(url)
        try:
            return await client.is_available()
        finally:
            await client.close()

    try:
        return asyncio.run(_check())
    except Exception:
        return False


def list_models_sync(url: str = "http://localhost:11434") -> List[OllamaModel]:
    """Synchronous model list. Useful for `aither init` setup wizard."""
    async def _list() -> List[OllamaModel]:
        client = OllamaClient(url)
        try:
            return await client.list_models()
        finally:
            await client.close()

    try:
        return asyncio.run(_list())
    except Exception:
        return []
