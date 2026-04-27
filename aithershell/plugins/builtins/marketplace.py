"""
Model Marketplace Plugin for AitherShell
==========================================

Browse, search, download, and deploy models from HuggingFace, CivitAI,
and the local Ollama library.

Usage:
    /models                           — Show installed models
    /models search <query>            — Search all registries
    /models search --hf <query>       — Search HuggingFace only
    /models search --civitai <query>  — Search CivitAI only
    /models search --ollama <query>   — Search Ollama library
    /models download <id>             — Download/pull a model
    /models download <id> --source hf — Specify source registry
    /models deploy <id>               — Deploy model to cloud GPU
    /models recommend                 — Get recommendations for your hardware
    /models info <id>                 — Model details and metadata

Aliases: /market, /hub
"""

import json
import os
from typing import Any, Dict, List, Optional

from aithershell.plugins import SlashCommand

try:
    from aithershell.auth import AuthStore
except ImportError:
    AuthStore = None  # type: ignore


def _genesis_url() -> str:
    return os.environ.get("AITHER_GENESIS_URL", "http://localhost:8100")


def _headers() -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if AuthStore:
        token = AuthStore.get_active_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        profile = AuthStore.get_active_profile() if hasattr(AuthStore, "get_active_profile") else None
        if profile and profile.get("tenant_id"):
            headers["X-Tenant-ID"] = profile["tenant_id"]
    return headers


class MarketplacePlugin(SlashCommand):
    name: str = "models"
    aliases: List[str] = ["market", "hub"]
    description: str = "Browse, search, download, and deploy models"
    category: str = "ai"

    async def run(self, args: List[str], ctx: Dict[str, Any]) -> Optional[str]:
        if not args:
            return await self._installed(ctx)

        sub = args[0].lower()
        dispatch = {
            "search": self._search,
            "find": self._search,
            "download": self._download,
            "pull": self._download,
            "deploy": self._deploy,
            "recommend": self._recommend,
            "info": self._info,
            "installed": self._installed,
            "list": self._installed,
            "help": self._help,
        }

        handler = dispatch.get(sub)
        if handler:
            return await handler(args[1:], ctx)

        # Default: treat as search query
        return await self._search(args, ctx)

    async def _search(self, args: List[str], ctx: Dict[str, Any]) -> str:
        import httpx

        source = None
        query_parts = []
        i = 0
        while i < len(args):
            if args[i] in ("--hf", "--huggingface"):
                source = "huggingface"
            elif args[i] in ("--civitai", "--cai"):
                source = "civitai"
            elif args[i] in ("--ollama", "--local"):
                source = "ollama"
            else:
                query_parts.append(args[i])
            i += 1

        query = " ".join(query_parts) if query_parts else ""
        if not query:
            return "Usage: `/models search <query>` — add `--hf`, `--civitai`, or `--ollama` to filter"

        params = {"q": query, "limit": 10}
        if source:
            params["source"] = source

        async with httpx.AsyncClient(base_url=_genesis_url(), headers=_headers(), timeout=30) as c:
            resp = await c.get("/marketplace/search", params=params)

        if resp.status_code != 200:
            return f"❌ Search failed: {resp.status_code}"

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return f"No models found for **{query}**"

        lines = [f"🔍 **{data.get('total', len(results))} models** for *{query}*\n"]
        for m in results[:10]:
            name = m.get("name") or m.get("id", "?")
            source_tag = m.get("source", "")
            downloads = m.get("downloads", "")
            dl_str = f"  ⬇ {downloads}" if downloads else ""
            lines.append(f"  `{name}` [{source_tag}]{dl_str}")
            if m.get("description"):
                desc = m["description"][:80]
                lines.append(f"    {desc}")

        return "\n".join(lines)

    async def _download(self, args: List[str], ctx: Dict[str, Any]) -> str:
        import httpx

        if not args:
            return "Usage: `/models download <model_id>` — e.g. `/models download llama3.2:8b`"

        model_id = args[0]
        source = None
        if len(args) > 1 and args[1] == "--source" and len(args) > 2:
            source = args[2]

        body: Dict[str, Any] = {"model_id": model_id}
        if source:
            body["source"] = source

        async with httpx.AsyncClient(base_url=_genesis_url(), headers=_headers(), timeout=30) as c:
            resp = await c.post("/marketplace/download", json=body)

        if resp.status_code != 200:
            return f"❌ Download failed: {resp.status_code} — {resp.text[:200]}"

        data = resp.json()
        if data.get("status") == "already_available":
            return f"✅ **{model_id}** is already available locally"
        return f"⬇️ Download started for **{model_id}** — task: `{data.get('task_id', '?')}`"

    async def _deploy(self, args: List[str], ctx: Dict[str, Any]) -> str:
        import httpx

        if not args:
            return "Usage: `/models deploy <model_id>` — deploys to a cloud GPU via your account"

        model_id = args[0]
        body: Dict[str, Any] = {"model_id": model_id}

        async with httpx.AsyncClient(base_url=_genesis_url(), headers=_headers(), timeout=30) as c:
            resp = await c.post("/marketplace/deploy", json=body)

        if resp.status_code != 200:
            return f"❌ Deploy failed: {resp.status_code} — {resp.text[:200]}"

        data = resp.json()
        session = data.get("session_id", "?")
        return f"🚀 Deployment started for **{model_id}** — session: `{session}`\nUse `/cloud status {session}` to track progress"

    async def _recommend(self, args: List[str], ctx: Dict[str, Any]) -> str:
        import httpx

        async with httpx.AsyncClient(base_url=_genesis_url(), headers=_headers(), timeout=15) as c:
            resp = await c.get("/marketplace/recommendations")

        if resp.status_code != 200:
            return f"❌ Failed to get recommendations: {resp.status_code}"

        data = resp.json()
        models = data.get("recommended", [])
        if not models:
            return "No recommendations available"

        lines = ["💡 **Recommended Models**\n"]
        for m in models:
            lines.append(f"  `{m.get('id', '?')}` — {m.get('reason', '')}")
        return "\n".join(lines)

    async def _info(self, args: List[str], ctx: Dict[str, Any]) -> str:
        if not args:
            return "Usage: `/models info <model_id>`"
        
        import httpx
        model_id = args[0]
        
        try:
            async with httpx.AsyncClient(base_url=_genesis_url(), headers=_headers(), timeout=15) as c:
                # Try to fetch model details
                resp = await c.get(f"/marketplace/models/{model_id}")
                
                if resp.status_code != 200:
                    # Fallback: search for the model
                    search_resp = await c.get("/marketplace/search", params={"q": model_id})
                    if search_resp.status_code == 200:
                        data = search_resp.json()
                        results = data.get("results", [])
                        for item in results:
                            if item.get("id", "").lower() == model_id.lower():
                                return self._format_model_info(item)
                    return f"ℹ️ Model `{model_id}` not found. Use `/models search` to discover models."
                
                data = resp.json()
                return self._format_model_info(data)
        except Exception as e:
            return f"❌ Error fetching model info: {e}"
    
    def _format_model_info(self, model: Dict[str, Any]) -> str:
        """Format model details as markdown."""
        lines = [f"📋 **{model.get('name', model.get('id', 'Unknown'))}**\n"]
        
        if model.get("description"):
            lines.append(f"  {model['description']}\n")
        
        if model.get("size_display"):
            lines.append(f"  **Size:** {model['size_display']}")
        
        if model.get("context_length"):
            lines.append(f"  **Context:** {model['context_length']:,} tokens")
        
        if model.get("provider"):
            lines.append(f"  **Provider:** {model['provider']}")
        
        if model.get("license"):
            lines.append(f"  **License:** {model['license']}")
        
        if model.get("rating"):
            lines.append(f"  **Rating:** {'⭐' * int(model['rating'])} ({model['rating']:.1f})")
        
        if model.get("downloads"):
            lines.append(f"  **Downloads:** {model['downloads']:,}")
        
        if model.get("tags"):
            tags = model['tags'] if isinstance(model['tags'], list) else [model['tags']]
            lines.append(f"  **Tags:** {', '.join(tags)}")
        
        if model.get("url"):
            lines.append(f"\n  🔗 {model['url']}")
        
        return "\n".join(lines)

    async def _installed(self, ctx: Dict[str, Any]) -> str:
        import httpx

        async with httpx.AsyncClient(base_url=_genesis_url(), headers=_headers(), timeout=15) as c:
            resp = await c.get("/marketplace/installed")

        if resp.status_code != 200:
            return f"❌ Failed to list models: {resp.status_code}"

        data = resp.json()
        models = data.get("models", [])
        if not models:
            return "No models installed. Use `/models search <query>` to find models."

        lines = [f"📦 **{len(models)} installed models**\n"]
        for m in models:
            name = m.get("name") or m.get("id", "?")
            size = m.get("size_display", "")
            lines.append(f"  `{name}` {size}")
        return "\n".join(lines)

    async def _help(self, args: List[str], ctx: Dict[str, Any]) -> str:
        return self.get_help()

    def get_help(self) -> str:
        return """📦 **Model Marketplace**

| Command | Description |
|---------|-------------|
| `/models` | List installed models |
| `/models search <query>` | Search all registries |
| `/models search --hf <q>` | Search HuggingFace only |
| `/models search --ollama <q>` | Search Ollama library |
| `/models download <id>` | Download / pull a model |
| `/models deploy <id>` | Deploy to cloud GPU |
| `/models recommend` | Hardware-aware recommendations |
"""
