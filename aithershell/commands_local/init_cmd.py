"""
`aither init` — interactive setup wizard.

Detects the user's environment, configures backends, links to portal,
and pulls the recommended local model. Designed to take a complete
beginner from "downloaded the binary" to "running real prompts" in
under 5 minutes.

Flow:
  1. Detect Ollama (offer to install if missing — print URL)
  2. Detect GPU (just informational)
  3. Offer to pull nemotron-orchestrator:8b
  4. Offer to link AitherOS Portal for cloud fallback
  5. Configure effort routing threshold
  6. Save config and run a smoke test
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from typing import Optional, Tuple

import click

from aithershell.config import AitherConfig, load_config, save_config
from aithershell.ollama_client import OllamaClient, OllamaError


DEFAULT_LOCAL_MODEL = "nemotron-orchestrator:8b"
RECOMMENDED_FALLBACK_MODEL = "llama3.2:3b"  # tiny, runs anywhere
PORTAL_DEFAULT = "https://portal.aitherium.com"


# ---------- Detection helpers ----------

def detect_ollama_binary() -> Optional[str]:
    """Find ollama CLI on PATH."""
    return shutil.which("ollama")


async def detect_ollama_running(url: str = "http://localhost:11434") -> bool:
    """Check if Ollama HTTP API is responding."""
    client = OllamaClient(url)
    try:
        return await client.is_available()
    finally:
        await client.close()


def detect_gpu() -> Tuple[bool, str]:
    """Return (has_gpu, description)."""
    # Try nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                first_gpu = result.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first_gpu.split(",")]
                if len(parts) >= 2:
                    return True, f"NVIDIA {parts[0]} ({int(parts[1])//1024} GB VRAM)"
        except Exception:
            pass

    # Try torch.cuda
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            return True, f"{name} ({mem} GB VRAM via torch)"
    except Exception:
        pass

    # macOS Metal
    if sys.platform == "darwin":
        return True, "Apple Metal (unified memory)"

    return False, "CPU only"


# ---------- Wizard ----------

def _print_banner() -> None:
    click.secho("\n" + "═" * 60, fg="cyan")
    click.secho("  AitherShell — Local Agent Setup", fg="cyan", bold=True)
    click.secho("═" * 60 + "\n", fg="cyan")


def _print_section(title: str) -> None:
    click.secho(f"\n→ {title}", fg="bright_cyan", bold=True)


async def run_init_async(force: bool = False) -> int:
    """Run the init wizard. Returns 0 on success, non-zero on cancel/error."""
    _print_banner()

    cfg = load_config()

    # ---- 1. Detect environment ----
    _print_section("Detecting environment")

    ollama_bin = detect_ollama_binary()
    if ollama_bin:
        click.secho(f"  ✓ Ollama CLI found: {ollama_bin}", fg="green")
    else:
        click.secho("  ✗ Ollama CLI not found on PATH", fg="yellow")

    ollama_url = cfg.backends.get("local", {}).get("url", "http://localhost:11434")
    ollama_up = await detect_ollama_running(ollama_url)
    if ollama_up:
        click.secho(f"  ✓ Ollama API responding at {ollama_url}", fg="green")
    else:
        click.secho(f"  ✗ Ollama API not responding at {ollama_url}", fg="yellow")

    has_gpu, gpu_desc = detect_gpu()
    icon = "✓" if has_gpu else "ℹ"
    color = "green" if has_gpu else "white"
    click.secho(f"  {icon} GPU: {gpu_desc}", fg=color)

    # ---- 2. Install instructions if no Ollama ----
    if not ollama_up:
        click.secho("\n  Ollama is required for local execution. Install:", fg="yellow")
        if sys.platform == "win32":
            click.secho("    https://ollama.com/download/windows", fg="bright_blue")
        elif sys.platform == "darwin":
            click.secho("    https://ollama.com/download/mac", fg="bright_blue")
            click.secho("    or: brew install ollama", fg="white")
        else:
            click.secho("    curl -fsSL https://ollama.com/install.sh | sh", fg="bright_blue")
        click.secho("\n  Then run `aither init` again.", fg="yellow")

        if not click.confirm("\n  Skip local setup and configure cloud only?", default=False):
            return 1
        # Skip to portal setup
        return await _setup_cloud_only(cfg)

    # ---- 3. Model selection / pull ----
    _print_section("Local model")

    client = OllamaClient(ollama_url)
    try:
        existing = await client.list_models()
        if existing:
            click.secho(f"  Installed models ({len(existing)}):", fg="white")
            for m in existing[:10]:
                click.secho(f"    • {m.name} ({m.size_gb:.1f} GB)", fg="white")

        # Suggest a model
        recommended = DEFAULT_LOCAL_MODEL if has_gpu else RECOMMENDED_FALLBACK_MODEL
        already_have = await client.has_model(recommended)

        if already_have:
            click.secho(f"  ✓ Recommended model already installed: {recommended}", fg="green")
            chosen_model = recommended
        else:
            click.secho(f"\n  Recommended: {recommended}", fg="bright_yellow")
            if has_gpu:
                click.secho("    (8B params, ~4.5 GB, runs at 30-60 tok/s on GPU)", fg="white")
            else:
                click.secho("    (3B params, ~2 GB, runs at 5-15 tok/s on CPU)", fg="white")

            if click.confirm(f"\n  Pull {recommended} now?", default=True):
                chosen_model = await _pull_with_progress(client, recommended)
                if not chosen_model:
                    click.secho("  Pull failed. Continuing without local model.", fg="yellow")
                    chosen_model = recommended  # set anyway so user can retry later
            else:
                chosen_model = recommended

        # Persist local backend
        cfg.backends["local"] = {
            "type": "ollama",
            "url": ollama_url,
            "model": chosen_model,
            "max_effort": cfg.backends.get("local", {}).get("max_effort", 6),
        }

    finally:
        await client.close()

    # ---- 4. Portal linking ----
    _print_section("AitherOS Portal (cloud fallback)")

    click.secho(
        "  For high-effort tasks (effort 7+), AitherShell can route to\n"
        "  the AitherOS Portal cloud (claude/gpt). This is optional.",
        fg="white",
    )

    if click.confirm("\n  Link an AitherOS Portal account?", default=False):
        from aithershell.commands_local.link_portal import link_portal_async
        portal_url = click.prompt("  Portal URL", default=PORTAL_DEFAULT)
        await link_portal_async(cfg, portal_url)

    # ---- 5. Routing config ----
    _print_section("Routing")

    threshold = cfg.routing.get("effort_threshold", 6)
    click.secho(
        f"  Effort 1-{threshold}: local model ({cfg.backends['local']['model']})",
        fg="white",
    )
    has_cloud_key = bool(cfg.backends.get("cloud", {}).get("api_key"))
    if has_cloud_key:
        click.secho(f"  Effort {threshold+1}-10: cloud (Portal)", fg="white")
    else:
        click.secho(
            f"  Effort {threshold+1}-10: cloud (NOT configured — will error)",
            fg="yellow",
        )

    if click.confirm("\n  Customize routing threshold?", default=False):
        new_threshold = click.prompt(
            "  Effort threshold (1-10, requests above this go to cloud)",
            type=click.IntRange(1, 10),
            default=threshold,
        )
        cfg.routing["effort_threshold"] = new_threshold

    # ---- 6. Save ----
    save_config(cfg)
    click.secho("\n  ✓ Config saved to ~/.aither/config.yaml", fg="green")

    # ---- 7. Smoke test ----
    if ollama_up and click.confirm("\n  Run a smoke test (`Say hello`)?", default=True):
        await _smoke_test(cfg)

    # ---- Done ----
    click.secho("\n" + "═" * 60, fg="cyan")
    click.secho("  Ready! Try:", fg="cyan", bold=True)
    click.secho('    aither "Write a Python CSV parser"', fg="white")
    click.secho('    aither --effort 9 "Design a distributed task queue"', fg="white")
    click.secho("═" * 60 + "\n", fg="cyan")
    return 0


async def _pull_with_progress(client: OllamaClient, model: str) -> Optional[str]:
    """Pull a model, show progress. Returns model name on success, None on failure."""
    click.secho(f"\n  Pulling {model}...", fg="white")
    last_pct = -1
    try:
        async for event in client.pull_model(model):
            status = event.get("status", "")
            total = event.get("total")
            completed = event.get("completed")
            if total and completed:
                pct = int(completed * 100 / total)
                if pct != last_pct and pct % 5 == 0:
                    bar_len = 30
                    filled = int(bar_len * pct / 100)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    click.echo(
                        f"\r  [{bar}] {pct:3d}%  {status[:30]:30}",
                        nl=False,
                    )
                    last_pct = pct
            elif status:
                click.echo(f"\r  ◦ {status:60}", nl=False)
        click.secho(f"\n  ✓ Pulled {model}", fg="green")
        return model
    except OllamaError as e:
        click.secho(f"\n  ✗ Pull failed: {e}", fg="red")
        return None
    except KeyboardInterrupt:
        click.secho("\n  ✗ Pull cancelled", fg="yellow")
        return None


async def _smoke_test(cfg: AitherConfig) -> None:
    """Generate a quick response to verify everything works."""
    backend_name, backend = cfg.select_backend(effort=3)
    if backend.type != "ollama":
        click.secho("  Skipping smoke test (no local backend).", fg="yellow")
        return

    client = OllamaClient(backend.url, timeout=30.0)
    click.secho(f"\n  [{backend_name}:{backend.model}] ", fg="bright_black", nl=False)
    try:
        async for chunk in client.chat_stream(
            backend.model,
            "Say hello in one short sentence.",
            temperature=0.5,
            max_tokens=50,
        ):
            click.echo(chunk, nl=False)
        click.echo()
        click.secho("  ✓ Smoke test passed", fg="green")
    except OllamaError as e:
        click.secho(f"\n  ✗ Smoke test failed: {e}", fg="red")
    finally:
        await client.close()


async def _setup_cloud_only(cfg: AitherConfig) -> int:
    """Cloud-only configuration path (when Ollama unavailable)."""
    _print_section("Cloud-only setup")
    cfg.routing["mode"] = "cloud_only"

    from aithershell.commands_local.link_portal import link_portal_async
    portal_url = click.prompt("  Portal URL", default=PORTAL_DEFAULT)
    linked = await link_portal_async(cfg, portal_url)

    if linked:
        save_config(cfg)
        click.secho("\n  ✓ Cloud-only mode configured.", fg="green")
        return 0
    click.secho("\n  ✗ Setup incomplete.", fg="red")
    return 1


def run_init(force: bool = False) -> int:
    """Sync entry point for CLI."""
    try:
        return asyncio.run(run_init_async(force))
    except KeyboardInterrupt:
        click.secho("\n\n  Cancelled. Run `aither init` again anytime.", fg="yellow")
        return 130
