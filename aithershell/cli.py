"""
AitherShell CLI Entry Point
============================

Click-based CLI for AitherShell with:
- Single query mode: aither "query"
- Interactive REPL: aither
- Configuration: aither --config
- Plugins: aither --plugins
- Multiple output formats: --print, --json, --private
- Persona control: --will persona-name
- Effort levels: --effort 1-10
- Safety modes: --safety paranoid/strict/professional/relaxed
"""

import asyncio
import json
import logging
import os
import sys
from typing import Optional

import click

from aithershell.config import save_default_config, load_config, AitherConfig
from aithershell.shell import run_repl
from aithershell.commands import execute_command, CommandError
from aithershell.genesis_client import GenesisClient, GenesisError
from aithershell.license import validate_license, enforce_license, get_tier
# Engine imports (merged from former adk.* namespace)
from aithershell.llm.ollama import OllamaProvider
from aithershell.llm.base import Message
from aithershell.pairing import PairingManager


# Local engine failure type — triggers cloud fallback in routing logic
class OllamaError(Exception):
    """Local engine failure — triggers cloud fallback if configured."""
    pass

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@click.group(invoke_without_command=True)
@click.argument("query", required=False, nargs=-1)
@click.option("--print", "output_format", flag_value="text", help="Plain text output")
@click.option("--json", "output_format", flag_value="json", help="JSON output")
@click.option("--private", is_flag=True, help="Private mode (no logging)")
@click.option("--effort", type=click.IntRange(1, 10), help="Effort level (1-10)")
@click.option(
    "--safety",
    type=click.Choice(["paranoid", "strict", "professional", "relaxed"]),
    help="Safety level",
)
@click.option("--will", help="Persona name (e.g., aither-prime)")
@click.option("--model", help="Model override")
@click.option("--max-tokens", type=int, help="Maximum tokens in response")
@click.option("--temperature", type=click.FloatRange(0.0, 2.0), help="Sampling temperature")
@click.option("--verbose", is_flag=True, help="Verbose logging")
@click.option("--init", is_flag=True, help="Run setup wizard (`aither init`)")
@click.option("--link", "--auth", "link", is_flag=True, help="Sign in via browser to your Aitherium account")
@click.option("--local", "force_backend", flag_value="local", help="Force local backend (Ollama)")
@click.option("--cloud", "force_backend", flag_value="cloud", help="Force cloud backend (Portal)")
@click.option("--genesis", "force_backend", flag_value="genesis", help="Force Genesis backend")
@click.option("--config", is_flag=True, help="Show configuration")
@click.option("--plugins", is_flag=True, help="List plugins")
@click.option("--status", is_flag=True, help="Check Genesis health")
@click.option("--history", type=int, help="Show history (optionally with count)")
@click.option("--completions", type=click.Choice(["bash", "zsh", "fish", "pwsh"]), help="Generate shell completions")
@click.version_option(version="1.1.0", prog_name="aither")
@click.pass_context
def cli(
    ctx,
    query,
    output_format,
    private,
    effort,
    safety,
    will,
    model,
    max_tokens,
    temperature,
    verbose,
    init,
    link,
    force_backend,
    config,
    plugins,
    status,
    history,
    completions,
):
    """AitherShell - The AI Operating System CLI.
    
    Usage:
        aither                              # Interactive REPL
        aither "question"                   # Single query
        aither --print "query"              # Plain text output
        aither --json "query"               # JSON output
        aither --private "query"            # Private mode
        aither --will persona "query"       # Use specific persona
        aither --init                       # Initialize config
        aither --config                     # Show configuration
        aither --status                     # Check Genesis health
    """
    setup_logging(verbose)

    # Check license first (before any other action)
    is_valid, license_msg = validate_license()
    if not is_valid and not (init or link or completions):
        # No license — kick off browser auth flow with their AitherIdentity account
        from aithershell.portal_link import authenticate_or_exit
        authed = authenticate_or_exit()
        if not authed:
            sys.exit(1)
        # Re-validate after auth
        is_valid, license_msg = validate_license()
        if not is_valid:
            click.secho(f"\n❌ {license_msg}", fg="red", err=True)
            click.secho(
                "  Authentication completed but no valid license was issued.",
                fg="yellow", err=True,
            )
            click.secho(
                "  Contact support@aitherium.com if this persists.",
                fg="yellow", err=True,
            )
            sys.exit(1)
        click.secho(f"\n  Continuing your command...\n", fg="cyan")

    if is_valid and verbose:
        click.secho(f"✅ {license_msg}", fg="green")
    
    # Load config
    save_default_config()
    aither_config = load_config()
    
    # Apply CLI overrides to config
    if effort:
        aither_config.effort = effort
    if safety:
        aither_config.safety_level = safety
    if will:
        aither_config.persona = will
    if model:
        aither_config.model = model
    if max_tokens:
        aither_config.max_tokens = max_tokens
    if temperature:
        aither_config.temperature = temperature
    if output_format == "json":
        aither_config.rich_output = False
        aither_config.stream = False
    if output_format == "text":
        aither_config.rich_output = False
    if private:
        aither_config.privacy_level = "private"
    
    # Handle special flags
    if completions:
        from aithershell.completions import print_completion_script
        print_completion_script(completions)
        return
    
    if init:
        # Run the merged engine setup wizard
        from aithershell.setup_cli import main as setup_main
        try:
            sys.exit(asyncio.run(setup_main()) or 0)
        except SystemExit:
            raise
        except Exception as e:
            click.secho(f"setup failed: {e}", fg="red", err=True)
            sys.exit(1)
        return

    if link:
        # Portal API-key device-flow link
        from aithershell.portal_link import link_portal
        ok = link_portal(aither_config)
        sys.exit(0 if ok else 1)
        return

    if config:
        asyncio.run(_cmd_config(aither_config))
        return
    
    if plugins:
        asyncio.run(_cmd_plugins(aither_config))
        return
    
    if status:
        asyncio.run(_cmd_status(aither_config))
        return
    
    if history is not None:
        asyncio.run(_cmd_history(aither_config, history))
        return
    
    # Main logic
    if query:
        # Single query mode
        query_text = " ".join(query)
        asyncio.run(_cmd_query(aither_config, query_text, output_format, force_backend))
    else:
        # Interactive REPL
        try:
            asyncio.run(run_repl(aither_config))
        except KeyboardInterrupt:
            print("\nGoodbye!")


async def _cmd_query(
    config: AitherConfig,
    query: str,
    output_format: Optional[str],
    force_backend: Optional[str] = None,
) -> None:
    """Execute a single query with effort-based routing.

    Routing decision (when no --local/--cloud/--genesis flag):
      * effort <= config.routing.effort_threshold → local Ollama
      * else → cloud (Portal) or Genesis depending on which is configured
    Falls back to cloud on local errors if config.routing.fallback_on_error.
    """
    effort = config.effort if config.effort is not None else 5
    backend_name, backend = config.select_backend(effort, force=force_backend)

    show_routing = output_format != "json"
    if show_routing and config.show_metadata:
        click.secho(f"[{backend_name}:{backend.model or 'auto'}] ", fg="bright_black", err=True, nl=False)

    try:
        if backend.type == "ollama":
            await _query_ollama(config, backend, query, output_format, effort)
        else:
            # Portal (cloud) or Genesis: both use the GenesisClient HTTP path
            await _query_genesis(config, backend, query, output_format)
    except OllamaError as e:
        # Local failed → try fallback
        if config.routing.get("fallback_on_error", True) and not force_backend:
            click.secho(
                f"\n  ⚠ Local backend failed ({e}). Falling back to cloud...",
                fg="yellow", err=True,
            )
            cloud = config.get_backend("cloud")
            if cloud.url and cloud.api_key:
                try:
                    await _query_genesis(config, cloud, query, output_format)
                    return
                except GenesisError as ge:
                    _print_error(ge.message, output_format)
                    sys.exit(1)
        _print_error(str(e), output_format)
        sys.exit(1)
    except GenesisError as e:
        _print_error(e.message, output_format)
        sys.exit(1)


async def _query_ollama(
    config: AitherConfig,
    backend,
    query: str,
    output_format: Optional[str],
    effort: int,
) -> None:
    """Stream from local Ollama via the merged engine OllamaProvider."""
    model = config.model or backend.model or "nemotron-orchestrator:8b"
    timeout = float(config.routing.get("timeout_seconds", 120))
    provider = OllamaProvider(host=backend.url, default_model=model, timeout=timeout)

    messages = [Message(role="user", content=query)]
    response = ""
    try:
        async for chunk in provider.chat_stream(
            messages=messages,
            model=model,
            temperature=config.temperature if config.temperature is not None else 0.7,
            max_tokens=config.max_tokens or 4096,
        ):
            if chunk.content:
                response += chunk.content
                if config.stream and output_format != "json":
                    print(chunk.content, end="", flush=True)
            if chunk.done:
                break

        if config.stream and output_format != "json":
            print()

        if output_format == "json":
            print(json.dumps({
                "status": "success",
                "response": response,
                "backend": "local",
                "model": model,
                "effort": effort,
            }, indent=2))
        elif not config.stream:
            print(response)
    except Exception as e:
        # Translate engine errors into the unified error path
        raise OllamaError(str(e)) from e


async def _query_genesis(
    config: AitherConfig,
    backend,
    query: str,
    output_format: Optional[str],
) -> None:
    """Stream from cloud Portal or Genesis (both speak the same HTTP API)."""
    base_url = backend.url or config.url
    client = GenesisClient(base_url=base_url)
    response = ""
    try:
        async for chunk in client.chat_stream(
            message=query,
            persona=config.persona,
            effort=config.effort,
            model=config.model or backend.model,
            max_tokens=config.max_tokens,
            safety_level=config.safety_level,
            private_mode=getattr(config, "privacy_level", "public") == "private",
        ):
            response += chunk
            if config.stream and output_format != "json":
                print(chunk, end="", flush=True)

        if config.stream and output_format != "json":
            print()

        if output_format == "json":
            print(json.dumps({
                "status": "success",
                "response": response,
                "backend": backend.type,
                "model": config.model or backend.model,
                "effort": config.effort,
            }, indent=2))
        elif not config.stream:
            print(response)
    finally:
        await client.close()


def _print_error(msg: str, output_format: Optional[str]) -> None:
    if output_format == "json":
        print(json.dumps({"status": "error", "error": msg}, indent=2))
    else:
        print(f"[ERROR] {msg}", file=sys.stderr)


async def _cmd_config(config: AitherConfig) -> None:
    """Show configuration."""
    await execute_command(config, "config", ["show"])


async def _cmd_plugins(config: AitherConfig) -> None:
    """List plugins."""
    await execute_command(config, "plugins", ["list"])


async def _cmd_status(config: AitherConfig) -> None:
    """Check Genesis status."""
    await execute_command(config, "status")


async def _cmd_history(config: AitherConfig, count: int) -> None:
    """Show command history."""
    await execute_command(config, "history", [str(count)] if count else [])


def _init_shell():
    """Initialize AitherShell config and show setup instructions."""
    from aithershell.config import CONFIG_DIR, CONFIG_FILE, PLUGINS_DIR

    save_default_config()

    print(f"""
AitherShell initialized!

Config:   {CONFIG_FILE}
Plugins:  {PLUGINS_DIR}

Shell completions (pick your shell):
  bash:  eval "$(aither --completions bash)"
  zsh:   eval "$(aither --completions zsh)"
  fish:  aither --completions fish | source
  pwsh:  aither --completions pwsh >> $PROFILE

Quick start:
  aither                          # Interactive shell
  aither "hello"                  # Single query
  aither --print "question"       # Script mode
  aither --json "question"        # JSON output
  aither --private "query"        # Private mode
  echo "prompt" | aither --print  # Pipe input

Config: edit {CONFIG_FILE}
Plugins: drop .yaml or .py files in {PLUGINS_DIR}
""")


def entry():
    """Main entry point for `aither` console_scripts."""
    try:
        cli()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    except CommandError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


def _find_cli_module() -> str:
    """Find aither_cli.py in the repo."""
    # Check env var
    root = os.environ.get("AITHEROS_ROOT")
    if root:
        candidate = os.path.join(root, "AitherOS", "aither_cli.py")
        if os.path.exists(candidate):
            return candidate

    # Check common locations relative to this file
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # Installed in repo: aithershell/ is sibling of AitherOS/
        os.path.join(here, "..", "..", "AitherOS", "aither_cli.py"),
        # Dev mode
        os.path.join(here, "..", "..", "..", "AitherOS", "aither_cli.py"),
        # CWD
        os.path.join(os.getcwd(), "AitherOS", "aither_cli.py"),
        os.path.join(os.getcwd(), "aither_cli.py"),
    ]

    for c in candidates:
        c = os.path.normpath(c)
        if os.path.exists(c):
            return c

    return ""


def _init_shell():
    """Initialize AitherShell config and show setup instructions."""
    from aithershell.config import save_default_config, CONFIG_DIR, CONFIG_FILE, PLUGINS_DIR

    save_default_config()

    print(f"""
AitherShell initialized!

Config:   {CONFIG_FILE}
Plugins:  {PLUGINS_DIR}

Shell completions (pick your shell):
  bash:  eval "$(register-python-argcomplete aither)"
  zsh:   eval "$(register-python-argcomplete aither)"
  fish:  register-python-argcomplete --shell fish aither | source
  pwsh:  aither --completions powershell >> $PROFILE

Quick start:
  aither                          # Interactive shell
  aither "hello"                  # Single query
  aither --print "question"       # Script mode
  aither --json "question"        # JSON output
  aither --private                # Private mode
  echo "prompt" | aither --print  # Pipe input

Config: edit {CONFIG_FILE}
Plugins: drop .yaml or .py files in {PLUGINS_DIR}
""")


if __name__ == "__main__":
    entry()
