"""
Portal linking via OAuth-style device-code flow.

User runs `aither link` → CLI shows a code → user goes to portal in browser →
approves → CLI polls and receives an API key → key is saved to config.

This is the bridge from a free local install to the paid cloud experience.
The portal endpoint (/link) doesn't need to exist yet — until then, this
falls back to manual API-key entry.
"""
from __future__ import annotations

import asyncio
import secrets
import time
import webbrowser
from typing import Optional, Tuple

import click
import httpx

from aithershell.config import AitherConfig, save_config


PORTAL_BASE_DEFAULT = "https://portal.aitherium.com"
DEVICE_CODE_PATH = "/api/auth/device/code"
DEVICE_POLL_PATH = "/api/auth/device/poll"
LINK_PAGE_PATH = "/link"


async def _request_device_code(portal_url: str) -> Optional[Tuple[str, str, int]]:
    """Ask portal for a device code.

    Returns (device_code, user_code, interval) or None if endpoint unavailable.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{portal_url}{DEVICE_CODE_PATH}",
                json={"client": "aithershell", "version": "1.0"},
            )
            if r.status_code == 200:
                data = r.json()
                return (
                    data["device_code"],
                    data["user_code"],
                    data.get("interval", 3),
                )
    except Exception:
        pass
    return None


async def _poll_for_token(portal_url: str, device_code: str, interval: int, timeout_s: int = 600) -> Optional[str]:
    """Poll until user approves and we get an API key."""
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.time() < deadline:
            try:
                r = await client.post(
                    f"{portal_url}{DEVICE_POLL_PATH}",
                    json={"device_code": device_code},
                )
                if r.status_code == 200:
                    data = r.json()
                    if data.get("api_key"):
                        return data["api_key"]
                    if data.get("status") == "denied":
                        return None
            except Exception:
                pass
            await asyncio.sleep(interval)
    return None


def _open_browser(url: str) -> None:
    """Open browser, swallow errors (headless boxes etc)."""
    try:
        webbrowser.open(url, new=2)
    except Exception:
        pass


async def link_portal_async(cfg: AitherConfig, portal_url: str = PORTAL_BASE_DEFAULT) -> bool:
    """Run the device-code flow and persist the API key.

    Returns True if linked successfully, False otherwise.
    Falls back to manual key entry if portal device flow isn't deployed yet.
    """
    click.secho(f"\n→ Linking AitherShell to {portal_url}", fg="cyan")

    # Try device flow
    code_info = await _request_device_code(portal_url)

    if code_info:
        device_code, user_code, interval = code_info
        link_url = f"{portal_url}{LINK_PAGE_PATH}?code={user_code}"

        click.secho("\n  Visit:", fg="white")
        click.secho(f"    {link_url}", fg="bright_blue", underline=True)
        click.secho("  Or enter code manually at:", fg="white")
        click.secho(f"    {portal_url}{LINK_PAGE_PATH}", fg="bright_blue")
        click.secho(f"  Code: ", fg="white", nl=False)
        click.secho(user_code, fg="bright_yellow", bold=True)

        if click.confirm("\n  Open browser?", default=True):
            _open_browser(link_url)

        click.secho("\n  Waiting for approval (Ctrl+C to cancel)...", fg="white")
        try:
            api_key = await _poll_for_token(portal_url, device_code, interval)
        except KeyboardInterrupt:
            click.secho("\n  Cancelled.", fg="yellow")
            return False

        if not api_key:
            click.secho("  Linking failed or denied.", fg="red")
            return False

        # Persist
        cfg.api_key = api_key
        if "cloud" in cfg.backends:
            cfg.backends["cloud"]["api_key"] = api_key
            cfg.backends["cloud"]["url"] = portal_url
        save_config(cfg)
        click.secho("  ✓ Linked successfully. API key saved to ~/.aither/config.yaml", fg="green")
        return True

    # Fallback: manual key entry (portal device flow not deployed yet)
    click.secho(
        "\n  Portal device flow not available yet. Falling back to manual entry.",
        fg="yellow",
    )
    click.secho(f"  1. Sign in at {portal_url}", fg="white")
    click.secho("  2. Generate an API key in your account settings", fg="white")
    click.secho("  3. Paste it below (or press Enter to skip)\n", fg="white")

    api_key = click.prompt("  API key", default="", show_default=False, hide_input=True)
    api_key = api_key.strip()
    if not api_key:
        click.secho("  Skipped. You can run `aither link` later.", fg="yellow")
        return False

    cfg.api_key = api_key
    if "cloud" in cfg.backends:
        cfg.backends["cloud"]["api_key"] = api_key
        cfg.backends["cloud"]["url"] = portal_url
    save_config(cfg)
    click.secho("  ✓ API key saved to ~/.aither/config.yaml", fg="green")
    return True


def link_portal(cfg: AitherConfig, portal_url: str = PORTAL_BASE_DEFAULT) -> bool:
    """Sync wrapper for CLI use."""
    return asyncio.run(link_portal_async(cfg, portal_url))
