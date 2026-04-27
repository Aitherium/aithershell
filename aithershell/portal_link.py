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
IDP_BASE_DEFAULT = "https://idp.aitherium.com"
# AitherIdentity device-code endpoints (talk to IDP directly, not Portal)
DEVICE_CODE_PATH = "/auth/device/code"
DEVICE_POLL_PATH = "/auth/device/token"
LINK_PAGE_PATH = "/link"


async def _request_device_code(
    idp_url: str,
    portal_url: str,
) -> Optional[Tuple[str, str, int, str]]:
    """Ask IDP for a device code.

    Returns (device_code, user_code, interval, verification_uri_complete)
    or None if endpoint unavailable.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{idp_url}{DEVICE_CODE_PATH}",
                json={"client_name": "AitherShell", "scopes": ["read", "write", "agent"]},
            )
            if r.status_code == 200:
                data = r.json()
                # Prefer IDP-supplied URL; fall back to portal /link.
                vuri = data.get("verification_uri_complete") or (
                    f"{portal_url}{LINK_PAGE_PATH}?code={data['user_code']}"
                )
                return (
                    data["device_code"],
                    data["user_code"],
                    data.get("interval", 5),
                    vuri,
                )
    except Exception:
        pass
    return None


async def _poll_for_token(
    idp_url: str, device_code: str, interval: int, timeout_s: int = 600
) -> Optional[dict]:
    """Poll IDP until user approves and we get credentials.

    Returns dict with keys: api_key/access_token, license_key, user, tier
    or None on denial/timeout.
    """
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.time() < deadline:
            try:
                r = await client.post(
                    f"{idp_url}{DEVICE_POLL_PATH}",
                    json={"device_code": device_code},
                )
                if r.status_code == 200:
                    data = r.json()
                    if data.get("status") == "complete":
                        return data
                    if data.get("status") in ("denied", "access_denied"):
                        return None
                    # status == "authorization_pending" → keep polling
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


async def link_portal_async(
    cfg: AitherConfig,
    portal_url: str = PORTAL_BASE_DEFAULT,
    idp_url: str = IDP_BASE_DEFAULT,
) -> bool:
    """Run the device-code flow and persist the API key.

    CLI talks DIRECTLY to AitherIdentity (idp.aitherium.com). Portal hosts
    the /link page where the user signs in and authorizes the device.
    """
    click.secho(f"\n→ Linking AitherShell via {idp_url}", fg="cyan")

    code_info = await _request_device_code(idp_url, portal_url)

    if code_info:
        device_code, user_code, interval, link_url = code_info

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
            creds = await _poll_for_token(idp_url, device_code, interval)
        except KeyboardInterrupt:
            click.secho("\n  Cancelled.", fg="yellow")
            return False

        if not creds:
            click.secho("  Linking failed or denied.", fg="red")
            return False

        # IDP returns access_token + api_key (alias) + license_key + tier + user
        api_key = creds.get("api_key") or creds.get("access_token", "")
        license_key = creds.get("license_key", "")
        user_obj = creds.get("user", {}) or {}
        email = user_obj.get("email") or creds.get("email", "")
        tier = creds.get("tier", "free")

        if api_key:
            cfg.api_key = api_key
            if "cloud" in cfg.backends:
                cfg.backends["cloud"]["api_key"] = api_key
                cfg.backends["cloud"]["url"] = portal_url
            save_config(cfg)

        if license_key:
            _save_license(license_key)

        click.secho(f"\n  ✓ Signed in as {email or 'user'} ({tier} tier)", fg="green", bold=True)
        if api_key:
            click.secho("  ✓ API key saved to ~/.aither/config.yaml", fg="green")
        if license_key:
            click.secho("  ✓ License key saved to ~/.aither/license.key", fg="green")
        return True

    # Fallback: manual key entry (IDP unreachable / device flow not deployed)
    click.secho(
        "\n  IDP device flow not available. Falling back to manual entry.",
        fg="yellow",
    )
    fallback_url = f"{portal_url}/login?return=/settings/api-keys"
    click.secho(f"  1. Opening browser: {fallback_url}", fg="white")
    click.secho("  2. Sign in (or create an account)", fg="white")
    click.secho("  3. Generate an API key in Settings → API Keys", fg="white")
    click.secho("  4. Paste it below (or press Enter to skip)\n", fg="white")

    _open_browser(fallback_url)

    api_key = click.prompt("  API key", default="", show_default=False, hide_input=True)
    api_key = api_key.strip()
    if not api_key:
        click.secho("  Skipped. You can run `aither auth` later.", fg="yellow")
        return False

    cfg.api_key = api_key
    if "cloud" in cfg.backends:
        cfg.backends["cloud"]["api_key"] = api_key
        cfg.backends["cloud"]["url"] = portal_url
    save_config(cfg)
    click.secho("  ✓ API key saved to ~/.aither/config.yaml", fg="green")

    # Optional: paste a license key too
    license_key = click.prompt(
        "  License key (optional, get from portal account page)",
        default="",
        show_default=False,
        hide_input=True,
    ).strip()
    if license_key:
        _save_license(license_key)
        click.secho("  ✓ License key saved to ~/.aither/license.key", fg="green")
    return True


def _save_license(license_key: str) -> None:
    """Persist license key to ~/.aither/license.key (mode 0600 on POSIX)."""
    from pathlib import Path
    import os

    license_path = Path.home() / ".aither" / "license.key"
    license_path.parent.mkdir(parents=True, exist_ok=True)
    license_path.write_text(license_key.strip() + "\n", encoding="utf-8")
    try:
        os.chmod(license_path, 0o600)
    except (OSError, NotImplementedError):
        pass  # Windows or restricted FS


def link_portal(cfg: AitherConfig, portal_url: str = PORTAL_BASE_DEFAULT) -> bool:
    """Sync wrapper for CLI use.

    Honors AITHER_PORTAL_URL (browser /link host) and AITHER_IDP_URL
    (CLI device-code endpoints) env vars for dev/staging overrides.
    """
    import os
    portal_url = os.environ.get("AITHER_PORTAL_URL", portal_url).rstrip("/")
    idp_url = os.environ.get("AITHER_IDP_URL", IDP_BASE_DEFAULT).rstrip("/")
    return asyncio.run(link_portal_async(cfg, portal_url, idp_url))


def authenticate_or_exit(portal_url: str = PORTAL_BASE_DEFAULT) -> bool:
    """First-run auth flow triggered when no license is found.

    Asks user to authenticate via browser (device-code flow with their
    AitherIdentity account). On success, license + API key are saved and
    the user can re-run their command.

    Returns True if authentication succeeded.
    """
    from aithershell.config import load_config

    click.secho("\n  Welcome to AitherShell!", fg="cyan", bold=True)
    click.secho(
        "  No license found. Sign in with your Aitherium account to continue.\n",
        fg="white",
    )
    click.secho(
        "    • Free tier: 5 queries/day, all local features",
        fg="white",
    )
    click.secho(
        "    • Pro tier:  Unlimited, cloud fallback, priority support\n",
        fg="white",
    )

    if not click.confirm("  Open browser to sign in?", default=True):
        click.secho(
            f"\n  No problem. Sign in any time at {portal_url}",
            fg="yellow",
        )
        click.secho("  Then run:  aither auth", fg="yellow")
        return False

    cfg = load_config()
    return link_portal(cfg, portal_url)
