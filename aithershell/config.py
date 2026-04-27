"""
AitherShell Configuration
=========================

Config resolution order (later overrides earlier):
1. Built-in defaults
2. ~/.aither/config.yaml
3. .aither.yaml (project-local)
4. Environment variables (AITHER_*)
5. CLI flags
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

CONFIG_DIR = Path.home() / ".aither"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
PROJECT_CONFIG = Path(".aither.yaml")
SESSION_FILE = CONFIG_DIR / "session.json"
PLUGINS_DIR = CONFIG_DIR / "plugins"
PRIVATE_DIR = CONFIG_DIR / "private"


@dataclass
class BackendConfig:
    """Single backend (local or cloud) config."""
    type: str = "ollama"  # ollama | portal | genesis | openai
    url: str = ""
    model: str = ""
    api_key: str = ""
    max_effort: int = 10  # only route to this backend for effort <= max_effort


@dataclass
class RoutingConfig:
    """Effort-based routing config."""
    mode: str = "hybrid"          # hybrid | local_only | cloud_only | genesis_only
    effort_threshold: int = 6      # effort > threshold → cloud
    timeout_seconds: int = 120     # local timeout → fallback to cloud
    fallback_on_error: bool = True # local error → try cloud


@dataclass
class AitherConfig:
    """Shell configuration with layered resolution."""

    # Connection
    url: str = "https://localhost:8001"
    will_url: str = "https://localhost:8097"
    gateway_url: str = "https://gateway.aitherium.com"

    # Auth
    api_key: str = ""
    tenant_id: str = ""
    identity_url: str = "https://localhost:8112"

    # Auth state (populated from ~/.aither/auth.json at load time)
    auth_token: Optional[str] = None
    auth_user: Optional[Dict[str, Any]] = None

    # Defaults
    model: Optional[str] = None
    persona: Optional[str] = None
    effort: Optional[int] = None
    safety_level: str = "professional"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    privacy_level: str = "public"

    # Behavior
    stream: bool = True
    rich_output: bool = True
    show_thinking: bool = False
    show_metadata: bool = False

    # Session
    session_id: Optional[str] = None
    auto_continue: bool = False

    # Shell
    prompt: str = "aither> "
    history_file: str = str(CONFIG_DIR / "history")
    max_history: int = 10000

    # Plugins
    plugin_dirs: list = field(default_factory=lambda: [str(PLUGINS_DIR)])

    # ========== Local Onramp (v1.1) ==========
    # Backend definitions
    backends: Dict[str, Any] = field(default_factory=lambda: {
        "local": {
            "type": "ollama",
            "url": "http://localhost:11434",
            "model": "nemotron-orchestrator:8b",
            "max_effort": 6,
        },
        "cloud": {
            "type": "portal",
            "url": "https://api.aitherium.com",
            "api_key": "",
            "model": "auto",
            "max_effort": 10,
        },
        "genesis": {
            "type": "genesis",
            "url": "https://localhost:8001",
            "model": "auto",
            "max_effort": 10,
        },
    })

    # Routing
    routing: Dict[str, Any] = field(default_factory=lambda: {
        "mode": "hybrid",
        "effort_threshold": 6,
        "timeout_seconds": 120,
        "fallback_on_error": True,
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_backend(self, name: str) -> BackendConfig:
        """Get a backend config by name."""
        data = self.backends.get(name, {})
        return BackendConfig(
            type=data.get("type", "ollama"),
            url=data.get("url", ""),
            model=data.get("model", ""),
            api_key=data.get("api_key", ""),
            max_effort=data.get("max_effort", 10),
        )

    def select_backend(self, effort: int, force: Optional[str] = None) -> tuple[str, BackendConfig]:
        """Select backend based on effort level and routing mode.

        Args:
            effort: 1-10 effort level for the request
            force: Force a specific backend name ('local', 'cloud', 'genesis'), bypassing routing

        Returns:
            (backend_name, BackendConfig)
        """
        if force:
            return force, self.get_backend(force)

        mode = self.routing.get("mode", "hybrid")
        threshold = self.routing.get("effort_threshold", 6)

        if mode == "local_only":
            return "local", self.get_backend("local")
        if mode == "cloud_only":
            return "cloud", self.get_backend("cloud")
        if mode == "genesis_only":
            return "genesis", self.get_backend("genesis")

        # hybrid: effort decides
        if effort <= threshold:
            local = self.get_backend("local")
            if local.url and local.model:
                return "local", local
            # local not configured → fallback to cloud
        return "cloud", self.get_backend("cloud")


def _ensure_dirs():
    """Create config directories if needed."""
    for d in [CONFIG_DIR, PLUGINS_DIR, PRIVATE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def save_config(cfg: AitherConfig) -> None:
    """Persist config back to ~/.aither/config.yaml.

    Preserves any unknown keys in the existing file.
    """
    _ensure_dirs()
    # Read existing
    existing: Dict[str, Any] = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                existing = yaml.safe_load(f) or {}
        except Exception:
            existing = {}

    # Merge: cfg fields take precedence
    data = cfg.to_dict()
    existing.update(data)

    # Strip transient fields we don't want persisted
    for k in ("auth_token", "auth_user", "session_id"):
        existing.pop(k, None)

    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(existing, f, default_flow_style=False, sort_keys=False)


def load_config() -> AitherConfig:
    """Load config with layered resolution."""
    _ensure_dirs()
    cfg = AitherConfig()

    # Layer 1: ~/.aither/config.yaml
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                data = yaml.safe_load(f) or {}
            _apply_dict(cfg, data)
        except Exception:
            pass

    # Layer 2: .aither.yaml (project-local)
    if PROJECT_CONFIG.exists():
        try:
            with open(PROJECT_CONFIG, "r") as f:
                data = yaml.safe_load(f) or {}
            _apply_dict(cfg, data)
        except Exception:
            pass

    # Layer 3: Environment variables
    env_map = {
        "AITHER_URL": "url",
        "AITHER_ORCHESTRATOR_URL": "url",
        "AITHER_WILL_URL": "will_url",
        "AITHER_GATEWAY_URL": "gateway_url",
        "AITHER_API_KEY": "api_key",
        "AITHER_TENANT_ID": "tenant_id",
        "AITHER_IDENTITY_URL": "identity_url",
        "AITHER_MODEL": "model",
        "AITHER_PERSONA": "persona",
        "AITHER_EFFORT": "effort",
        "AITHER_SAFETY": "safety_level",
        "AITHER_MAX_TOKENS": "max_tokens",
        "AITHER_PROMPT": "prompt",
    }
    for env_key, attr in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            # Type coercion
            current = getattr(cfg, attr)
            if isinstance(current, int) or attr in ("effort", "max_tokens"):
                try:
                    val = int(val)
                except ValueError:
                    continue
            elif isinstance(current, bool):
                val = val.lower() in ("1", "true", "yes")
            setattr(cfg, attr, val)

    # Layer 4: Load auth token from ~/.aither/auth.json
    # If no auth exists, auto-provision root (like Linux console login).
    try:
        from aithershell.auth import AuthStore, ensure_root_profile
        token = AuthStore.get_active_token()
        if not token:
            # No valid session — provision root for local use
            ensure_root_profile()
            token = AuthStore.get_active_token()
        if token:
            cfg.auth_token = token
            if not cfg.api_key:
                cfg.api_key = token
            user = AuthStore.get_active_user()
            if user:
                cfg.auth_user = user
                if not cfg.tenant_id and user.get("tenant_id"):
                    cfg.tenant_id = user["tenant_id"]
    except Exception:
        pass  # auth module not available or broken store

    return cfg


def save_default_config():
    """Write a default config.yaml if none exists."""
    _ensure_dirs()
    if CONFIG_FILE.exists():
        return
    default = """# AitherShell Configuration
# ========================
# Edit this file to set defaults. CLI flags override these.
# Env vars (AITHER_URL, AITHER_MODEL, etc.) also override.

# Genesis connection
url: https://localhost:8001
will_url: https://localhost:8097

# Default model (null = auto-select by effort)
# model: aither-orchestrator

# Default effort level (1-10, null = auto)
# effort: 5

# Safety level: professional | casual | unrestricted
safety_level: professional

# Default persona
# persona: aither

# Streaming (disable for scripts)
stream: true

# Rich terminal output (disable for plain text)
rich_output: true

# Show model thinking/reasoning
show_thinking: false

# Show metadata (effort, neurons, elapsed)
show_metadata: false

# Shell prompt
prompt: "aither> "

# Plugin directories (list)
plugin_dirs:
  - ~/.aither/plugins
"""
    with open(CONFIG_FILE, "w") as f:
        f.write(default)


def _apply_dict(cfg: AitherConfig, data: dict):
    """Apply a dict to config, ignoring unknown keys."""
    for key, val in data.items():
        if hasattr(cfg, key) and val is not None:
            setattr(cfg, key, val)
