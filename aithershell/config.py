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
from __future__ import annotations

import os
import logging
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


# ============================================================
# Engine Config (from former adk.config — environment-driven)
# ============================================================

import json as _json_engine
from typing import Any as _Any_engine

logger = logging.getLogger("aithershell.engine_config")


# ---------------------------------------------------------------------------
# ~/.aither/config.json helpers
# ---------------------------------------------------------------------------

_CONFIG_PATH_JSON = Path.home() / ".aither" / "config.json"
_CONFIG_PATH_YAML = Path.home() / ".aither" / "config.yaml"
# Prefer YAML (shared with AitherShell), fall back to JSON (legacy)
_CONFIG_PATH = _CONFIG_PATH_YAML if _CONFIG_PATH_YAML.exists() else _CONFIG_PATH_JSON


def load_saved_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load persisted config from ``~/.aither/config.yaml`` or ``config.json``.

    Tries YAML first (shared with AitherShell), falls back to JSON (legacy).
    Returns an empty dict when the file does not exist or cannot be parsed.
    """
    # Try YAML first
    yaml_path = config_path or _CONFIG_PATH_YAML
    if yaml_path.exists() and yaml_path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            logger.debug("Failed to read YAML config from %s", yaml_path)

    # Fall back to JSON
    json_path = config_path or _CONFIG_PATH_JSON
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Failed to read JSON config from %s", json_path)
    return {}


def save_saved_config(data: dict[str, Any], config_path: Path | None = None) -> Path:
    """Merge *data* into the persisted ADK config and write it back.

    Creates ``~/.aither/`` if it does not exist.  Returns the path that was
    written for caller convenience.
    """
    path = config_path or _CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_saved_config(path)
    existing.update(data)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return path


@dataclass
class Config:
    """ADK configuration, populated from environment variables with sensible defaults.

    If AITHER_PROFILE is set (or auto-detected via AgentSetup), the hardware profile
    YAML is loaded and its model/limits settings are applied as defaults — env vars
    always override profile values.
    """

    # LLM backend: "ollama", "openai", "anthropic", "auto"
    llm_backend: str = field(default_factory=lambda: os.getenv("AITHER_LLM_BACKEND", "auto"))

    # Model selection (env vars override profile)
    model: str = field(default_factory=lambda: os.getenv("AITHER_MODEL", ""))
    small_model: str = field(default_factory=lambda: os.getenv("AITHER_SMALL_MODEL", ""))
    large_model: str = field(default_factory=lambda: os.getenv("AITHER_LARGE_MODEL", ""))

    # Ollama
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))

    # OpenAI-compatible
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Anthropic
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # General API key (for gateway or fallback)
    aither_api_key: str = field(default_factory=lambda: os.getenv("AITHER_API_KEY", ""))

    # Server
    server_port: int = field(default_factory=lambda: int(os.getenv("AITHER_PORT", "8080")))
    server_host: str = field(default_factory=lambda: os.getenv("AITHER_HOST", "0.0.0.0"))

    # Phonehome (opt-in)
    phonehome_enabled: bool = field(
        default_factory=lambda: os.getenv("AITHER_PHONEHOME", "").lower() in ("true", "1", "yes")
    )
    gateway_url: str = field(
        default_factory=lambda: os.getenv("AITHER_GATEWAY_URL", "https://gateway.aitherium.com")
    )

    # Prefer local inference over gateway even when AITHER_API_KEY is set
    prefer_local: bool = field(
        default_factory=lambda: os.getenv("AITHER_PREFER_LOCAL", "").lower() in ("true", "1", "yes")
    )

    # Register agent with gateway on startup (opt-in)
    register_agent: bool = field(
        default_factory=lambda: os.getenv("AITHER_REGISTER_AGENT", "").lower() in ("true", "1", "yes")
    )

    # Tenant context (set by ``aither connect``, stored in ~/.aither/config.json)
    tenant_id: str = field(
        default_factory=lambda: os.getenv("AITHER_TENANT_ID", "")
    )

    # Data directory
    data_dir: str = field(
        default_factory=lambda: os.getenv("AITHER_DATA_DIR", os.path.expanduser("~/.aither"))
    )

    # Observability — AitherOS service URLs (auto-detected from localhost)
    chronicle_url: str = field(
        default_factory=lambda: os.getenv("AITHER_CHRONICLE_URL", "")
    )
    watch_url: str = field(
        default_factory=lambda: os.getenv("AITHER_WATCH_URL", "")
    )
    pulse_url: str = field(
        default_factory=lambda: os.getenv("AITHER_PULSE_URL", "")
    )

    # LLMFit sidecar — hardware-aware model scoring
    # If empty, the llmfit client auto-resolves via convention (port 8793)
    llmfit_url: str = field(
        default_factory=lambda: os.getenv("AITHER_LLMFIT_URL", "")
    )

    # JSON structured logging (default on)
    json_logging: bool = field(
        default_factory=lambda: os.getenv("AITHER_JSON_LOGGING", "true").lower() not in ("false", "0", "no")
    )

    # Hardware profile (auto-detected or set via AITHER_PROFILE)
    profile: str = field(default_factory=lambda: os.getenv("AITHER_PROFILE", ""))

    # Profile-derived settings (populated by from_profile/apply_profile)
    max_context: int = 0          # 0 = unlimited (let model decide)
    max_concurrent: int = 0       # 0 = unlimited
    profile_models: dict = field(default_factory=dict)  # {default, small, large, embedding, ...}

    @classmethod
    def from_env(cls) -> Config:
        """Create config from current environment variables.

        If AITHER_PROFILE is set, loads and applies the hardware profile.
        If not set, checks ~/.aither/detected_profile from a previous auto_setup().

        Also loads ``tenant_id`` and ``api_key`` from ``~/.aither/config.json``
        when those values are not already set via environment variables.
        """
        config = cls()
        if config.profile:
            config.apply_profile(config.profile)
        else:
            # Try auto-detected profile from previous setup run
            marker = Path(config.data_dir) / "detected_profile"
            if marker.exists():
                try:
                    detected = marker.read_text(encoding="utf-8").strip()
                    if detected:
                        config.apply_profile(detected)
                except Exception:
                    pass

        # Backfill from saved config.json (env vars always win)
        saved = load_saved_config()
        if not config.tenant_id and saved.get("tenant_id"):
            config.tenant_id = saved["tenant_id"]
        if not config.aither_api_key and saved.get("api_key"):
            config.aither_api_key = saved["api_key"]

        return config

    @classmethod
    def from_profile(cls, profile_name: str) -> Config:
        """Create config from a hardware profile name."""
        config = cls(profile=profile_name)
        config.apply_profile(profile_name)
        return config

    def apply_profile(self, profile_name: str) -> None:
        """Load a hardware profile YAML and apply its settings.

        Profile settings are defaults — env vars always win.
        Looks for profiles in: ./profiles/, package profiles/, ~/.aither/profiles/
        """
        try:
            import yaml
        except ImportError:
            logger.debug("PyYAML not installed, skipping profile load")
            return

        # Search paths for profile YAML
        search_dirs = [
            Path("profiles"),                                    # CWD
            Path(__file__).parent.parent / "profiles",           # package root
            Path(self.data_dir) / "profiles",                    # ~/.aither/profiles/
        ]

        profile_path = None
        for d in search_dirs:
            candidate = d / f"{profile_name}.yaml"
            if candidate.exists():
                profile_path = candidate
                break

        if not profile_path:
            logger.debug("Profile '%s' not found in %s", profile_name, [str(d) for d in search_dirs])
            return

        try:
            data = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.warning("Failed to load profile %s: %s", profile_name, e)
            return

        self.profile = profile_name

        # Apply models (env vars override)
        models = data.get("models", {})
        self.profile_models = models
        if not self.model:
            # Use 'default' or 'chat' key from profile
            self.model = models.get("default", models.get("chat", ""))
        if not self.small_model:
            self.small_model = models.get("small", "")
        if not self.large_model:
            self.large_model = models.get("large", models.get("reasoning", ""))

        # Apply limits
        limits = data.get("limits", {})
        if not self.max_context:
            self.max_context = limits.get("max_context", 0)
        if not self.max_concurrent:
            self.max_concurrent = limits.get("max_concurrent", 0)

        logger.info("Applied profile '%s': model=%s, small=%s, large=%s, max_context=%d",
                     profile_name, self.model, self.small_model, self.large_model, self.max_context)

    def get_api_key(self) -> str:
        """Return the best available API key for the configured backend."""
        if self.llm_backend == "anthropic":
            return self.anthropic_api_key or self.aither_api_key
        if self.llm_backend == "openai":
            return self.openai_api_key or self.aither_api_key
        return self.aither_api_key or self.openai_api_key or self.anthropic_api_key

    def get_llmfit_client(self):
        """Create a LLMFitClient initialized with this config's llmfit_url.

        Returns None if the llmfit module isn't installed.
        """
        try:
            from aithershell.llmfit import get_llmfit
            return get_llmfit(base_url=self.llmfit_url or None)
        except ImportError:
            logger.debug("adk.llmfit module not available")
            return None
