"""
AitherShell Built-in Commands
==============================

Commands that don't require CLI framework:
- help: Show help
- plugins: List/manage plugins
- config: Show/set configuration
- status: Check Genesis health
- history: Show command history
- exit: Exit shell

These are called by cli.py and shell.py.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from aithershell.config import AitherConfig, CONFIG_FILE, PLUGINS_DIR
from aithershell.genesis_client import GenesisClient
from aithershell.plugins import PluginRegistry

logger = logging.getLogger(__name__)


class CommandError(Exception):
    """Command execution error."""
    pass


class Commands:
    """Built-in commands for AitherShell."""

    def __init__(self, config: AitherConfig):
        """Initialize commands.
        
        Args:
            config: AitherConfig instance
        """
        self.config = config
        self.genesis_client = GenesisClient(base_url=config.url)
        self.plugin_registry = PluginRegistry(config.plugin_dirs)

    async def help(self, *args) -> str:
        """Show help for built-in commands.
        
        Returns:
            Help text
        """
        return """
AitherShell Built-in Commands:

help                Show this help
plugins [list]      List available plugins
plugins load        Reload plugins from disk
config show         Show current configuration
config set KEY VAL  Set configuration value
config file         Show config file path
status              Check Genesis health
history [N]         Show last N history items (default: 20)
exit                Exit shell

Examples:
  aither help
  aither plugins list
  aither config show
  aither status
  aither history 10
"""

    async def plugins(self, *args) -> str:
        """List or manage plugins.
        
        Args:
            *args: Subcommand and arguments
            
        Returns:
            Plugin information
            
        Raises:
            CommandError: If command fails
        """
        subcommand = args[0] if args else "list"
        
        if subcommand == "list":
            plugins = self.plugin_registry.list_plugins()
            if not plugins:
                return "No plugins loaded."
            
            lines = ["Available plugins:\n"]
            for plugin in plugins:
                aliases = f" (aliases: {', '.join(plugin.aliases)})" if plugin.aliases else ""
                enabled = "" if plugin.enabled else " [DISABLED]"
                lines.append(f"  {plugin.name}{enabled}{aliases}")
                lines.append(f"    {plugin.description}")
            
            return "\n".join(lines)
        
        elif subcommand == "load":
            count = self.plugin_registry.load_plugins()
            return f"Loaded {count} plugins."
        
        else:
            raise CommandError(f"Unknown plugins subcommand: {subcommand}")

    async def config(self, *args) -> str:
        """Show or set configuration.
        
        Args:
            *args: Subcommand and arguments
            
        Returns:
            Configuration information
            
        Raises:
            CommandError: If command fails
        """
        if not args:
            return await self.config("show")
        
        subcommand = args[0]
        
        if subcommand == "show":
            cfg_dict = self.config.to_dict()
            # Redact sensitive values
            if cfg_dict.get("auth_token"):
                cfg_dict["auth_token"] = "[REDACTED]"
            if cfg_dict.get("api_key"):
                cfg_dict["api_key"] = "[REDACTED]"
            return json.dumps(cfg_dict, indent=2, default=str)
        
        elif subcommand == "file":
            return str(CONFIG_FILE)
        
        elif subcommand == "set":
            if len(args) < 3:
                raise CommandError("config set requires KEY and VALUE")
            
            key = args[1]
            val = args[2]
            
            # Type coercion
            if hasattr(self.config, key):
                current = getattr(self.config, key)
                if isinstance(current, bool):
                    val = val.lower() in ("1", "true", "yes")
                elif isinstance(current, int):
                    try:
                        val = int(val)
                    except ValueError:
                        raise CommandError(f"Invalid int value: {val}")
            
            setattr(self.config, key, val)
            
            # Save to ~/.aither/config.yaml
            self._save_config()
            
            return f"Set {key} = {val}"
        
        else:
            raise CommandError(f"Unknown config subcommand: {subcommand}")

    def _save_config(self) -> None:
        """Save current config to ~/.aither/config.yaml."""
        try:
            cfg_dict = self.config.to_dict()
            # Redact sensitive fields
            cfg_dict.pop("auth_token", None)
            cfg_dict.pop("auth_user", None)
            
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg_dict, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise CommandError(f"Failed to save config: {e}")

    async def status(self, *args) -> str:
        """Check Genesis health.
        
        Returns:
            Status information
        """
        try:
            healthy = await self.genesis_client.health_check()
            if healthy:
                return f"Genesis ({self.config.url}): HEALTHY ✓"
            else:
                return f"Genesis ({self.config.url}): UNREACHABLE ✗"
        except Exception as e:
            return f"Genesis ({self.config.url}): ERROR - {e}"
        finally:
            await self.genesis_client.close()

    async def history(self, *args) -> str:
        """Show command history.
        
        Args:
            *args: Optional count (default: 20)
            
        Returns:
            History text
        """
        try:
            count = int(args[0]) if args else 20
        except ValueError:
            raise CommandError(f"Invalid count: {args[0]}")
        
        history_file = Path(self.config.history_file)
        if not history_file.exists():
            return "History is empty."
        
        try:
            lines = history_file.read_text(encoding="utf-8").splitlines()
            recent = lines[-count:] if count else lines
            
            output = []
            start_idx = len(lines) - len(recent)
            for i, line in enumerate(recent, start=start_idx):
                output.append(f"{i+1:4d}  {line}")
            
            return "\n".join(output)
        except Exception as e:
            logger.error(f"Failed to read history: {e}")
            return f"Error reading history: {e}"

    async def exit(self, *args) -> str:
        """Exit the shell.
        
        Raises:
            KeyboardInterrupt: Always raises to signal exit
        """
        raise KeyboardInterrupt()


# Command dispatcher
async def execute_command(
    config: AitherConfig,
    command: str,
    args: Optional[List[str]] = None,
) -> str:
    """Execute a built-in command.
    
    Args:
        config: AitherConfig instance
        command: Command name
        args: Command arguments
        
    Returns:
        Command output
        
    Raises:
        CommandError: If command fails
    """
    commands = Commands(config)
    args = args or []
    
    method = getattr(commands, command, None)
    if not method:
        raise CommandError(f"Unknown command: {command}")
    
    if not callable(method):
        raise CommandError(f"{command} is not a command")
    
    try:
        return await method(*args)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(f"Command '{command}' failed: {e}")
        raise CommandError(f"Command failed: {e}")
