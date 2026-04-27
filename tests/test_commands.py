"""
Tests for Built-in Commands
============================
"""

import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from aithershell.config import AitherConfig
from aithershell.commands import Commands, execute_command, CommandError


class TestCommands:
    """Test Commands class."""

    def test_commands_init(self):
        """Test Commands initialization."""
        config = AitherConfig()
        commands = Commands(config)
        
        assert commands.config == config
        assert commands.genesis_client is not None
        assert commands.plugin_registry is not None

    @pytest.mark.asyncio
    async def test_help_command(self):
        """Test help command."""
        config = AitherConfig()
        commands = Commands(config)
        
        result = await commands.help()
        assert "help" in result.lower()
        assert "plugins" in result.lower()

    @pytest.mark.asyncio
    async def test_plugins_list_empty(self):
        """Test plugins command with no plugins."""
        config = AitherConfig()
        commands = Commands(config)
        
        with mock.patch.object(commands.plugin_registry, "list_plugins", return_value=[]):
            result = await commands.plugins("list")
            assert "No plugins" in result

    @pytest.mark.asyncio
    async def test_plugins_list_with_plugins(self):
        """Test plugins command with plugins."""
        config = AitherConfig()
        commands = Commands(config)
        
        from aithershell.plugins import Plugin
        
        plugin = Plugin(
            name="test",
            aliases=["t"],
            description="Test plugin",
            action="test",
        )
        
        with mock.patch.object(commands.plugin_registry, "list_plugins", return_value=[plugin]):
            result = await commands.plugins("list")
            assert "test" in result
            assert "Test plugin" in result

    @pytest.mark.asyncio
    async def test_plugins_load(self):
        """Test plugins load command."""
        config = AitherConfig()
        commands = Commands(config)
        
        with mock.patch.object(commands.plugin_registry, "load_plugins", return_value=5):
            result = await commands.plugins("load")
            assert "5" in result
            assert "plugins" in result.lower()

    @pytest.mark.asyncio
    async def test_plugins_unknown_subcommand(self):
        """Test plugins with unknown subcommand."""
        config = AitherConfig()
        commands = Commands(config)
        
        with pytest.raises(CommandError):
            await commands.plugins("unknown")

    @pytest.mark.asyncio
    async def test_config_show(self):
        """Test config show command."""
        config = AitherConfig(url="http://test:8001", model="test-model")
        commands = Commands(config)
        
        result = await commands.config("show")
        assert "http://test:8001" in result
        assert "test-model" in result

    @pytest.mark.asyncio
    async def test_config_show_redacts_secrets(self):
        """Test that config show redacts sensitive values."""
        config = AitherConfig(api_key="secret-key")
        commands = Commands(config)
        
        result = await commands.config("show")
        assert "REDACTED" in result
        assert "secret-key" not in result

    @pytest.mark.asyncio
    async def test_config_file(self):
        """Test config file command."""
        config = AitherConfig()
        commands = Commands(config)
        
        result = await commands.config("file")
        assert ".aither" in result

    @pytest.mark.asyncio
    async def test_config_set(self):
        """Test config set command."""
        config = AitherConfig()
        commands = Commands(config)
        
        with mock.patch.object(commands, "_save_config"):
            result = await commands.config("set", "model", "test-model")
            assert "Set" in result
            assert config.model == "test-model"

    @pytest.mark.asyncio
    async def test_config_set_missing_args(self):
        """Test config set with missing arguments."""
        config = AitherConfig()
        commands = Commands(config)
        
        with pytest.raises(CommandError):
            await commands.config("set", "model")

    @pytest.mark.asyncio
    async def test_status_command(self):
        """Test status command."""
        config = AitherConfig()
        commands = Commands(config)
        
        with mock.patch.object(commands.genesis_client, "health_check") as mock_health:
            mock_health.return_value = True
            
            result = await commands.status()
            assert "HEALTHY" in result or "✓" in result

    @pytest.mark.asyncio
    async def test_status_command_failure(self):
        """Test status command when Genesis is down."""
        config = AitherConfig()
        commands = Commands(config)
        
        with mock.patch.object(commands.genesis_client, "health_check") as mock_health:
            mock_health.return_value = False
            
            result = await commands.status()
            assert "UNREACHABLE" in result or "✗" in result

    @pytest.mark.asyncio
    async def test_history_command_empty(self):
        """Test history command when empty."""
        config = AitherConfig()
        commands = Commands(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = Path(tmpdir) / "history"
            config.history_file = str(history_file)
            
            result = await commands.history()
            assert "empty" in result.lower()

    @pytest.mark.asyncio
    async def test_history_command_with_entries(self):
        """Test history command with entries."""
        config = AitherConfig()
        commands = Commands(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = Path(tmpdir) / "history"
            history_file.write_text("cmd1\ncmd2\ncmd3\ncmd4\ncmd5")
            config.history_file = str(history_file)
            
            result = await commands.history()
            assert "cmd" in result

    @pytest.mark.asyncio
    async def test_history_command_with_count(self):
        """Test history command with count parameter."""
        config = AitherConfig()
        commands = Commands(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = Path(tmpdir) / "history"
            history_file.write_text("cmd1\ncmd2\ncmd3")
            config.history_file = str(history_file)
            
            result = await commands.history(1)
            # Should have only 1 line (excluding header)
            lines = result.split("\n")
            # Filter empty lines
            lines = [l for l in lines if l.strip()]
            assert len(lines) <= 1

    @pytest.mark.asyncio
    async def test_history_invalid_count(self):
        """Test history command with invalid count."""
        config = AitherConfig()
        commands = Commands(config)
        
        with pytest.raises(CommandError):
            await commands.history("not_a_number")

    @pytest.mark.asyncio
    async def test_exit_command(self):
        """Test exit command."""
        config = AitherConfig()
        commands = Commands(config)
        
        with pytest.raises(KeyboardInterrupt):
            await commands.exit()


class TestExecuteCommand:
    """Test execute_command function."""

    @pytest.mark.asyncio
    async def test_execute_command_success(self):
        """Test executing a command."""
        config = AitherConfig()
        
        result = await execute_command(config, "help")
        assert "Commands" in result

    @pytest.mark.asyncio
    async def test_execute_command_not_found(self):
        """Test executing non-existent command."""
        config = AitherConfig()
        
        with pytest.raises(CommandError):
            await execute_command(config, "nonexistent")

    @pytest.mark.asyncio
    async def test_execute_command_with_args(self):
        """Test executing command with arguments."""
        config = AitherConfig()
        
        with mock.patch("aithershell.commands.Commands.config") as mock_config:
            mock_config.return_value = "test"
            
            result = await execute_command(config, "config", ["show"])
            assert result == "test"
