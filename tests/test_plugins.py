"""
Tests for Plugin System
=======================
"""

import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from aithershell.plugins import (
    Plugin,
    PluginRegistry,
    PluginError,
    PluginLoadError,
    PluginExecutionError,
    PluginCapabilityError,
)


class TestPlugin:
    """Test Plugin class."""

    def test_plugin_creation(self):
        """Test creating a plugin."""
        plugin = Plugin(
            name="test",
            aliases=["t"],
            description="Test plugin",
            action="http://localhost:9000/test",
        )
        assert plugin.name == "test"
        assert plugin.aliases == ["t"]
        assert plugin.enabled is True

    def test_plugin_validation(self):
        """Test plugin validation."""
        with pytest.raises(ValueError):
            Plugin(name="", aliases=[], description="", action="test")
        
        with pytest.raises(ValueError):
            Plugin(name="test", aliases=[], description="", action="")

    def test_plugin_to_dict(self):
        """Test plugin to_dict()."""
        plugin = Plugin(
            name="test",
            aliases=["t"],
            description="Test",
            action="test",
            requires_capabilities=["read"],
        )
        d = plugin.to_dict()
        assert d["name"] == "test"
        assert d["aliases"] == ["t"]
        assert d["requires_capabilities"] == ["read"]


class TestPluginRegistry:
    """Test PluginRegistry class."""

    def test_registry_init(self):
        """Test registry initialization."""
        registry = PluginRegistry(plugin_dirs=["/tmp/plugins"])
        assert len(registry.plugin_dirs) == 1
        assert registry.user_capabilities == []

    def test_set_user_capabilities(self):
        """Test setting user capabilities."""
        registry = PluginRegistry()
        registry.set_user_capabilities(["read", "write", "admin"])
        assert registry.user_capabilities == ["read", "write", "admin"]

    def test_load_yaml_plugin(self):
        """Test loading YAML plugin."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            yaml_file = plugin_dir / "test.yaml"
            
            yaml_file.write_text("""
name: test-plugin
aliases: ["tp"]
description: Test YAML plugin
action: http://localhost:9000/test
requires_capabilities: ["read"]
enabled: true
""")
            
            registry = PluginRegistry(plugin_dirs=[str(plugin_dir)])
            plugin = registry.get_plugin("test-plugin")
            
            assert plugin is not None
            assert plugin.name == "test-plugin"
            assert plugin.aliases == ["tp"]
            assert "read" in plugin.requires_capabilities

    def test_load_python_plugin(self):
        """Test loading Python plugin."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            py_file = plugin_dir / "test_plugin.py"
            
            py_file.write_text("""
PLUGIN = {
    "name": "py-plugin",
    "aliases": ["pyp"],
    "description": "Test Python plugin",
    "action": "http://localhost:9000/test",
    "requires_capabilities": ["write"],
}
""")
            
            registry = PluginRegistry(plugin_dirs=[str(plugin_dir)])
            plugin = registry.get_plugin("py-plugin")
            
            assert plugin is not None
            assert plugin.name == "py-plugin"

    def test_get_plugin_by_alias(self):
        """Test getting plugin by alias."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            yaml_file = plugin_dir / "test.yaml"
            
            yaml_file.write_text("""
name: test
aliases: ["t", "test-alias"]
description: Test
action: http://localhost/test
""")
            
            registry = PluginRegistry(plugin_dirs=[str(plugin_dir)])
            
            # Get by name
            plugin = registry.get_plugin("test")
            assert plugin.name == "test"
            
            # Get by alias
            plugin = registry.get_plugin("t")
            assert plugin.name == "test"
            
            plugin = registry.get_plugin("test-alias")
            assert plugin.name == "test"

    def test_list_plugins_no_duplicates(self):
        """Test that list_plugins() returns no duplicates from aliases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            yaml_file = plugin_dir / "test.yaml"
            
            yaml_file.write_text("""
name: test
aliases: ["t1", "t2"]
description: Test
action: test
""")
            
            registry = PluginRegistry(plugin_dirs=[str(plugin_dir)])
            plugins = registry.list_plugins()
            
            # Should have only 1 plugin despite 3 aliases
            assert len(plugins) == 1
            assert plugins[0].name == "test"

    @pytest.mark.asyncio
    async def test_execute_plugin_not_found(self):
        """Test executing non-existent plugin."""
        registry = PluginRegistry()
        
        with pytest.raises(PluginExecutionError, match="Plugin not found"):
            await registry.execute_plugin("nonexistent")

    @pytest.mark.asyncio
    async def test_execute_plugin_disabled(self):
        """Test executing disabled plugin."""
        plugin = Plugin(
            name="disabled",
            aliases=[],
            description="Disabled",
            action="test",
            enabled=False,
        )
        
        registry = PluginRegistry()
        registry.plugins["disabled"] = plugin
        
        with pytest.raises(PluginExecutionError, match="disabled"):
            await registry.execute_plugin("disabled")

    @pytest.mark.asyncio
    async def test_execute_plugin_capability_check(self):
        """Test RBAC capability checking."""
        plugin = Plugin(
            name="admin-only",
            aliases=[],
            description="Admin only",
            action="test",
            requires_capabilities=["admin"],
        )
        
        registry = PluginRegistry()
        registry.plugins["admin-only"] = plugin
        # User has no capabilities
        registry.set_user_capabilities([])
        
        with pytest.raises(PluginCapabilityError):
            await registry.execute_plugin("admin-only")

    @pytest.mark.asyncio
    async def test_execute_plugin_api(self):
        """Test executing API plugin."""
        plugin = Plugin(
            name="api-plugin",
            aliases=[],
            description="API plugin",
            action="http://localhost:9000/test",
        )
        
        registry = PluginRegistry()
        registry.plugins["api-plugin"] = plugin
        
        with mock.patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock.AsyncMock()
            mock_response = mock.AsyncMock()
            mock_response.json.return_value = {"result": "ok"}
            mock_client.get = mock.AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await registry.execute_plugin("api-plugin")
            assert "ok" in result


class TestPluginErrors:
    """Test error classes."""

    def test_plugin_error(self):
        """Test PluginError."""
        error = PluginError("test error")
        assert str(error) == "test error"

    def test_plugin_load_error(self):
        """Test PluginLoadError."""
        error = PluginLoadError("Failed to load")
        assert isinstance(error, PluginError)

    def test_plugin_execution_error(self):
        """Test PluginExecutionError."""
        error = PluginExecutionError("Execution failed")
        assert isinstance(error, PluginError)

    def test_plugin_capability_error(self):
        """Test PluginCapabilityError."""
        error = PluginCapabilityError("Unauthorized")
        assert isinstance(error, PluginError)
