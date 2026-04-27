"""
Tests for AitherShell Configuration
====================================
"""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from aithershell.config import (
    AitherConfig,
    load_config,
    save_default_config,
    _apply_dict,
)


class TestAitherConfig:
    """Test AitherConfig dataclass."""

    def test_default_values(self):
        """Test that defaults are set correctly."""
        cfg = AitherConfig()
        assert cfg.url == "https://localhost:8001"
        assert cfg.api_key == ""
        assert cfg.stream is True
        assert cfg.rich_output is True
        assert cfg.max_history == 10000

    def test_to_dict(self):
        """Test conversion to dict."""
        cfg = AitherConfig(url="http://test:8001", model="test-model")
        d = cfg.to_dict()
        
        assert d["url"] == "http://test:8001"
        assert d["model"] == "test-model"
        assert isinstance(d, dict)

    def test_apply_dict(self):
        """Test _apply_dict function."""
        cfg = AitherConfig()
        data = {
            "url": "http://new:9000",
            "model": "gpt-4",
            "unknown_key": "should_be_ignored",
        }
        _apply_dict(cfg, data)
        
        assert cfg.url == "http://new:9000"
        assert cfg.model == "gpt-4"


class TestLoadConfig:
    """Test config loading with layering."""

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_load_config_defaults(self):
        """Test loading config with only defaults."""
        with mock.patch("aithershell.config.CONFIG_FILE") as mock_file, \
             mock.patch("aithershell.config.PROJECT_CONFIG") as mock_proj:
            mock_file.exists.return_value = False
            mock_proj.exists.return_value = False
            
            cfg = load_config()
            assert cfg.url == "https://localhost:8001"
            assert cfg.stream is True

    def test_load_config_from_file(self):
        """Test loading config from ~/.aither/config.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("""
url: http://custom:8001
model: custom-model
stream: false
""")
            
            with mock.patch("aithershell.config.CONFIG_FILE", config_file), \
                 mock.patch("aithershell.config.PROJECT_CONFIG") as mock_proj, \
                 mock.patch.dict("os.environ", {}, clear=True):
                mock_proj.exists.return_value = False
                
                cfg = load_config()
                assert cfg.url == "http://custom:8001"
                assert cfg.model == "custom-model"
                assert cfg.stream is False

    def test_load_config_env_override(self):
        """Test that environment variables override files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("url: http://file:8001")
            
            with mock.patch("aithershell.config.CONFIG_FILE", config_file), \
                 mock.patch("aithershell.config.PROJECT_CONFIG") as mock_proj, \
                 mock.patch.dict("os.environ", {"AITHER_URL": "http://env:8001"}):
                mock_proj.exists.return_value = False
                
                cfg = load_config()
                # Env should override file
                assert cfg.url == "http://env:8001"

    def test_load_config_type_coercion(self):
        """Test that env vars are properly type-coerced."""
        with mock.patch("aithershell.config.CONFIG_FILE") as mock_file, \
             mock.patch("aithershell.config.PROJECT_CONFIG") as mock_proj, \
             mock.patch.dict("os.environ", {"AITHER_EFFORT": "5"}):
            mock_file.exists.return_value = False
            mock_proj.exists.return_value = False
            
            cfg = load_config()
            assert cfg.effort == 5
            assert isinstance(cfg.effort, int)


class TestSaveDefaultConfig:
    """Test default config creation."""

    def test_save_default_config(self):
        """Test that default config is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            
            with mock.patch("aithershell.config.CONFIG_FILE", config_file):
                save_default_config()
                
                assert config_file.exists()
                content = config_file.read_text()
                assert "url:" in content
                assert "stream:" in content

    def test_save_default_config_idempotent(self):
        """Test that saving twice is safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            
            with mock.patch("aithershell.config.CONFIG_FILE", config_file):
                save_default_config()
                first_content = config_file.read_text()
                
                save_default_config()
                second_content = config_file.read_text()
                
                assert first_content == second_content
