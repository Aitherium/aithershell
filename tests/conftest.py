"""
Pytest Configuration for AitherShell Tests
==========================================

Provides fixtures and configuration for all tests.
"""

import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Provide a temporary config directory."""
    config_dir = tmp_path / ".aither"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def tmp_history_file(tmp_config_dir):
    """Provide a temporary history file."""
    history_file = tmp_config_dir / "history"
    history_file.touch()
    return history_file


@pytest.fixture
def tmp_plugins_dir(tmp_config_dir):
    """Provide a temporary plugins directory."""
    plugins_dir = tmp_config_dir / "plugins"
    plugins_dir.mkdir()
    return plugins_dir


@pytest.fixture
def aither_config_factory(tmp_config_dir, tmp_history_file):
    """Factory for creating AitherConfig instances."""
    def create_config(**kwargs):
        from aithershell.config import AitherConfig
        
        config_kwargs = {
            "history_file": str(tmp_history_file),
        }
        config_kwargs.update(kwargs)
        return AitherConfig(**config_kwargs)
    
    return create_config


# Mark async tests
pytest_plugins = ("pytest_asyncio",)
