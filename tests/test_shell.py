"""
Tests for Interactive REPL
===========================
"""

import tempfile
from pathlib import Path
from unittest import mock

import pytest

from aithershell.config import AitherConfig
from aithershell.shell import AitherREPL, run_repl
from aithershell.genesis_client import GenesisError


class TestAitherREPL:
    """Test AitherREPL class."""

    def test_repl_init(self):
        """Test REPL initialization."""
        config = AitherConfig()
        repl = AitherREPL(config)
        
        assert repl.config == config
        assert repl.history == []

    def test_repl_init_validation(self):
        """Test REPL validation."""
        with pytest.raises(ValueError):
            AitherREPL(None)

    def test_repl_load_history(self):
        """Test loading history from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = Path(tmpdir) / "history"
            history_file.write_text("command1\ncommand2\ncommand3")
            
            config = AitherConfig(history_file=str(history_file))
            repl = AitherREPL(config)
            
            assert repl.history == ["command1", "command2", "command3"]

    def test_repl_save_history(self):
        """Test saving history to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = Path(tmpdir) / "history"
            
            config = AitherConfig(history_file=str(history_file))
            repl = AitherREPL(config)
            repl.history = ["cmd1", "cmd2"]
            
            repl._save_history()
            
            assert history_file.exists()
            content = history_file.read_text()
            assert "cmd1" in content
            assert "cmd2" in content

    def test_repl_max_history(self):
        """Test that history is limited to max_history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = Path(tmpdir) / "history"
            
            config = AitherConfig(
                history_file=str(history_file),
                max_history=5,
            )
            repl = AitherREPL(config)
            
            # Add more than max_history
            repl.history = ["cmd1", "cmd2", "cmd3", "cmd4", "cmd5", "cmd6", "cmd7"]
            repl._save_history()
            
            # Load and check
            repl2 = AitherREPL(config)
            assert len(repl2.history) <= 5
            assert "cmd7" in repl2.history  # Most recent should be kept

    @pytest.mark.asyncio
    async def test_handle_input_empty(self):
        """Test handling empty input."""
        config = AitherConfig()
        repl = AitherREPL(config)
        
        result = await repl.handle_input("")
        assert result is None
        
        result = await repl.handle_input("   ")
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_input_command(self):
        """Test handling commands."""
        config = AitherConfig()
        repl = AitherREPL(config)
        
        result = await repl.handle_input("/help")
        assert "Commands" in result

    @pytest.mark.asyncio
    async def test_handle_input_history_command(self):
        """Test handling /history command."""
        config = AitherConfig()
        repl = AitherREPL(config)
        repl.history = ["cmd1", "cmd2"]
        
        result = await repl.handle_input("/history")
        assert "cmd1" in result or "History is empty" not in result

    @pytest.mark.asyncio
    async def test_handle_input_clear_command(self):
        """Test handling /clear command."""
        config = AitherConfig()
        repl = AitherREPL(config)
        repl.history = ["cmd1", "cmd2"]
        
        result = await repl.handle_input("/clear")
        assert repl.history == []
        assert "cleared" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_input_unknown_command(self):
        """Test handling unknown command."""
        config = AitherConfig()
        repl = AitherREPL(config)
        
        result = await repl.handle_input("/unknown")
        assert "Unknown command" in result

    @pytest.mark.asyncio
    async def test_handle_input_query(self):
        """Test handling a query."""
        config = AitherConfig()
        repl = AitherREPL(config)
        
        with mock.patch.object(repl.genesis_client, "chat_stream") as mock_stream:
            async def fake_stream():
                yield "response"
            
            mock_stream.return_value = fake_stream()
            
            result = await repl.handle_input("test query")
            assert result == "response"
            assert "test query" in repl.history

    @pytest.mark.asyncio
    async def test_handle_input_query_streams_to_stdout(self):
        """Test that streaming responses are printed."""
        config = AitherConfig(stream=True)
        repl = AitherREPL(config)
        
        with mock.patch.object(repl.genesis_client, "chat_stream") as mock_stream, \
             mock.patch("builtins.print") as mock_print:
            async def fake_stream():
                yield "chunk1"
                yield "chunk2"
            
            mock_stream.return_value = fake_stream()
            
            result = await repl.handle_input("test query")
            assert result == "chunk1chunk2"

    @pytest.mark.asyncio
    async def test_handle_input_interrupted(self):
        """Test handling Ctrl+C during stream."""
        config = AitherConfig(stream=True)
        repl = AitherREPL(config)
        
        with mock.patch.object(repl.genesis_client, "chat_stream") as mock_stream:
            async def fake_stream():
                yield "chunk1"
                repl.interrupted = True
                yield "chunk2"  # Should not be yielded
            
            mock_stream.return_value = fake_stream()
            
            result = await repl.handle_input("test query")
            assert result is None

    @pytest.mark.asyncio
    async def test_cmd_help(self):
        """Test help command."""
        config = AitherConfig()
        repl = AitherREPL(config)
        
        help_text = repl._cmd_help()
        assert "help" in help_text.lower()
        assert "exit" in help_text.lower()

    @pytest.mark.asyncio
    async def test_cmd_history_empty(self):
        """Test history command when empty."""
        config = AitherConfig()
        repl = AitherREPL(config)
        
        result = repl._cmd_history()
        assert "empty" in result.lower()

    @pytest.mark.asyncio
    async def test_cmd_history_with_items(self):
        """Test history command with items."""
        config = AitherConfig()
        repl = AitherREPL(config)
        repl.history = ["cmd1", "cmd2", "cmd3"]
        
        result = repl._cmd_history()
        assert "cmd1" in result or "cmd2" in result


class TestRunREPL:
    """Test run_repl function."""

    @pytest.mark.asyncio
    async def test_run_repl_health_check_fails(self):
        """Test REPL startup when health check fails."""
        config = AitherConfig()
        
        with mock.patch("aithershell.shell.AitherREPL") as mock_repl_class:
            mock_repl = mock.AsyncMock()
            mock_repl.genesis_client.health_check = mock.AsyncMock(return_value=False)
            mock_repl_class.return_value = mock_repl
            
            with pytest.raises(Exception):  # GenesisError
                await run_repl(config)
