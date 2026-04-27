"""
Tests for Genesis HTTP Client
==============================
"""

import asyncio
from unittest import mock

import httpx
import pytest

from aithershell.genesis_client import (
    GenesisClient,
    GenesisError,
    GenesisConnectionError,
    GenesisTimeoutError,
)


class TestGenesisClient:
    """Test GenesisClient class."""

    def test_init_default(self):
        """Test initialization with defaults."""
        client = GenesisClient()
        assert client.base_url == "http://localhost:8001"
        assert client.timeout == 30.0
        assert client.max_retries == 3

    def test_init_custom(self):
        """Test initialization with custom values."""
        client = GenesisClient(
            base_url="http://custom:9000",
            timeout=60.0,
            max_retries=5,
        )
        assert client.base_url == "http://custom:9000"
        assert client.timeout == 60.0
        assert client.max_retries == 5

    def test_init_validation(self):
        """Test that invalid values raise errors."""
        with pytest.raises(ValueError):
            GenesisClient(timeout=-1)
        
        with pytest.raises(ValueError):
            GenesisClient(max_retries=-1)

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing client."""
        client = GenesisClient()
        await client._get_client()
        assert client._client is not None
        
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with GenesisClient() as client:
            assert isinstance(client, GenesisClient)
        
        assert client._client is None

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        client = GenesisClient()
        
        with mock.patch.object(client, "_get_client") as mock_get:
            mock_response = mock.AsyncMock()
            mock_response.status_code = 200
            
            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_client
            
            result = await client.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""
        client = GenesisClient()
        
        with mock.patch.object(client, "_get_client") as mock_get:
            mock_response = mock.AsyncMock()
            mock_response.status_code = 500
            
            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_client
            
            result = await client.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self):
        """Test health check with connection error."""
        client = GenesisClient()
        
        with mock.patch.object(client, "_get_client") as mock_get:
            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(
                side_effect=httpx.ConnectError("Connection failed")
            )
            mock_get.return_value = mock_client
            
            result = await client.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_chat_stream_success(self):
        """Test successful chat stream."""
        client = GenesisClient()
        
        with mock.patch.object(client, "_stream_request") as mock_stream:
            async def fake_stream():
                yield "Hello "
                yield "world"
            
            mock_stream.return_value = fake_stream()
            
            chunks = []
            async for chunk in client.chat_stream("test query"):
                chunks.append(chunk)
            
            assert chunks == ["Hello ", "world"]

    @pytest.mark.asyncio
    async def test_chat_stream_with_options(self):
        """Test chat stream with all options."""
        client = GenesisClient()
        
        with mock.patch.object(client, "_stream_request") as mock_stream:
            async def fake_stream():
                yield "response"
            
            mock_stream.return_value = fake_stream()
            
            chunks = []
            async for chunk in client.chat_stream(
                message="test",
                persona="aither",
                effort=5,
                model="claude-opus",
                max_tokens=100,
                temperature=0.7,
                safety_level="professional",
                private_mode=True,
                session_id="session123",
            ):
                chunks.append(chunk)
            
            # Verify _stream_request was called with correct payload
            call_args = mock_stream.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "/chat"
            payload = call_args[1]["json"]
            assert payload["message"] == "test"
            assert payload["persona"] == "aither"
            assert payload["effort"] == 5

    @pytest.mark.asyncio
    async def test_chat_full_response(self):
        """Test chat() method that waits for full response."""
        client = GenesisClient()
        
        with mock.patch.object(client, "chat_stream") as mock_stream:
            async def fake_stream():
                yield "Hello "
                yield "world"
            
            mock_stream.return_value = fake_stream()
            
            response = await client.chat("test")
            assert response == "Hello world"

    @pytest.mark.asyncio
    async def test_stream_request_retry_success(self):
        """Test retry logic with eventual success."""
        client = GenesisClient(max_retries=2)
        
        with mock.patch.object(client, "_get_client") as mock_get:
            # First call fails, second succeeds
            responses = [
                mock.AsyncMock(side_effect=httpx.ConnectError("Failed")),
                mock.AsyncMock(),
            ]
            
            mock_client = mock.AsyncMock()
            mock_client.stream = mock.AsyncMock()
            mock_get.return_value = mock_client
            
            # Set up mock_client.stream to raise on first call, succeed on second
            call_count = 0
            async def mock_stream(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise httpx.ConnectError("Connection failed")
                
                # Return successful response
                mock_response = mock.AsyncMock()
                mock_response.status_code = 200
                mock_response.aiter_text = mock.AsyncMock(
                    return_value=mock.AsyncMock(
                        __aiter__=mock.AsyncMock(return_value=["test"])
                    )
                )
                return mock.AsyncMock(__aenter__=mock.AsyncMock(return_value=mock_response))
            
            mock_client.stream = mock_stream


class TestGenesisErrors:
    """Test error classes."""

    def test_genesis_error(self):
        """Test GenesisError."""
        error = GenesisError("test error", status_code=500, attempt=2)
        assert error.message == "test error"
        assert error.status_code == 500
        assert error.attempt == 2
        assert error.timestamp

    def test_connection_error(self):
        """Test GenesisConnectionError."""
        error = GenesisConnectionError("Connection failed", attempt=1)
        assert isinstance(error, GenesisError)
        assert error.message == "Connection failed"

    def test_timeout_error(self):
        """Test GenesisTimeoutError."""
        error = GenesisTimeoutError("Request timed out", attempt=3)
        assert isinstance(error, GenesisError)
        assert error.message == "Request timed out"
