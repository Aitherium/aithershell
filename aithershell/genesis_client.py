"""
AitherShell Genesis HTTP Client
=================================

High-performance async HTTP client for Genesis (port 8001).
Features streaming responses, exponential backoff retry logic,
and comprehensive error handling.
"""

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Dict, Any, Optional
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)


class GenesisClient:
    """
    Async HTTP client for Genesis orchestrator.
    
    Handles:
    - Streaming responses (SSE or chunked JSON)
    - Exponential backoff retries
    - Configurable timeouts
    - Comprehensive error handling
    - Request/response logging
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        enable_logging: bool = True,
    ):
        """Initialize Genesis client.
        
        Args:
            base_url: Genesis base URL (default: http://localhost:8001)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum retry attempts (default: 3)
            backoff_factor: Exponential backoff multiplier (default: 2.0)
            enable_logging: Enable request/response logging (default: True)
            
        Raises:
            ValueError: If timeout or max_retries are invalid
        """
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.enable_logging = enable_logging
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async client.
        
        Returns:
            httpx.AsyncClient instance
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the async client.
        
        Raises:
            Exception: If client close fails
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def health_check(self) -> bool:
        """Check Genesis health.
        
        Returns:
            True if Genesis is responsive, False otherwise
            
        Examples:
            >>> async with GenesisClient() as client:
            ...     healthy = await client.health_check()
            ...     if healthy:
            ...         print("Genesis is ready")
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.base_url}/health",
                timeout=self.timeout / 2,  # Shorter timeout for health check
            )
            return response.status_code == 200
        except (httpx.RequestError, asyncio.TimeoutError):
            if self.enable_logging:
                logger.debug(f"Health check failed for {self.base_url}")
            return False

    async def chat_stream(
        self,
        message: str,
        persona: Optional[str] = None,
        effort: Optional[int] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        safety_level: Optional[str] = None,
        private_mode: bool = False,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream a chat request to Genesis.
        
        Args:
            message: User message/query
            persona: Persona name (e.g., "aither", "aither-prime")
            effort: Effort level 1-10 (None = auto-select)
            model: Model override (e.g., "claude-opus")
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-2.0)
            safety_level: Safety level (paranoid, strict, professional, relaxed)
            private_mode: Enable privacy mode (redact query from logs)
            session_id: Session ID for context continuity
            
        Yields:
            String chunks from streaming response
            
        Raises:
            GenesisConnectionError: If connection fails
            GenesisTimeoutError: If request times out
            GenesisError: For other errors
            
        Examples:
            >>> async with GenesisClient() as client:
            ...     async for chunk in client.chat_stream("Hello Genesis"):
            ...         print(chunk, end="", flush=True)
        """
        payload = {
            "message": message,
        }
        
        # Add optional fields
        if persona:
            payload["persona"] = persona
        if effort is not None:
            payload["effort"] = effort
        if model:
            payload["model"] = model
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if safety_level:
            payload["safety_level"] = safety_level
        if private_mode:
            payload["private_mode"] = True
        if session_id:
            payload["session_id"] = session_id
        
        if self.enable_logging:
            logger.debug(f"Chat stream request: {json.dumps(payload, default=str)}")
        
        async for chunk in self._stream_request(
            "POST",
            "/chat",
            json=payload,
        ):
            yield chunk

    async def _stream_request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Internal method to stream a request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional httpx arguments
            
        Yields:
            String chunks from response
            
        Raises:
            GenesisConnectionError: If all retries fail
            GenesisTimeoutError: If timeout occurs
            GenesisError: For other errors
        """
        url = f"{self.base_url}{endpoint}"
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                client = await self._get_client()
                
                if self.enable_logging:
                    logger.debug(f"{method} {endpoint} (attempt {attempt + 1}/{self.max_retries + 1})")
                
                async with client.stream(method, url, **kwargs) as response:
                    if response.status_code >= 400:
                        error_text = await response.aread()
                        error_msg = error_text.decode("utf-8", errors="replace")
                        raise GenesisError(
                            f"Genesis returned {response.status_code}: {error_msg}",
                            status_code=response.status_code,
                        )
                    
                    # Stream chunks
                    async for chunk in response.aiter_text():
                        if chunk:
                            yield chunk
                
                return  # Success
                
            except asyncio.TimeoutError as e:
                last_error = GenesisTimeoutError(
                    f"Request timed out after {self.timeout}s",
                    attempt=attempt,
                )
                if self.enable_logging:
                    logger.warning(f"Timeout on attempt {attempt + 1}")
                    
            except httpx.ConnectError as e:
                last_error = GenesisConnectionError(
                    f"Failed to connect to Genesis: {e}",
                    attempt=attempt,
                )
                if self.enable_logging:
                    logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                    
            except httpx.RequestError as e:
                last_error = GenesisConnectionError(
                    f"Request error: {e}",
                    attempt=attempt,
                )
                if self.enable_logging:
                    logger.warning(f"Request error on attempt {attempt + 1}: {e}")
                    
            except GenesisError:
                raise  # Don't retry application errors
                
            # Exponential backoff before retry
            if attempt < self.max_retries:
                wait_time = (self.backoff_factor ** attempt)
                if self.enable_logging:
                    logger.debug(f"Waiting {wait_time:.1f}s before retry")
                await asyncio.sleep(wait_time)
        
        # All retries exhausted
        if last_error:
            raise last_error
        else:
            raise GenesisConnectionError(
                f"Failed to connect to Genesis after {self.max_retries + 1} attempts",
                attempt=self.max_retries,
            )

    async def chat(
        self,
        message: str,
        **kwargs,
    ) -> str:
        """Send a chat message and wait for full response.
        
        Args:
            message: User message/query
            **kwargs: Additional arguments passed to chat_stream
            
        Returns:
            Full response text
            
        Raises:
            GenesisConnectionError: If connection fails
            GenesisTimeoutError: If request times out
            GenesisError: For other errors
            
        Examples:
            >>> async with GenesisClient() as client:
            ...     response = await client.chat("What is the meaning of life?")
            ...     print(response)
        """
        response_text = ""
        async for chunk in self.chat_stream(message, **kwargs):
            response_text += chunk
        return response_text


# Error classes
class GenesisError(Exception):
    """Base Genesis error."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, attempt: int = 0):
        """Initialize Genesis error.
        
        Args:
            message: Error message
            status_code: HTTP status code (if applicable)
            attempt: Attempt number (for retries)
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.attempt = attempt
        self.timestamp = datetime.utcnow().isoformat()


class GenesisConnectionError(GenesisError):
    """Connection or network error."""
    pass


class GenesisTimeoutError(GenesisError):
    """Request timeout error."""
    pass
