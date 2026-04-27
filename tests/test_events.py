"""
Tests for Events/Telemetry
===========================
"""

import time
import threading
from unittest import mock

import pytest

from aithershell.pulse_events import (
    EventSeverity,
    PrivacyLevel,
    QueryContext,
    PulseEventClient,
    get_pulse_client,
    create_query_id,
    query_context_from_request,
)


class TestEventSeverity:
    """Test EventSeverity enum."""

    def test_severity_values(self):
        """Test severity enum values."""
        assert EventSeverity.INFO.value == "info"
        assert EventSeverity.WARNING.value == "warning"
        assert EventSeverity.ERROR.value == "error"
        assert EventSeverity.CRITICAL.value == "critical"


class TestPrivacyLevel:
    """Test PrivacyLevel enum."""

    def test_privacy_values(self):
        """Test privacy level enum values."""
        assert PrivacyLevel.PUBLIC.value == "public"
        assert PrivacyLevel.PRIVATE.value == "private"
        assert PrivacyLevel.REDACTED.value == "redacted"


class TestQueryContext:
    """Test QueryContext dataclass."""

    def test_query_context_creation(self):
        """Test creating QueryContext."""
        ctx = QueryContext(
            query_id="q123",
            persona="aither",
            effort=5,
        )
        assert ctx.query_id == "q123"
        assert ctx.persona == "aither"
        assert ctx.effort == 5

    def test_query_context_defaults(self):
        """Test QueryContext defaults."""
        ctx = QueryContext(query_id="q123")
        assert ctx.privacy_level == PrivacyLevel.PUBLIC.value
        assert ctx.model is None


class TestPulseEventClient:
    """Test PulseEventClient class."""

    def test_client_init_default(self):
        """Test client initialization with defaults."""
        client = PulseEventClient()
        assert client.service_name == "aithershell"
        assert client.flush_interval_ms == 1000
        assert client.max_batch_size == 100
        assert client.enabled is True

    def test_client_init_custom(self):
        """Test client initialization with custom values."""
        client = PulseEventClient(
            service_name="test-service",
            pulse_url="http://custom:8081",
            flush_interval_ms=500,
            max_batch_size=50,
            enabled=False,
        )
        assert client.service_name == "test-service"
        assert client.pulse_url == "http://custom:8081"
        assert client.flush_interval_ms == 500
        assert client.max_batch_size == 50
        assert client.enabled is False

    def test_client_start_stop(self):
        """Test starting and stopping client."""
        client = PulseEventClient(enabled=False)  # Disable to avoid actual HTTP
        client.start()
        assert client.running is True
        
        client.stop()
        assert client.running is False

    def test_emit_event(self):
        """Test emitting an event."""
        client = PulseEventClient(enabled=True)
        client.emit("test.event", {"key": "value"})
        
        # Event should be queued
        assert len(client.event_queue) == 1
        event = client.event_queue[0]
        assert event["event"] == "test.event"
        assert event["data"]["key"] == "value"

    def test_emit_query_started(self):
        """Test emitting query started event."""
        client = PulseEventClient(enabled=True)
        client.emit_query_started(
            query_id="q123",
            persona="aither",
            effort=5,
            tokens_estimated=100,
            query_text="test query",
        )
        
        assert len(client.event_queue) == 1
        event = client.event_queue[0]
        assert event["event"] == "aithershell.query.started"

    def test_emit_query_completed(self):
        """Test emitting query completed event."""
        client = PulseEventClient(enabled=True)
        client.emit_query_completed(
            query_id="q123",
            duration_ms=1234.5,
            tokens_used=150,
            model="claude-opus",
            cached=False,
        )
        
        assert len(client.event_queue) == 1
        event = client.event_queue[0]
        assert event["event"] == "aithershell.query.completed"

    def test_emit_error(self):
        """Test emitting error event."""
        client = PulseEventClient(enabled=True)
        client.emit_error(
            error_type="api_timeout",
            severity=EventSeverity.ERROR.value,
            retry_count=2,
            message="Timeout after 30s",
        )
        
        assert len(client.event_queue) == 1
        event = client.event_queue[0]
        assert event["event"] == "aithershell.error"

    def test_emit_plugin_executed(self):
        """Test emitting plugin execution event."""
        client = PulseEventClient(enabled=True)
        client.emit_plugin_executed(
            plugin_name="my-plugin",
            duration_ms=234.5,
            success=True,
        )
        
        assert len(client.event_queue) == 1
        event = client.event_queue[0]
        assert event["event"] == "aithershell.plugin.executed"

    def test_privacy_filter_public(self):
        """Test privacy filter with public level."""
        client = PulseEventClient()
        data = {"query_text": "secret query"}
        
        filtered = client._apply_privacy_filter(data, PrivacyLevel.PUBLIC.value)
        assert filtered["query_text"] == "secret query"

    def test_privacy_filter_private(self):
        """Test privacy filter with private level."""
        client = PulseEventClient()
        data = {"query_text": "secret query", "query_id": "q123"}
        
        filtered = client._apply_privacy_filter(data, PrivacyLevel.PRIVATE.value)
        assert "query_text" not in filtered
        assert filtered["query_id"] == "q123"

    def test_privacy_filter_redacted(self):
        """Test privacy filter with redacted level."""
        client = PulseEventClient()
        data = {"query_text": "secret query"}
        
        filtered = client._apply_privacy_filter(data, PrivacyLevel.REDACTED.value)
        assert "[REDACTED:" in filtered["query_text"]

    def test_update_metrics(self):
        """Test updating metrics."""
        client = PulseEventClient()
        
        client.update_metrics(
            queries_completed=10,
            errors_total=2,
        )
        
        assert client.metrics["queries_completed"] == 10
        assert client.metrics["errors_total"] == 2


class TestGlobalClient:
    """Test global client instance."""

    def test_get_pulse_client_singleton(self):
        """Test that get_pulse_client returns singleton."""
        client1 = get_pulse_client(enabled=False)
        client2 = get_pulse_client(enabled=False)
        
        assert client1 is client2


class TestHelperFunctions:
    """Test helper functions."""

    def test_create_query_id(self):
        """Test query ID creation."""
        qid = create_query_id()
        assert isinstance(qid, str)
        assert len(qid) > 0

    def test_query_context_from_request(self):
        """Test creating QueryContext from request parameters."""
        ctx = query_context_from_request(
            query_id="q123",
            persona="aither",
            effort=5,
            tokens_estimated=100,
        )
        
        assert ctx.query_id == "q123"
        assert ctx.persona == "aither"
        assert ctx.effort == 5
        assert ctx.start_time is not None

    def test_query_context_generates_id(self):
        """Test that QueryContext generates ID if not provided."""
        ctx = query_context_from_request()
        assert ctx.query_id is not None
        assert len(ctx.query_id) > 0

