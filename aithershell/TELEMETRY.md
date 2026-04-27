"""
AitherShell Telemetry Integration Guide
========================================

This document explains how to integrate AitherShell telemetry into your CLI commands.

## Quick Start

```python
from aithershell.events import get_pulse_client, create_query_id
from aithershell.metrics import get_metrics_collector
from aithershell.telemetry_config import TelemetryContext, load_telemetry_config

# Load configuration
config = load_telemetry_config()

# Initialize telemetry
pulse = get_pulse_client(
    pulse_url=config.pulse_url,
    enabled=config.emit_events,
)
pulse.start()

metrics = get_metrics_collector(enabled=config.metrics_export)

# Execute a query with automatic telemetry
query_id = create_query_id()
with TelemetryContext(query_id, model="gpt-4", effort=5) as ctx:
    result = execute_query("What is 2+2?")
    ctx.tokens_used = 10
    ctx.duration_ms = 150

pulse.stop()
```

## Configuration

AitherShell telemetry is configured via (in order of precedence):

1. **Environment Variables** (highest priority):
   - `AITHER_TELEMETRY_EMIT_EVENTS=true|false`
   - `AITHER_TELEMETRY_PULSE_URL=http://localhost:8081`
   - `AITHER_TELEMETRY_METRICS=true|false`
   - `AITHER_TELEMETRY_PRIVACY=public|private|redacted`

2. **Project Config** (`.aither.yaml`):
   ```yaml
   observability:
     emit_events: true
     pulse_url: "http://localhost:8081"
     flush_interval_ms: 1000
     metrics_export: true
   ```

3. **User Config** (`~/.aither/config.yaml`):
   ```yaml
   observability:
     default_privacy_level: "private"
     save_history: false
   ```

4. **Built-in Defaults**

## Event Types

### 1. Query Started
Emitted when user query begins execution.

```python
client.emit_query_started(
    query_id="abc123",
    persona="assistant",
    effort=5,
    tokens_estimated=1000,
    query_text="What is AI?",  # Filtered if privacy_level="private"
    privacy_level="public",
)
```

**Payload:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "service": "aithershell",
  "event": "aithershell.query.started",
  "privacy_level": "public",
  "data": {
    "query_id": "abc123",
    "persona": "assistant",
    "effort": 5,
    "tokens_estimated": 1000,
    "query_text": "What is AI?"
  }
}
```

### 2. Query Completed
Emitted when query finishes successfully.

```python
client.emit_query_completed(
    query_id="abc123",
    duration_ms=1234.5,
    tokens_used=150,
    model="gpt-4",
    cached=False,
)
```

### 3. Error Event
Emitted when error occurs during query.

```python
client.emit_error(
    error_type="api_timeout",
    severity="error",  # info, warning, error, critical
    retry_count=2,
    message="Request timed out after 30s",
    query_id="abc123",
)
```

### 4. Plugin Executed
Emitted when plugin executes.

```python
client.emit_plugin_executed(
    plugin_name="github_plugin",
    duration_ms=500.0,
    success=True,
    error_message=None,
)
```

## Privacy Modes

### Public (Default)
- Query text included in events
- Full context available for analysis
- **Use for:** Development, internal testing

### Private
- Query text excluded from events
- Query ID preserved for correlation
- **Use for:** Production, sensitive queries

### Redacted
- Sensitive fields hashed (e.g., query text)
- Still useful for metrics without revealing content
- **Use for:** Hybrid scenarios

**Example:**
```python
with TelemetryContext(query_id, privacy_level="private") as ctx:
    # This query text will NOT appear in events
    result = execute_secret_query()
```

## Prometheus Metrics

AitherShell exports 6 Prometheus metrics:

### 1. Counter: Queries Total
```
aithershell_queries_total{model="gpt-4", effort="5"} 42
```
Total queries executed by model and effort level.

### 2. Histogram: Query Duration (seconds)
```
aithershell_query_duration_seconds_bucket{le="1.0"} 10
aithershell_query_duration_seconds_bucket{le="5.0"} 35
aithershell_query_duration_seconds_bucket{le="10.0"} 40
aithershell_query_duration_seconds_sum 156.23
aithershell_query_duration_seconds_count 42
```
Query execution duration with buckets: 1s, 5s, 10s.

### 3. Histogram: Tokens Used
```
aithershell_tokens_used_bucket{le="500"} 5
aithershell_tokens_used_bucket{le="2000"} 30
aithershell_tokens_used_bucket{le="8000"} 42
aithershell_tokens_used_sum 45890
aithershell_tokens_used_count 42
```
Tokens consumed per query.

### 4. Counter: Errors Total
```
aithershell_errors_total{error_type="api_timeout"} 2
aithershell_errors_total{error_type="auth_failed"} 1
```
Total errors by type.

### 5. Counter: Cache Hits
```
aithershell_model_cache_hits_total{model="gpt-4"} 8
```
Cache hits per model.

### 6. Gauge: Cloud Compute Costs
```
aithershell_cloud_compute_costs_usd_total 12.34
```
Total cloud compute costs in USD.

## Batch Export Format

Events are batched and sent to Pulse at `POST /metrics`:

```json
{
  "batch": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "service": "aithershell",
      "event": "aithershell.query.started",
      "privacy_level": "public",
      "data": { ... }
    },
    ...
  ],
  "count": 42,
  "timestamp": "2024-01-15T10:30:01Z"
}
```

**Flush Interval:** 1000ms (configurable)
**Max Batch Size:** 100 events (auto-flush when reached)

## Service Registration

On startup, AitherShell registers with Pulse:

```
POST http://localhost:8081/services/register
{
  "service_name": "aithershell",
  "port": 0,
  "container_name": "client:hostname",
  "transport_endpoints": ["stdio", "http_client"],
  "environment": "local",
  "client_version": "1.0.0",
  "capabilities": ["query_execution", "artifact_browse"]
}
```

## Heartbeat

Every 30 seconds, AitherShell sends a health report:

```
POST http://localhost:8081/health/report
{
  "service_name": "aithershell",
  "status": "healthy",
  "consecutive_failures": 0,
  "metrics": {
    "queries_completed": 42,
    "errors_total": 1,
    "local_cache_hits": 8,
    "uptime_seconds": 3600
  }
}
```

## Error Handling

**All telemetry operations are non-blocking:**
- Emit operations return immediately
- Pulse communication happens in background threads
- Network errors don't interrupt query execution
- Graceful degradation if Pulse unavailable

**Logging:**
- Telemetry errors logged to stderr (not stdout)
- Use `logging.getLogger("aithershell.telemetry")` to access
- Default level: WARNING (only errors shown)

**Example:**
```python
import logging

# Show debug telemetry logs
logger = logging.getLogger("aithershell.telemetry")
logger.setLevel(logging.DEBUG)
```

## Integration Points

### In CLI Entry (cli.py)

```python
from aithershell.telemetry_config import load_telemetry_config

async def entry():
    # Load config
    config = load_telemetry_config()
    
    # Initialize telemetry
    pulse = get_pulse_client(pulse_url=config.pulse_url)
    pulse.start()
    
    try:
        # ... CLI logic ...
    finally:
        pulse.stop()
```

### In Query Execution

```python
from aithershell.telemetry_config import TelemetryContext, create_query_id

async def execute_query(query_text, model, effort, privacy_level):
    query_id = create_query_id()
    
    with TelemetryContext(
        query_id,
        model=model,
        effort=effort,
        privacy_level=privacy_level,
    ) as ctx:
        try:
            result = await api.query(query_text)
            ctx.tokens_used = result.tokens
            ctx.cached = result.from_cache
            return result
        except Exception as e:
            # Error will be automatically emitted by context
            raise
```

### Metrics Export Endpoint

If running AitherShell as a service with HTTP endpoint:

```python
from fastapi import FastAPI
from aithershell.metrics import get_metrics_collector

app = FastAPI()

@app.get("/metrics")
async def metrics():
    collector = get_metrics_collector()
    return collector.to_prometheus_text()
```

## Testing

Run telemetry tests:

```bash
pytest aithershell/test_telemetry.py -v
```

Key test areas:
- Event payload schema validation
- Privacy mode filtering
- Metrics collection
- Batch export format
- Error handling (no blocking)
- RBAC capabilities
- Graceful degradation

## Troubleshooting

### Events not appearing in Pulse
1. Check `emit_events: true` in config
2. Verify Pulse URL: `echo $AITHER_TELEMETRY_PULSE_URL`
3. Check firewall: `curl http://localhost:8081/health`
4. Enable debug logging: `logging.getLogger("aithershell.telemetry").setLevel("DEBUG")`

### High memory usage
1. Reduce `max_batch_size` (default: 100)
2. Reduce `flush_interval_ms` (default: 1000)
3. Disable `trace_queries` option

### Privacy concerns
1. Use `privacy_level="private"` to exclude query text
2. Disable `save_history` in config
3. Use `metrics_export` without events for metrics-only mode

## Best Practices

1. **Always use TelemetryContext** for automatic error handling
2. **Set privacy_level appropriately** for your use case
3. **Don't modify event_queue directly** - use public emit methods
4. **Call pulse.stop()** in finally block to flush remaining events
5. **Use metrics for SLO tracking**, events for debugging
6. **Monitor consecutive_failures** for Pulse availability

## Related Files

- `events.py` - Pulse event client
- `metrics.py` - Prometheus metrics collector
- `telemetry_config.py` - Configuration and context manager
- `test_telemetry.py` - Test suite
"""
