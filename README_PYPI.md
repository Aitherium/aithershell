# AitherShell v1.0

**The kernel shell for AitherOS** — Like Claude Code CLI is to Claude, AitherShell is to AitherOS.

[![PyPI version](https://badge.fury.io/py/aithershell.svg)](https://badge.fury.io/py/aithershell)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/aitherium/aithershell/actions/workflows/publish.yml/badge.svg)](https://github.com/aitherium/aithershell/actions)

## What is AitherShell?

AitherShell is a powerful, extensible CLI shell that gives you direct access to AitherOS capabilities:

- **Interactive REPL** — Talk to AitherOS like you would a person
- **Batch mode** — Script complex workflows
- **10 built-in plugins** — scope, secret, project, marketplace, companion, cloud, aitherzero, github_app, gaming, setup
- **Extensible** — Drop YAML or Python plugins into `~/.aither/plugins/`
- **Observable** — Every query tracked in Pulse + Prometheus + Grafana
- **Privacy-first** — Private mode excludes query text from logs
- **Multi-model** — Route queries across local and cloud models

---

## Quick Start

### Installation

```bash
# With pip
pip install aithershell

# With Homebrew (macOS/Linux)
brew install aitherium/aithershell/aithershell

# With Conda
conda install -c aitherium aithershell

# With Docker
docker run -it aitherium/aithershell:latest
```

### First Query

```bash
# Interactive mode
aither

# Single query
aither "What is the AitherOS architecture?"

# JSON output (for scripts)
aither --json "List all services" | jq '.result'

# Private mode (no logging)
aither --private "My secret query"

# Control effort level
aither --effort 8 "Deep analysis required"
```

### Output Formats

```bash
# Plain text (default)
aither "question"

# JSON (for parsing)
aither --json "question"

# Pretty text (rich formatting)
aither --print "question"  

# Pipe to other tools
aither "list" --print | grep "service"
```

---

## Configuration

Create `~/.aither/config.yaml`:

```yaml
# Genesis server
url: http://localhost:8001

# Model to use
model: aither-orchestrator

# Effort level (1-10, higher = more compute)
effort: 5

# Safety level
safety_level: professional

# Stream responses
stream: true

# Enable telemetry
telemetry_enabled: true

# Pulse server (for events)
pulse_url: http://localhost:8081
```

All config values can be overridden via CLI flags or environment variables:

```bash
# CLI override
aither --model gpt-4 "question"

# Environment variables
export AITHER_URL=http://localhost:8001
export AITHER_MODEL=aither-orchestrator
export AITHER_EFFORT=8
aither "question"
```

---

## Plugins

### Using Built-in Plugins

```bash
# Scope plugin (view system graph)
aither scope show services

# Secret plugin (manage secrets)
aither secret list
aither secret get database-password

# Project plugin (project management)
aither project status
aither project deploy staging

# Marketplace plugin (browse plugins)
aither marketplace search react
aither marketplace install salesforce-connector

# Companion plugin (AI companion)
aither companion chat

# Cloud plugin (cloud operations)
aither cloud sync S3

# AitherZero plugin (automation scripts)
aither aitherzero run backup

# GitHub App plugin
aither github-app list-prs

# Gaming plugin (fun mode)
aither gaming play quiz

# Setup plugin (configuration)
aither setup init
aither setup wizard
```

### Creating Custom Plugins

Drop a file in `~/.aither/plugins/`:

**YAML Plugin** (`~/.aither/plugins/status.yaml`):
```yaml
name: status
description: Show system status
aliases: [st]
action:
  type: api
  method: GET
  url: "{genesis}/health"
output: json
```

**Python Plugin** (`~/.aither/plugins/custom.py`):
```python
from aithershell_agent_framework import AgentBuilder

class CustomPlugin(AgentBuilder):
    name = "custom"
    description = "My custom plugin"
    
    async def execute(self, ctx):
        result = await self.query("What do you need?")
        return result
```

---

## Shell Completions

Generate shell completions:

```bash
# Bash
aither --completions bash | sudo tee /etc/bash_completion.d/aither
source /etc/bash_completion.d/aither

# Zsh
aither --completions zsh | sudo tee /usr/local/share/zsh/site-functions/_aither
compinit

# Fish
aither --completions fish | sudo tee /usr/local/share/fish/vendor_completions.d/aither.fish

# PowerShell
aither --completions pwsh | Out-File -Encoding UTF8 $PROFILE
. $PROFILE
```

Then:
```bash
aither --<TAB>  # See all options
aither scope <TAB>  # See scope subcommands
```

---

## Advanced Usage

### Effort Levels

```bash
# Fast, cheap (effort 1-3)
aither --effort 1 "Quick summary"

# Balanced (effort 4-6)
aither --effort 5 "Detailed analysis"

# Deep, expensive (effort 7-8)
aither --effort 7 "Comprehensive research"

# Critical (effort 9-10)
aither --effort 10 "Mission-critical decision"
```

### Safety Levels

```bash
aither --safety paranoid "Assume adversarial inputs"
aither --safety strict "Standard security"
aither --safety professional "Balanced approach"
aither --safety relaxed "Trust user input"
```

### Personas (Wills)

```bash
aither --will aither-prime "Query as primary persona"
aither --will researcher "Query as researcher"
aither --will executor "Query as executor"
```

### Streaming Responses

```bash
# Stream (default)
aither "long query"

# No streaming
aither --no-stream "long query"
```

---

## Telemetry & Monitoring

AitherShell emits comprehensive telemetry:

- **Pulse Events:** 3 events per query (query.started, query.completed, error)
- **Prometheus Metrics:** Query volume, latency, error rate, model distribution
- **Grafana Dashboard:** `http://localhost:3000/d/aithershell`

All telemetry can be disabled:

```yaml
# ~/.aither/config.yaml
telemetry_enabled: false
```

Or per-query:
```bash
aither --private "secret query"  # Excludes query_text from logs
```

---

## Troubleshooting

### Connection refused to Genesis

```bash
# Check Genesis is running
curl http://localhost:8001/health

# Point to different Genesis server
aither --url http://production.aitherium.com:8001 "query"

# Or set environment variable
export AITHER_URL=http://production.aitherium.com:8001
```

### Plugin not found

```bash
# List available plugins
aither --plugins

# Check plugin directory
ls ~/.aither/plugins/

# Validate plugin syntax
aither setup validate-plugin ~/.aither/plugins/custom.py
```

### Timeout errors

```bash
# Increase timeout (default 30s)
aither --timeout 60 "slow query"

# Or in config.yaml
timeout: 60
```

---

## Examples

### Data Analysis
```bash
aither --json "Analyze sales data from Q1" | jq '.trends[]'
```

### Code Generation
```bash
aither "Generate Python function to parse CSV files"
```

### System Administration
```bash
aither scope show infrastructure
aither project deploy production
```

### Research
```bash
aither --effort 8 "Comprehensive analysis of machine learning trends"
```

### Creative Writing
```bash
aither --will creative "Write a short story about AI"
```

---

## API Reference

```bash
aither COMMAND [OPTIONS] [QUERY]

Commands:
  (no command)         # Interactive REPL
  QUERY                # Single query
  setup                # Configuration setup
  --init               # Initialize config
  --config             # Show configuration
  --plugins            # List plugins
  --status             # Check Genesis health
  --history [COUNT]    # Show query history
  --completions SHELL  # Generate shell completions

Options:
  --url URL                   Genesis server URL
  --model MODEL               Model to use
  --effort LEVEL              Effort level (1-10)
  --safety LEVEL              Safety level (paranoid/strict/professional/relaxed)
  --will NAME                 Persona name
  --json                      JSON output
  --print                     Plain text output
  --private                   Private mode (no logging)
  --stream/--no-stream        Enable/disable streaming
  --timeout SECONDS           Query timeout
  --verbose                   Verbose logging
  --help                      Show help
  --version                   Show version
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](https://github.com/aitherium/aithershell/blob/main/CONTRIBUTING.md)

```bash
# Clone repo
git clone https://github.com/aitherium/aithershell.git
cd aithershell

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Lint
ruff check aithershell/
mypy aithershell/

# Format
black aithershell/
```

---

## License

MIT License — See [LICENSE](https://github.com/aitherium/aithershell/blob/main/LICENSE)

---

## Community

- 💬 [GitHub Discussions](https://github.com/aitherium/aithershell/discussions)
- 🐛 [Issue Tracker](https://github.com/aitherium/aithershell/issues)
- 📖 [Documentation](https://github.com/aitherium/aithershell/wiki)
- 💙 [Discord Community](https://discord.gg/aitherium)

---

## Roadmap

- ✅ Core CLI shell (v1.0)
- ✅ 10 built-in plugins
- ✅ Telemetry & observability
- 🔄 Web terminal in AitherPortal (v1.1)
- 🔄 Agent Framework SDK (v1.2)
- 🔄 Plugin marketplace (v1.3)
- 🔄 Enterprise deployment (v2.0)

---

**Made with ❤️ by the Aitherium team**
