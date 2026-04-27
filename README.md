# AitherShell

> The kernel shell for **AitherOS** — a powerful CLI for autonomous AI orchestration, knowledge management, and system control.

[![PyPI](https://img.shields.io/pypi/v/aithershell)](https://pypi.org/project/aithershell/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Tests](https://github.com/aitherium/aithershell/actions/workflows/publish.yml/badge.svg)](https://github.com/aitherium/aithershell/actions)

## Quick Start

### Installation

```bash
# via pip
pip install aithershell

# via Homebrew (coming soon)
brew install aithershell

# via Docker
docker run aitherium/aithershell:latest aither --help
```

### First Command

```bash
aither --help
aither --version
aither prompt "What is AitherOS?"
```

## Relationship to Aither Framework

**AitherShell** is the end-user CLI. **Developers** extend it by building agents using the **Aither** framework.

```
Developer (uses Aither) → Builds agents → Registers with platform
                                              ↓
End User (uses AitherShell) → Runs agents via `aither prompt`
```

**Learn more:** See [DEVELOPER-GUIDE.md](./DEVELOPER-GUIDE.md) for building custom agents with Aither.

For complete ecosystem overview: [Aitherium Ecosystem](../.AITHERIUM/ECOSYSTEM-PRODUCTS.md)

---

## Features

🚀 **Autonomous Agent Orchestration**
- Dispatch work to specialized agents (code, research, security, etc.)
- Real-time streaming responses with rich output
- Integrated telemetry & observability

📚 **Knowledge Graph Integration**
- Query codebase semantics (CodeGraph + Repowise)
- Persistent context across sessions
- RAG-powered retrieval with embeddings

🔐 **Enterprise-Ready**
- HMAC-SHA256 signed capability tokens
- Multi-workspace support
- Audit logging via AitherChronicle

🎯 **10 Production Plugins**
- `genesis` — AitherOS orchestrator integration
- `code` — CodeGraph semantic search
- `web` — Web search & content fetch
- `voice` — Text-to-speech & voice input
- `agent` — A2A agent dispatch
- `memory` — Working memory & context
- `script` — Shell automation & job runners
- `config` — Service & credential management
- `observe` — Telemetry & metrics
- `auth` — Identity & authorization

## Documentation

- **[Installation Guide](./docs/installation.md)** — Detailed setup for all platforms
- **[User Guide](./docs/user-guide.md)** — Commands, plugins, configuration
- **[API Reference](./docs/api-reference.md)** — Python SDK for automation
- **[Plugin Development](./docs/plugin-development.md)** — Build custom plugins
- **[Troubleshooting](./docs/troubleshooting.md)** — Common issues & solutions

## Architecture

```
aither (CLI)
├── Core Engine
│   ├── Shell (Interactive REPL)
│   ├── CLI Parser (Click)
│   └── Output Formatter (Rich)
├── Plugins (10 built-in)
│   ├── genesis (AitherOS)
│   ├── code (CodeGraph)
│   ├── web (Search)
│   ├── agent (A2A)
│   ├── memory (Context)
│   └── ... (5 more)
└── Infrastructure
    ├── Telemetry (Prometheus metrics)
    ├── Events (Flux pub/sub)
    ├── Logging (AitherChronicle)
    └── Auth (Capability tokens)
```

## Examples

### Interactive Mode
```bash
aither
> help
> system status
> query "What services are running?"
> code search "authentication"
> run-script deploy.sh
```

### Command Line
```bash
# Get Genesis status
aither genesis status

# Search codebase
aither code search "microservice"

# Dispatch agent task
aither agent dispatch research "explain AitherOS architecture"

# Manage configuration
aither config get SERVICE_PORT
aither config set LOGLEVEL DEBUG
```

### Python SDK
```python
from aithershell import AitherShell

shell = AitherShell()
response = shell.dispatch_agent("code", "find_symbols", {"query": "authentication"})
print(response)
```

## Requirements

- **Python:** 3.10, 3.11, or 3.12
- **OS:** Linux, macOS, or Windows
- **Memory:** 256 MB minimum
- **Network:** Internet access for agent features

## Configuration

AitherShell stores configuration in `~/.aither/`:

```
~/.aither/
├── config.yaml          # Global configuration
├── plugins/             # Installed plugins
├── history              # Command history
└── cache/               # Cached responses
```

### Environment Variables

```bash
AITHER_LOGLEVEL=DEBUG           # Logging level (INFO, DEBUG, WARNING)
AITHER_PLUGINS_DIR=./plugins    # Custom plugin directory
AITHER_TELEMETRY=enabled        # Enable Prometheus metrics
AITHER_WORKSPACE=myspace        # Workspace name
```

## Development

### Setup

```bash
git clone https://github.com/aitherium/aithershell.git
cd aithershell
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=aithershell --cov-report=html
```

### Building Distribution

```bash
python -m build
twine upload dist/*
```

### Building Docker Image

```bash
docker build -t aitherium/aithershell:latest .
docker run aitherium/aithershell:latest aither --version
```

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Write tests for your changes
4. Submit a pull request

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for detailed guidelines.

## License

MIT License — see [`LICENSE`](./LICENSE) for details.

## Community

- **Issues:** [GitHub Issues](https://github.com/aitherium/aithershell/issues)
- **Discussions:** [GitHub Discussions](https://github.com/aitherium/aithershell/discussions)
- **Security:** [Security Policy](./SECURITY.md)

---

**Part of the [Aitherium](https://aitherium.com) platform.**

For more information about AitherOS, visit [AitherOS Documentation](https://github.com/aitherium/AitherOS/wiki).
