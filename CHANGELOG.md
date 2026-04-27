# Changelog

All notable changes to AitherShell are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-27

### Added
- ✨ Initial public release
- 🎯 Core CLI shell with interactive REPL
- 🔌 10 production-ready plugins:
  - `genesis` — AitherOS orchestrator integration
  - `code` — CodeGraph semantic search
  - `web` — Web search & content fetching
  - `voice` — Text-to-speech & voice input
  - `agent` — A2A agent dispatch
  - `memory` — Working memory & context management
  - `script` — Shell automation & job runners
  - `config` — Service & credential management
  - `observe` — Telemetry & metrics collection
  - `auth` — Identity & authorization
- 📊 Telemetry integration (Prometheus metrics, AitherChronicle logging)
- 🔐 HMAC-SHA256 capability tokens for secure agent dispatch
- 📚 Comprehensive documentation (user guide, API reference, plugin dev guide)
- 🐳 Docker image support (Python 3.10-slim base)
- 📦 Multi-platform distribution (pip, Homebrew, conda, Docker)
- ✅ 99 unit tests with full coverage
- 🔄 GitHub Actions CI/CD (test on 3 OS × 3 Python versions, auto-publish)

### Features
- Interactive shell mode with command history and tab completion
- Command-line mode for scripting and automation
- Plugin system for extensibility
- Configuration management (YAML in ~/.aither/)
- Event streaming from AitherOS services
- Rich output formatting with tables, trees, syntax highlighting
- Telemetry dashboard integration
- Multi-workspace support

### Supported Platforms
- Linux (x86_64, ARM64)
- macOS (Intel, Apple Silicon)
- Windows (x86_64)

### Supported Python Versions
- Python 3.10
- Python 3.11
- Python 3.12

### Known Limitations
- Remote shell execution requires AitherOS deployment
- Some plugins require specific AitherOS services running
- Voice input requires FFmpeg on Linux/macOS

## Unreleased

### Planned for 1.1.0
- [ ] Conda-forge distribution
- [ ] npm wrapper for JavaScript ecosystem
- [ ] Standalone binary builds (PyInstaller)
- [ ] Obsidian integration via Vault plugin
- [ ] VS Code extension for embedded shell

### Planned for 1.2.0
- [ ] Plugin marketplace (hosted)
- [ ] Custom plugin templates
- [ ] Advanced debugging tools
- [ ] Performance profiler

### Planned for 2.0.0 (Agent Framework SDK)
- [ ] Closed-source SDK for enterprise
- [ ] License key verification system
- [ ] Advanced agent orchestration
- [ ] Custom capability definitions

---

See https://github.com/aitherium/aithershell/releases for detailed release notes.
