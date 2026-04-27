## AitherShell v1.0.0 - Production Release

**A closed-source, commercial CLI for running AI agents from your terminal.**

### What's New

- ✅ Compiled Windows binary (aither.exe, 86 MB)
- ✅ License key validation (free/pro/enterprise tiers)
- ✅ Interactive REPL with command history
- ✅ Multiple output formats (text, JSON, private)
- ✅ Persona support (12 AI personas)
- ✅ Safety levels (paranoid, strict, professional, relaxed)
- ✅ Effort-based model routing (1-10)
- ✅ Plugin system with 10 built-in plugins

### Installation

**Windows:**
- Download ither.exe from below
- Run the executable

**macOS/Linux (coming soon):**
\\\ash
chmod +x aither
export AITHERIUM_LICENSE_KEY="your-key"
./aither --help
\\\

**Homebrew (coming soon):**
\\\ash
brew install aithershell
\\\

### Get a License

- **Free:** 5 queries/day → https://aitherium.com/free
- **Pro:** /month (unlimited) → https://aitherium.com/pro
- **Enterprise:** Custom pricing → sales@aitherium.com

### Quick Start

\\\ash
# Set your license key
export AITHERIUM_LICENSE_KEY="free:user123:2026-12-31:signature..."

# Interactive shell
aither

# Single query
aither prompt "What is AitherOS?"

# With specific persona
aither prompt "Explain quantum computing" --will aither-scientist
\\\

### System Requirements

- Windows 7+ (x86_64)
- macOS 10.12+ (Intel or Apple Silicon - coming soon)
- Linux 2.6.32+ (coming soon)
- No external dependencies required

### Licensing

**Proprietary License** - See LICENSE for details

This is a closed-source commercial product. Unauthorized copying, modification, or distribution is prohibited.

### Support

- Documentation: https://docs.aitherium.com/aithershell
- Issues: https://github.com/Aitherium/aithershell/issues
- Email: support@aitherium.com

### Roadmap

- macOS binary (week 2)
- Linux binary (week 2)
- Docker image (week 3)
- Homebrew package (week 3)
- Windows installer (week 4)
- Web portal at shell.aitherium.com (week 4)

---

**Built with ❤️ by Aitherium**
