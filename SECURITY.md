# Security Policy

## Reporting Security Vulnerabilities

**Do NOT open a public issue for security vulnerabilities.**

Instead, please email security@aitherium.com with:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge receipt within 24 hours and provide an update on next steps within 48 hours.

## Supported Versions

We provide security updates for:

- Latest release (e.g., 1.x.y)
- Previous major version (e.g., 0.x.y) for 6 months after latest release

## Security Considerations

### API Keys and Secrets

- Never commit secrets or API keys to the repository
- Use environment variables or `~/.aither/config.yaml` (gitignored)
- AitherShell integrates with AitherSecrets for credential management

### Network Security

- All communication with AitherOS services uses mTLS
- Requests are signed with HMAC-SHA256 capability tokens
- Telemetry data is encrypted in transit

### Data Privacy

- Local commands (shell, config) run on your machine only
- Agent requests are sent to configured AitherOS deployment
- Memory/context may be stored locally in `~/.aither/`
- No data is sent to third parties without explicit consent

## Security Best Practices

1. Keep AitherShell updated: `pip install --upgrade aithershell`
2. Use strong authentication with AitherOS
3. Rotate API keys regularly
4. Review logs in `~/.aither/logs/` for suspicious activity
5. Report security concerns immediately

## Dependencies

We maintain a security audit of all Python dependencies. See `requirements.txt` for the full list.

To check for vulnerable dependencies:

```bash
pip install safety
safety check
```

## Compliance

AitherShell follows:

- OWASP Top 10 secure coding practices
- CWE/SANS Top 25 recommendations
- PEP 3156 async/await security guidelines
- Python cryptographic best practices (via cryptography package)

---

For more information, visit https://aitherium.com/security
