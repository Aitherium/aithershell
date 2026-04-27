# Contributing to AitherShell

We love your input! We want to make contributing to AitherShell as easy and transparent as possible.

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `develop`
2. If you've added code that should be tested, add tests
3. Make sure your code lints (use `ruff check`)
4. Issue that pull request!

## Pull Request Process

1. Update the README.md with any new features
2. Update the CHANGELOG.md with a note on your changes
3. Increase the version number in setup.py to the new version
4. Ensure all tests pass: `pytest tests/ -v`
5. Your PR will be reviewed and merged once approved

## Reporting Bugs

When reporting a bug, please include:

- Your operating system name and version
- Python version (`python --version`)
- Detailed steps to reproduce
- Expected behavior
- Actual behavior
- Any error messages or stack traces

## Suggesting Enhancements

Feature suggestions are tracked as GitHub Issues. Provide:

- Use case: why you need this feature
- Expected behavior: how it should work
- Current behavior: what's missing
- Examples of the feature in action

## Code Style

We follow PEP 8 with some exceptions:

- Use 4 spaces for indentation
- Line length: 100 characters (enforced by ruff)
- Type hints required for public APIs
- Docstrings required for all modules/classes/functions

Run linting:
```bash
ruff check aithershell tests --fix
```

## Testing

All new code must have tests:

```bash
pytest tests/ -v
pytest tests/test_cli.py::test_command_name -v  # Single test
pytest tests/ --cov=aithershell --cov-report=html  # With coverage
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

---

Questions? Open an issue or start a discussion!
