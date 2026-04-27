# AitherShell Standalone Repository — Setup & Deployment Guide

## Overview

AitherShell is now a standalone GitHub repository (`aitherium/aithershell`) with:
- Independent versioning (semantic versioning v1.0.0+)
- Separate CI/CD pipeline (tests, builds, publishes automatically)
- Multi-platform distribution (PyPI, Docker Hub, Homebrew, etc.)
- Clean separation from AitherOS monolith

**Repository Location:** `D:\aithershell\` (local) → `https://github.com/aitherium/aithershell` (remote)

---

## Step 1: Create GitHub Repository

### Via GitHub Web UI

1. Go to https://github.com/new
2. Create repository:
   - **Repository name:** `aithershell`
   - **Owner:** `aitherium` (organization)
   - **Description:** "The kernel shell for AitherOS"
   - **Visibility:** Public
   - **Initialize with:** None (we have existing code)
3. Click "Create repository"

### Result

GitHub will show:

```
Quick setup — choose your SSH or HTTPS key:
…or push an existing repository from the command line

git remote add origin https://github.com/aitherium/aithershell.git
git branch -M main
git push -u origin main
```

---

## Step 2: Push Local Repository to GitHub

```bash
cd D:\aithershell

# Add GitHub as remote
git remote add origin https://github.com/aitherium/aithershell.git

# Rename branch (convention: main for public repos)
git branch -M main

# Push everything
git push -u origin main

# Verify
git remote -v
```

Expected output:

```
origin  https://github.com/aitherium/aithershell.git (fetch)
origin  https://github.com/aitherium/aithershell.git (push)
```

---

## Step 3: Configure GitHub Secrets

Navigate to: **Settings → Secrets and variables → Actions**

Add these secrets (required for CI/CD auto-publish):

| Secret | Value | Source |
|--------|-------|--------|
| `PYPI_API_TOKEN` | PyPI token with `aithershell` scope | https://pypi.org/account/tokens/ (create scoped token) |
| `DOCKERHUB_USERNAME` | Docker Hub username | Docker Hub account settings |
| `DOCKERHUB_TOKEN` | Docker Hub access token | Docker Hub account settings |

### How to Get Tokens

**PyPI Token:**
1. Go to https://pypi.org/account/tokens/
2. Create token: `aithershell-github-actions`
3. Scope: Only `aithershell` project
4. Copy token, add as `PYPI_API_TOKEN` secret

**Docker Hub Token:**
1. Go to https://hub.docker.com/settings/security
2. Create access token: `aithershell-github-actions`
3. Copy token, add as `DOCKERHUB_TOKEN` secret

---

## Step 4: Create Release Branch Structure

The workflow is optimized for this branch structure:

```
main          ← Production releases (tags v1.0.0, v1.0.1, etc.)
develop       ← Development branch (pull requests merge here)
```

Set this up:

```bash
cd D:\aithershell

# Create and push develop branch
git checkout -b develop
git push -u origin develop

# Set main as default branch on GitHub
# (Settings → Default branch → main)
```

---

## Step 5: Publish First Release

### Tag Version 1.0.0

```bash
cd D:\aithershell

# Tag commit
git tag v1.0.0

# Push tag (triggers GitHub Actions)
git push origin v1.0.0

# Verify
git tag -l
```

### What GitHub Actions Does (Automatically)

The `.github/workflows/publish.yml` workflow:

1. **Tests** (on 3 OS × 3 Python versions)
   - Runs: `pytest tests/ -v`
   - Duration: ~5 minutes

2. **Builds Distribution**
   - Creates wheel + tarball
   - Validates with twine
   - Duration: ~1 minute

3. **Publishes to PyPI**
   - Uses `PYPI_API_TOKEN` secret
   - URL: https://pypi.org/project/aithershell/
   - Duration: ~30 seconds

4. **Builds Docker Image**
   - Pushes to Docker Hub: `aitherium/aithershell:v1.0.0`
   - Also tags as `latest`
   - Duration: ~2 minutes

5. **Creates GitHub Release**
   - Attaches wheel + tarball as artifacts
   - URL: https://github.com/aitherium/aithershell/releases/tag/v1.0.0
   - Duration: ~30 seconds

**Total publish time: ~10 minutes**

---

## Step 6: Verify Publication

After 10 minutes, verify on each platform:

### PyPI

```bash
pip install aithershell
aither --version
# Should show: aithershell, version 1.0.0
```

URL: https://pypi.org/project/aithershell/

### Docker Hub

```bash
docker pull aitherium/aithershell:v1.0.0
docker run aitherium/aithershell:v1.0.0 aither --help
```

URL: https://hub.docker.com/r/aitherium/aithershell

### GitHub Releases

URL: https://github.com/aitherium/aithershell/releases

---

## Development Workflow

### For Contributors

```bash
# Clone the repo
git clone https://github.com/aitherium/aithershell.git
cd aithershell

# Create feature branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/new-plugin

# Make changes, test, commit
pytest tests/ -v
git add .
git commit -m "feat(plugin): add new-plugin"

# Push and create pull request
git push origin feature/new-plugin
# Then open PR on GitHub: develop ← feature/new-plugin
```

### For Releases

```bash
# On main branch
git checkout main
git pull origin main

# Tag new release
git tag v1.0.1

# Push tag (CI/CD does the rest)
git push origin v1.0.1
```

---

## Repository Structure

```
aithershell/
├── aithershell/          # Main package
│   ├── __init__.py
│   ├── cli.py           # CLI entry point
│   ├── shell.py         # REPL shell
│   ├── plugins.py       # Plugin system
│   ├── config.py        # Configuration
│   ├── telemetry.py     # Telemetry
│   └── plugins/         # 10 built-in plugins
├── tests/               # Unit tests (99 tests)
├── examples/            # Example scripts
├── completions/         # Shell completions
├── setup.py             # PyPI metadata
├── setup.cfg            # Config
├── pyproject.toml       # Build system
├── Dockerfile           # Docker image
├── Formula/             # Homebrew formula
├── .github/workflows/   # GitHub Actions (CI/CD)
├── README.md            # Main documentation
├── CHANGELOG.md         # Release notes
├── CONTRIBUTING.md      # Contributing guide
├── SECURITY.md          # Security policy
└── LICENSE              # MIT License
```

---

## CI/CD Triggers

The GitHub Actions workflow triggers on:

| Trigger | Event | Action |
|---------|-------|--------|
| `git push origin v*` | Tag push | Test → Build → Publish |
| `git push origin develop` | Push to develop | Run tests only |
| Pull request | PR created | Run tests only |

---

## Troubleshooting

### CI/CD Fails to Publish

**Check:** GitHub Secrets are set correctly

```bash
# On GitHub: Settings → Secrets and variables → Actions
# Verify:
# - PYPI_API_TOKEN is set (not empty)
# - DOCKERHUB_USERNAME is set
# - DOCKERHUB_TOKEN is set
```

**Check:** Token permissions

- PyPI token: Must have `aithershell` project scope
- Docker token: Must have full access or repo write permission

### Package Not Appearing on PyPI

**Check:** Wait 2-3 minutes (PyPI indexing delay)

**Check:** Verify tag was pushed

```bash
git push origin v1.0.0 --dry-run  # Test push
git push origin v1.0.0             # Actually push
```

**Check:** Workflow status on GitHub

https://github.com/aitherium/aithershell/actions

---

## Next Steps

1. ✅ Create GitHub repository
2. ✅ Configure secrets (PYPI_API_TOKEN, DOCKERHUB_*)
3. ✅ Push local repo to GitHub
4. 🔜 **Tag v1.0.0 and publish**
5. 🔜 Verify on PyPI, Docker Hub, GitHub Releases
6. 🔜 Update AitherOS README with link to aithershell repo
7. 🔜 Begin Phase 2: AitherPortal web terminal

---

## Commands Reference

```bash
# Local setup
cd D:\aithershell

# Add GitHub remote
git remote add origin https://github.com/aitherium/aithershell.git

# Push to GitHub
git branch -M main
git push -u origin main

# Create and push release tag
git tag v1.0.0
git push origin v1.0.0

# Verify publication
pip install aithershell
docker pull aitherium/aithershell:v1.0.0

# View workflow status
# https://github.com/aitherium/aithershell/actions
```

---

**Created:** 2026-04-27  
**Status:** Ready for GitHub repository creation and first release  
**Next Action:** Create GitHub repo and configure secrets, then tag v1.0.0
