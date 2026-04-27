#!/usr/bin/env python
"""
AitherShell - The kernel shell for AitherOS

Setup script for PyPI distribution.
"""

from setuptools import setup, find_packages

setup(
    name="aithershell",
    version="1.0.0",
    description="AitherShell - The kernel shell for AitherOS",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Aitherium",
    author_email="alex@aitherium.com",
    url="https://github.com/aitherium/aithershell",
    license="MIT",
    packages=find_packages(exclude=["tests", "examples"]),
    python_requires=">=3.10",
    install_requires=[
        "httpx>=0.26.0",
        "pyyaml>=6.0",
        "rich>=13.0",
        "argcomplete>=3.0",
        "click>=8.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.20.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1.0",
            "mypy>=1.0",
        ],
        "voice": [
            "sounddevice",
            "numpy",
            "webrtcvad",
        ],
        "all": [
            "sounddevice",
            "numpy",
            "webrtcvad",
        ],
    },
    entry_points={
        "console_scripts": [
            "aither=aithershell.cli:entry",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development",
        "Topic :: System :: Shells",
    ],
    keywords=[
        "aitheros",
        "shell",
        "cli",
        "agent",
        "ai",
        "kernel",
        "automation",
    ],
    project_urls={
        "Documentation": "https://github.com/aitherium/aithershell/wiki",
        "Source": "https://github.com/aitherium/aithershell",
        "Tracker": "https://github.com/aitherium/aithershell/issues",
    },
)
