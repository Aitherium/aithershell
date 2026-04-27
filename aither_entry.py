"""PyInstaller entry shim - calls the click CLI entry point."""
from aithershell.cli import entry

if __name__ == "__main__":
    entry()
