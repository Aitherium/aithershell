"""
AitherShell - The Kernel Shell for AitherOS
============================================

The primary programmable interface to AitherOS.
Like bash is to Linux, AitherShell is to AitherOS.

Usage:
    aither                                  # Interactive shell
    aither "question"                       # Single query
    aither --print "question"               # Script mode (plain text)
    aither --json "question"                # Script mode (JSON)
    echo "prompt" | aither --print          # Stdin pipe
    aither --private                        # Private mode
    aither --will aither-prime              # Switch persona
"""

__version__ = "0.1.0"
