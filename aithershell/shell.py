"""
AitherShell Interactive REPL
=============================

Interactive shell environment with:
- Multi-line input support (end with "...")
- Command history saved to ~/.aither/history
- Graceful Ctrl+C handling
- Response streaming
- Rich terminal output
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional, List

from aithershell.config import AitherConfig, CONFIG_DIR
from aithershell.genesis_client import GenesisClient, GenesisError

logger = logging.getLogger(__name__)


class AitherREPL:
    """
    Interactive REPL for AitherShell.
    
    Features:
    - Multi-line input (end with "...")
    - Command history
    - Streaming responses
    - Interrupt handling
    """

    def __init__(self, config: AitherConfig):
        """Initialize REPL.
        
        Args:
            config: AitherConfig instance
            
        Raises:
            ValueError: If config is invalid
        """
        if not config:
            raise ValueError("config cannot be None")
        
        self.config = config
        self.genesis_client = GenesisClient(
            base_url=config.url,
            timeout=30.0,
            enable_logging=True,
        )
        self.history: List[str] = []
        self.history_file = Path(config.history_file)
        self._load_history()
        self.interrupted = False

    def _load_history(self) -> None:
        """Load command history from disk.
        
        Reads from ~/.aither/history, limits to max_history lines.
        """
        try:
            if self.history_file.exists():
                lines = self.history_file.read_text(encoding="utf-8").splitlines()
                self.history = lines[-self.config.max_history :]
        except Exception as e:
            logger.warning(f"Failed to load history: {e}")

    def _save_history(self) -> None:
        """Save command history to disk.
        
        Saves to ~/.aither/history, maintains max_history limit.
        """
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            lines = self.history[-self.config.max_history :]
            self.history_file.write_text("\n".join(lines), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")

    def _handle_sigint(self, signum, frame):
        """Handle Ctrl+C interrupt.
        
        Args:
            signum: Signal number
            frame: Stack frame
        """
        self.interrupted = True
        print("\n^C")

    async def handle_input(self, user_input: str) -> Optional[str]:
        """Process user input and return response.
        
        Args:
            user_input: User query/command
            
        Returns:
            Response text or None if cancelled
            
        Raises:
            GenesisError: If Genesis request fails
        """
        if not user_input or not user_input.strip():
            return None
        
        user_input = user_input.strip()
        
        # Add to history
        self.history.append(user_input)
        self._save_history()
        
        # Check for built-in commands
        if user_input.startswith("/"):
            return await self._handle_command(user_input)
        
        # Stream to Genesis
        response_text = ""
        try:
            async for chunk in self.genesis_client.chat_stream(
                message=user_input,
                persona=self.config.persona,
                effort=self.config.effort,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                safety_level=self.config.safety_level,
            ):
                if self.interrupted:
                    self.interrupted = False
                    return None
                
                response_text += chunk
                
                # Stream to stdout if enabled
                if self.config.stream:
                    print(chunk, end="", flush=True)
            
            if response_text:
                print()  # Newline after streaming response
            
            return response_text
            
        except GenesisError as e:
            print(f"\n[ERROR] {e.message}", file=sys.stderr)
            logger.error(f"Genesis error: {e}")
            raise

    async def _handle_command(self, cmd: str) -> Optional[str]:
        """Handle built-in commands.
        
        Args:
            cmd: Command string (starts with "/")
            
        Returns:
            Command output or None
        """
        cmd = cmd.lstrip("/").strip()
        parts = cmd.split(None, 1)
        command = parts[0] if parts else ""
        arg = parts[1] if len(parts) > 1 else ""
        
        if command == "help":
            return self._cmd_help()
        elif command == "history":
            return self._cmd_history()
        elif command == "clear":
            self.history.clear()
            self._save_history()
            return "History cleared."
        elif command == "exit" or command == "quit":
            raise KeyboardInterrupt()
        else:
            return f"Unknown command: /{command}. Type /help for help."

    def _cmd_help(self) -> str:
        """Show help for built-in commands."""
        return """
AitherShell Built-in Commands:

/help               Show this help
/history            Show command history
/clear              Clear history
/exit, /quit        Exit shell

Multi-line input: end a line with "..." to continue
"""

    def _cmd_history(self) -> str:
        """Show command history."""
        if not self.history:
            return "History is empty."
        
        lines = [f"{i+1:3d} {cmd}" for i, cmd in enumerate(self.history[-20:])]
        return "\n".join(lines)

    async def run_repl(self) -> None:
        """Run the interactive REPL.
        
        Raises:
            KeyboardInterrupt: When user exits
            GenesisError: If Genesis is unavailable
        """
        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._handle_sigint)
        
        # Check Genesis health
        print("Checking Genesis connection...", file=sys.stderr)
        healthy = await self.genesis_client.health_check()
        if not healthy:
            print(
                "[ERROR] Genesis is not responding. Check your connection.",
                file=sys.stderr,
            )
            raise GenesisError("Genesis health check failed")
        
        print(f"Welcome to AitherShell {self.config.url}")
        print("Type /help for commands, Ctrl+C to exit\n")
        
        try:
            while True:
                try:
                    # Read input (may be multi-line)
                    user_input = await self._read_input()
                    if user_input is None:
                        break
                    
                    # Process input
                    await self.handle_input(user_input)
                    
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                    
        finally:
            await self.genesis_client.close()

    async def _read_input(self) -> Optional[str]:
        """Read input from stdin, supporting multi-line.
        
        Returns:
            User input string or None if EOF
            
        Multi-line input: end line with "..." to continue
        """
        try:
            lines = []
            
            while True:
                # Use asyncio to prevent blocking the event loop
                line = await asyncio.get_event_loop().run_in_executor(
                    None,
                    input,
                    self.config.prompt if not lines else "... ",
                )
                
                if not line:
                    continue
                
                lines.append(line)
                
                # Check for multi-line continuation
                if line.endswith("..."):
                    lines[-1] = lines[-1][:-3]  # Remove "..."
                    continue
                
                break
            
            result = " ".join(lines)
            return result if result.strip() else None
            
        except EOFError:
            return None
        except KeyboardInterrupt:
            self.interrupted = True
            return None


async def run_repl(config: AitherConfig) -> None:
    """Run the AitherShell REPL.
    
    Args:
        config: AitherConfig instance
        
    Raises:
        GenesisError: If Genesis is unavailable
    """
    repl = AitherREPL(config)
    await repl.run_repl()
