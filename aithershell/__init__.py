"""
AitherShell — The Agent OS for Local + Cloud LLMs
==================================================

The unified package for AitherOS agent execution. Combines:
  * Engine: agents, fleets, backends (Ollama/vLLM/cloud), MCP, memory, pairing
    (formerly aither-adk — now merged here)
  * CLI: the `aither` command, license validation, terminal UX
  * Distribution: PyInstaller binary build

Two ways to use:
  1. As a library:  pip install aithershell
                    from aithershell import AitherAgent, LLMRouter
  2. As a CLI:      aither              # interactive
                    aither "hello"      # one-shot
                    aither init         # setup wizard
                    aither link         # link to portal

Library API mirrors the former `adk.*` namespace; existing `from adk import X`
will keep working via the deprecated `aither-adk` shim package.
"""

__version__ = "1.1.0"

from aithershell.agent import AitherAgent
from aithershell.tools import tool, ToolRegistry
from aithershell.llm import LLMRouter
from aithershell.config import Config

__all__ = [
    "AitherAgent",
    "tool",
    "ToolRegistry",
    "LLMRouter",
    "Config",
    "AgentRegistry",
    "AgentForge",
    "FleetConfig",
    "ConversationStore",
    # Extended capabilities
    "LoopGuard",
    "AitherSandbox",
    "AgentMeter",
    "SkillManifest",
    # Observability
    "ChronicleClient",
    "WatchReporter",
    "MetricsCollector",
    "PulseClient",
    # Core infrastructure
    "ServiceBridge",
    "EventEmitter",
    "IntakeGuard",
    "ContextManager",
    "register_builtin_tools",
    "GraphMemory",
    "NanoGPT",
    "NeuronPool",
    "AutoNeuronFire",
    "DegenerationDetector",
    "strip_internal_tags",
    "CATEGORY_TOOLS",
    # Elysium cloud
    "Elysium",
    # Faculty graphs (local knowledge)
    "CodeGraph",
    "MemoryGraph",
    "EmbeddingProvider",
    # Standalone tools
    "repowise_search",
    "swarm_code",
    # Mesh relay
    "AitherNetRelay",
    # Chat + Mail
    "ChatRelay",
    "MailRelay",
    # MCP client + server
    "MCPAuth",
    "MCPBridge",
    "MCPServer",
    "MCPError",
    "MCPAuthError",
    "MCPBalanceError",
    # Multi-agent group chat
    "AeonSession",
    "AeonResponse",
    "AeonMessage",
    "group_chat",
    "AEON_PRESETS",
    # A2A protocol
    "A2AServer",
    # Unified storage
    "Strata",
    "StrataBackend",
    "LocalBackend",
    # Cross-platform pairing
    "PairingManager",
    "PairingResult",
    "PlatformIdentity",
    # Voice
    "VoiceClient",
    "TranscriptionResult",
    "SynthesisResult",
    "EmotionResult",
    # Cloud GPU deployment
    "CloudDeployClient",
    "get_cloud_deploy_client",
]


def connect_mcp(api_key: str = "", mcp_url: str = "https://mcp.aitherium.com"):
    """Quick access to the MCP bridge. Returns a coroutine."""
    from aithershell.mcp import connect_mcp as _connect
    return _connect(api_key=api_key, mcp_url=mcp_url)


def connect_federation(host: str = "http://localhost", tenant: str = "public"):
    """Quick access to the federation client for connecting to Elysium."""
    from aithershell.federation import FederationClient
    return FederationClient(host=host, tenant=tenant)


def auto_setup(**kwargs):
    """Quick access to agent self-setup. Returns a coroutine."""
    from aithershell.setup import auto_setup as _auto_setup
    return _auto_setup(**kwargs)


# Lazy imports for heavier modules
def __getattr__(name):
    if name == "AgentRegistry":
        from aithershell.registry import AgentRegistry
        return AgentRegistry
    if name == "AgentForge":
        from aithershell.forge import AgentForge
        return AgentForge
    if name == "FleetConfig":
        from aithershell.fleet import FleetConfig
        return FleetConfig
    if name == "ConversationStore":
        from aithershell.conversations import ConversationStore
        return ConversationStore
    if name == "LoopGuard":
        from aithershell.loop_guard import LoopGuard
        return LoopGuard
    if name == "AitherSandbox":
        from aithershell.sandbox import AitherSandbox
        return AitherSandbox
    if name == "AgentMeter":
        from aithershell.metering import AgentMeter
        return AgentMeter
    if name == "SkillManifest":
        from aithershell.identity import SkillManifest
        return SkillManifest
    if name == "ChronicleClient":
        from aithershell.chronicle import ChronicleClient
        return ChronicleClient
    if name == "WatchReporter":
        from aithershell.watch import WatchReporter
        return WatchReporter
    if name == "MetricsCollector":
        from aithershell.metrics import MetricsCollector
        return MetricsCollector
    if name == "PulseClient":
        from aithershell.pulse import PulseClient
        return PulseClient
    if name == "ServiceBridge":
        from aithershell.services import ServiceBridge
        return ServiceBridge
    if name == "EventEmitter":
        from aithershell.events import EventEmitter
        return EventEmitter
    if name == "IntakeGuard":
        from aithershell.safety import IntakeGuard
        return IntakeGuard
    if name == "ContextManager":
        from aithershell.context import ContextManager
        return ContextManager
    if name == "register_builtin_tools":
        from aithershell.builtin_tools import register_builtin_tools
        return register_builtin_tools
    if name == "GraphMemory":
        from aithershell.graph_memory import GraphMemory
        return GraphMemory
    if name == "NanoGPT":
        from aithershell.nanogpt import NanoGPT
        return NanoGPT
    if name == "NeuronPool":
        from aithershell.neurons import NeuronPool
        return NeuronPool
    if name == "AutoNeuronFire":
        from aithershell.neurons import AutoNeuronFire
        return AutoNeuronFire
    if name == "DegenerationDetector":
        from aithershell.llm.base import DegenerationDetector
        return DegenerationDetector
    if name == "strip_internal_tags":
        from aithershell.llm.base import strip_internal_tags
        return strip_internal_tags
    if name == "CATEGORY_TOOLS":
        from aithershell.neurons import CATEGORY_TOOLS
        return CATEGORY_TOOLS
    if name == "CodeGraph":
        from aithershell.faculties.code_graph import CodeGraph
        return CodeGraph
    if name == "MemoryGraph":
        from aithershell.faculties.memory_graph import MemoryGraph
        return MemoryGraph
    if name == "EmbeddingProvider":
        from aithershell.faculties.embeddings import EmbeddingProvider
        return EmbeddingProvider
    if name == "Elysium":
        from aithershell.elysium import Elysium
        return Elysium
    if name == "repowise_search":
        from aithershell.builtin_tools import repowise_search
        return repowise_search
    if name == "swarm_code":
        from aithershell.builtin_tools import swarm_code
        return swarm_code
    if name == "AitherNetRelay":
        from aithershell.relay import AitherNetRelay
        return AitherNetRelay
    if name == "ChatRelay":
        from aithershell.chat import ChatRelay
        return ChatRelay
    if name == "MailRelay":
        from aithershell.smtp import MailRelay
        return MailRelay
    if name == "MCPServer":
        from aithershell.mcp_server import MCPServer
        return MCPServer
    if name == "MCPAuth":
        from aithershell.mcp import MCPAuth
        return MCPAuth
    if name == "MCPBridge":
        from aithershell.mcp import MCPBridge
        return MCPBridge
    if name == "MCPError":
        from aithershell.mcp import MCPError
        return MCPError
    if name == "MCPAuthError":
        from aithershell.mcp import MCPAuthError
        return MCPAuthError
    if name == "MCPBalanceError":
        from aithershell.mcp import MCPBalanceError
        return MCPBalanceError
    if name == "AeonSession":
        from aithershell.aeon import AeonSession
        return AeonSession
    if name == "AeonResponse":
        from aithershell.aeon import AeonResponse
        return AeonResponse
    if name == "AeonMessage":
        from aithershell.aeon import AeonMessage
        return AeonMessage
    if name == "group_chat":
        from aithershell.aeon import group_chat
        return group_chat
    if name == "AEON_PRESETS":
        from aithershell.aeon import AEON_PRESETS
        return AEON_PRESETS
    if name == "A2AServer":
        from aithershell.a2a import A2AServer
        return A2AServer
    if name == "Strata":
        from aithershell.strata import Strata
        return Strata
    if name == "StrataBackend":
        from aithershell.strata import StrataBackend
        return StrataBackend
    if name == "LocalBackend":
        from aithershell.strata import LocalBackend
        return LocalBackend
    if name == "PairingManager":
        from aithershell.pairing import PairingManager
        return PairingManager
    if name == "PairingResult":
        from aithershell.pairing import PairingResult
        return PairingResult
    if name == "PlatformIdentity":
        from aithershell.pairing import PlatformIdentity
        return PlatformIdentity
    if name == "VoiceClient":
        from aithershell.voice import VoiceClient
        return VoiceClient
    if name == "TranscriptionResult":
        from aithershell.voice import TranscriptionResult
        return TranscriptionResult
    if name == "SynthesisResult":
        from aithershell.voice import SynthesisResult
        return SynthesisResult
    if name == "EmotionResult":
        from aithershell.voice import EmotionResult
        return EmotionResult
    if name == "CloudDeployClient":
        from aithershell.cloud_deploy import CloudDeployClient
        return CloudDeployClient
    if name == "get_cloud_deploy_client":
        from aithershell.cloud_deploy import get_cloud_deploy_client
        return get_cloud_deploy_client
    raise AttributeError(f"module 'adk' has no attribute {name!r}")
