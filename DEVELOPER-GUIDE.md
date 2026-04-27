# Developer Guide: Building Agents for AitherShell

> Learn how to extend AitherShell with custom agents, plugins, and skills using the **AitherSDK** Agent Framework.

## Overview

**AitherShell** is the end-user interface for the Aitherium platform. Developers extend it by building **agents** using the **AitherSDK** framework.

### Two Repositories, Two CLIs

| Repository | Purpose | Audience | CLI Commands | Location |
|---|---|---|---|---|
| **[aitherium/aithersdk](https://github.com/aitherium/aithersdk)** | Agent Framework Dev Kit | Developers | `aither new-agent`, `aither test`, `aither run` | https://github.com/aitherium/aithersdk |
| **[aitherium/aithershell](https://github.com/aitherium/aithershell)** | End-user CLI shell | Users & Operators | `aither shell`, `aither prompt` | https://github.com/aitherium/aithershell |

**Note:** Both use the `aither` command, but serve different purposes. AitherSDK is for development; AitherShell is for production.

### The Workflow

```
Developer
    ↓
1. Uses AitherSDK framework (aitherium/aithersdk)
   - Writes agent logic with `aither new-agent`
   - Implements skills and plugins
   - Tests locally with `aither test` and `aither run`
   - Builds and validates with developer CLI
    ↓
2. Registers agent with AitherShell or Aitherium platform
   - `aither register` (from AitherSDK)
    ↓
3. End User
   - Runs `aither prompt "query"` (from AitherShell)
   - Agent executes behind the scenes
   - Results streamed to terminal
```

## Setting Up AitherSDK Development

### 1. Clone AitherSDK Repository

```bash
git clone https://github.com/aitherium/aithersdk.git
cd aithersdk
```

### 2. Install in Development Mode

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install AitherSDK with development dependencies
pip install -e ".[dev]"

# Install AitherShell to test integration
pip install aithershell
```

### 3. Understand AitherSDK Structure

```
aithersdk/
├── aithersdk/
│   ├── agent.py          # Base Agent class
│   ├── skill.py          # Skill decorator & framework
│   ├── plugin.py         # Plugin system
│   ├── capability.py     # Capability model (security)
│   └── ...
├── cli/                  # Developer CLI (aither command)
├── examples/             # Example agents
│   ├── research_agent.py
│   ├── code_analyzer.py
│   └── ...
├── tests/
└── docs/
```

## Building Your First Agent

### Step 1: Create Agent Class

```python
# my_agent.py
from aithersdk import Agent, Capability, Skill

class MyAgent(Agent):
    """Description of what your agent does."""
    
    # Metadata
    name = "my-agent"
    description = "What this agent does and when to use it"
    version = "1.0.0"
    author = "Your Name"
    
    # Declare required capabilities (what the agent can access)
    capabilities = [
        Capability("code.search", read_only=True),
        Capability("memory.read", read_only=True),
        Capability("memory.write", read_write=True),
    ]
    
    # Define skills (what the agent can do)
    @Skill(
        name="analyze",
        description="Analyze code for patterns",
        inputs={"code": str, "focus_area": str},
        outputs={"analysis": dict, "confidence": float}
    )
    async def analyze(self, code: str, focus_area: str) -> dict:
        """Core agent logic here."""
        analysis = {}
        # Your implementation
        return analysis
```

### Step 2: Register with AitherShell

```bash
# After your agent is built and tested with AitherSDK:
aither register my_agent.py --name "my-agent"
```

Or programmatically:

```python
from aithershell import agent_registry

agent_registry.register(MyAgent)
```

### Step 3: Test with AitherShell

```bash
# Query the agent via AitherShell
aither prompt "analyze 'def hello(): pass'" --agent my-agent

# Or interactive mode
aither shell
> /use my-agent
> analyze "def hello(): pass"
```

## AitherSDK vs AitherShell CLI

Both products use the `aither` command, but for different purposes:

**AitherSDK CLI** (for developers - build and test):
```bash
aither new-agent my-agent          # Create agent from template
aither test                        # Run unit tests locally
aither run my_agent.py             # Execute agent locally
aither debug my_agent.py           # Interactive debugging
aither validate my_agent.py        # Validate schema
aither register my_agent.py        # Register with platform
aither publish                     # Publish to registry
```

**AitherShell CLI** (for end users - run and query):
```bash
aither shell                       # Interactive shell
aither prompt "query"              # One-off query
aither prompt "query" --agent xyz  # Use specific agent
aither plugins list                # List available agents
aither config                      # Configure shell
aither export                      # Export results
```

## Agent Anatomy

### 1. Capabilities (Security Model)

Capabilities control what your agent can access:

```python
from aithersdk import Capability

capabilities = [
    # Read-only capabilities
    Capability("code.search", read_only=True),      # Search codebase
    Capability("web.search", read_only=True),       # Search web
    Capability("memory.read", read_only=True),      # Read from memory
    
    # Read-write capabilities
    Capability("memory.write", read_write=True),    # Write to memory
    Capability("github.issues.create", read_write=True),  # Create issues
]
```

**Important:** Declare all capabilities upfront. Missing capabilities will cause runtime errors.

### 2. Skills (Agent Functions)

Skills are the agent's actions:

```python
from aithersdk import Skill

@Skill(
    name="investigate",
    description="Investigate a topic and synthesize findings",
    inputs={
        "query": str,
        "depth": int,  # 1-5
    },
    outputs={
        "findings": str,
        "sources": list,
        "confidence": float,
    }
)
async def investigate(self, query: str, depth: int = 3) -> dict:
    """Investigate a topic."""
    # Use capabilities
    code_results = await self.use_capability("code.search", query)
    web_results = await self.use_capability("web.search", query)
    
    # Process and synthesize
    findings = synthesize(code_results, web_results, depth)
    
    # Write to memory
    await self.use_capability("memory.write", {
        "query": query,
        "findings": findings,
        "timestamp": datetime.now().isoformat(),
    })
    
    return {
        "findings": findings,
        "sources": [code_results, web_results],
        "confidence": 0.85,
    }
```

### 3. Lifecycle Hooks

Control agent behavior at key points:

```python
class MyAgent(Agent):
    
    async def on_init(self):
        """Called when agent is first loaded."""
        print("Agent initializing...")
    
    async def on_setup(self):
        """Called before handling first task."""
        await self.use_capability("memory.read")  # Warm up cache
    
    async def on_task(self, task):
        """Called before each task."""
        print(f"Task: {task.skill_name}")
    
    async def on_complete(self, result):
        """Called after task completes."""
        print(f"Result: {result}")
    
    async def on_cleanup(self):
        """Called when agent is unloaded."""
        print("Agent shutting down...")
```

## Common Patterns

### Pattern 1: Research Agent

Investigates topics using multiple sources:

```python
class ResearchAgent(Agent):
    name = "research"
    capabilities = [
        Capability("code.search", read_only=True),
        Capability("web.search", read_only=True),
        Capability("memory.write", read_write=True),
    ]
    
    @Skill(name="investigate")
    async def investigate(self, query: str) -> dict:
        code = await self.use_capability("code.search", query)
        web = await self.use_capability("web.search", query)
        synthesis = await self.synthesize(code, web)
        await self.use_capability("memory.write", synthesis)
        return synthesis
```

### Pattern 2: Code Analysis Agent

Analyzes code patterns and risks:

```python
class CodeAnalyzerAgent(Agent):
    name = "code-analyzer"
    capabilities = [
        Capability("code.search", read_only=True),
        Capability("memory.read", read_only=True),
    ]
    
    @Skill(name="analyze")
    async def analyze(self, path: str) -> dict:
        code = await self.use_capability("code.search", path)
        patterns = self.detect_patterns(code)
        risks = self.assess_risks(code)
        return {
            "patterns": patterns,
            "risks": risks,
            "recommendation": self.recommend_fixes(risks),
        }
```

### Pattern 3: Orchestration Agent

Coordinates multiple agents:

```python
class OrchestratorAgent(Agent):
    name = "orchestrator"
    capabilities = [
        Capability("agent.dispatch", read_write=True),
        Capability("memory.write", read_write=True),
    ]
    
    @Skill(name="coordinate")
    async def coordinate(self, objective: str) -> dict:
        # Dispatch to other agents
        research = await self.dispatch_agent("research", "investigate", query=objective)
        analysis = await self.dispatch_agent("code-analyzer", "analyze", path=research["path"])
        
        # Synthesize results
        plan = self.create_plan(research, analysis)
        await self.use_capability("memory.write", plan)
        
        return plan
```

## Testing Your Agent

### Unit Tests

```python
# test_my_agent.py
import pytest
from aither.testing import AgentTestHarness
from my_agent import MyAgent

@pytest.fixture
def agent():
    return MyAgent()

@pytest.fixture
def harness(agent):
    return AgentTestHarness(agent)

def test_analyze_skill(harness):
    result = harness.execute_skill("analyze", code="def hello(): pass")
    assert result["analysis"] is not None
    assert result["confidence"] > 0.5

def test_capability_access(harness):
    # Verify agent can access required capabilities
    assert harness.has_capability("code.search")
    assert harness.has_capability("memory.read")
```

### Integration Tests

```python
def test_aithershell_integration():
    """Test agent works via AitherShell."""
    from aithershell import shell
    
    result = shell.query(
        "analyze 'def hello(): pass'",
        agent="my-agent"
    )
    assert result.success
    assert "analysis" in result.data
```

### Run Tests

```bash
pytest tests/ -v

# With coverage
pytest tests/ --cov=aither --cov-report=html
```

## Publishing Your Agent

### Option 1: Register Locally

For personal use or testing:

```bash
aither register ./my_agent.py
aither plugins list
```

### Option 2: Submit to Aitherium Registry

For community sharing:

1. Push to GitHub
2. Open PR on [aitherium/agents](https://github.com/aitherium/agents)
3. Aitherium team reviews and tests
4. Merged agents appear in `aither plugins search`

### Option 3: Create Agent Package

For distribution:

```bash
# Create setup.py
cat > setup.py << 'EOF'
from setuptools import setup

setup(
    name="my-aither-agent",
    version="1.0.0",
    packages=["my_agent"],
    install_requires=["aither>=1.0.0"],
    entry_points={
        "aither.agents": [
            "my-agent = my_agent:MyAgent",
        ],
    },
)
EOF

# Build and publish
python -m build
twine upload dist/
```

Then users can:

```bash
pip install my-aither-agent
aither prompt "query" --agent my-agent
```

## Debugging

### Enable Verbose Logging

```bash
aither shell --debug
aither prompt "query" --agent my-agent --debug
```

### Inspect Agent State

```python
# Inside your agent
import logging
logger = logging.getLogger("aither")

logger.debug(f"Capability access: {self.capabilities}")
logger.debug(f"Memory state: {await self.use_capability('memory.read')}")
```

### Profile Performance

```bash
aither profile --agent my-agent --skill analyze --input "code.py"
```

## Next Steps

1. **Read AitherSDK Docs** — Full API at https://github.com/aitherium/aithersdk/docs
2. **Explore Examples** — Study agents at https://github.com/aitherium/aithersdk/examples
3. **Join Community** — Discuss at https://github.com/aitherium/aithersdk/discussions
4. **Build & Share** — Create your agent and submit to registry

## Support

- **Questions:** Open issue on [aitherium/aithersdk](https://github.com/aitherium/aithersdk)
- **Bugs:** Report on agent repo (or AitherSDK if framework issue)
- **Ideas:** Discuss in [AitherSDK discussions](https://github.com/aitherium/aithersdk/discussions)

---

**Get started:** Clone [aitherium/aithersdk](https://github.com/aitherium/aithersdk) and build your first agent! 🚀
