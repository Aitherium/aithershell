# Developer Guide: Building Agents for AitherShell

> Learn how to extend AitherShell with custom agents, plugins, and skills using the **Aither** Agent Framework.

## Overview

**AitherShell** is the end-user interface for the Aitherium platform. Developers extend it by building **agents** using the **Aither** framework.

### Two Repositories

| Repository | Purpose | Audience | Location |
|---|---|---|---|
| **[aitherium/aither](https://github.com/aitherium/aither)** | Agent Framework Dev Kit | Developers | https://github.com/aitherium/aither |
| **[aitherium/aithershell](https://github.com/aitherium/aithershell)** | End-user CLI shell | Users & Operators | https://github.com/aitherium/aithershell |

### The Workflow

```
Developer
    ↓
1. Uses Aither framework (aitherium/aither)
   - Writes agent logic
   - Implements skills
   - Builds plugins
    ↓
2. Registers agent with AitherShell or Aitherium platform
    ↓
3. End User
   - Runs `aither prompt "query"`
   - Agent executes behind the scenes
   - Results streamed to terminal
```

## Setting Up Aither Development

### 1. Clone Aither Repository

```bash
git clone https://github.com/aitherium/aither.git
cd aither
```

### 2. Install in Development Mode

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install Aither with development dependencies
pip install -e ".[dev]"

# Install AitherShell to test integration
pip install aithershell
```

### 3. Understand Aither Structure

```
aither/
├── aither/
│   ├── agent.py          # Base Agent class
│   ├── skill.py          # Skill decorator & framework
│   ├── plugin.py         # Plugin system
│   ├── capability.py     # Capability model (security)
│   └── ...
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
from aither import Agent, Capability, Skill

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
# After your agent is built and tested:
aither register my_agent.py --name "my-agent"
```

Or programmatically:

```python
from aithershell import agent_registry

agent_registry.register(MyAgent)
```

### Step 3: Test with AitherShell

```bash
# Query the agent via shell
aither prompt "What patterns are in this code?" --agent my-agent

# Or interactive mode
aither shell
> /use my-agent
> analyze "def hello(): pass"
```

## Agent Anatomy

### 1. Capabilities (Security Model)

Capabilities control what your agent can access:

```python
from aither import Capability

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
from aither import Skill

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

1. **Read Aither Docs** — Full API at https://github.com/aitherium/aither/docs
2. **Explore Examples** — Study agents at https://github.com/aitherium/aither/examples
3. **Join Community** — Discuss at https://github.com/aitherium/aither/discussions
4. **Build & Share** — Create your agent and submit to registry

## Support

- **Questions:** Open issue on [aitherium/aither](https://github.com/aitherium/aither)
- **Bugs:** Report on agent repo (or Aither if framework issue)
- **Ideas:** Discuss in [Aither discussions](https://github.com/aitherium/aither/discussions)

---

**Get started:** Clone [aitherium/aither](https://github.com/aitherium/aither) and build your first agent! 🚀
