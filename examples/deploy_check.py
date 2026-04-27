"""Example Python Plugin — drop this in ~/.aither/plugins/"""

from aithershell.plugins import SlashCommand


class DeployCheck(SlashCommand):
    name = "deploy-check"
    description = "Check deployment status of all services"
    aliases = ["dc"]

    async def run(self, args, ctx):
        import httpx
        url = f"{ctx['config'].url}/deploy/status"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url)
                if r.status_code == 200:
                    data = r.json()
                    lines = [f"Deployment Status: {data.get('status', 'unknown')}"]
                    for svc in data.get("services", [])[:10]:
                        lines.append(f"  {svc['name']:20s} {svc.get('status', '?')}")
                    return "\n".join(lines)
                return f"Error: {r.status_code}"
        except Exception as e:
            return f"Error: {e}"
