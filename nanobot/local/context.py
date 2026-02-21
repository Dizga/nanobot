"""Local overrides for ContextBuilder."""

import platform

from nanobot.agent.context import ContextBuilder


class LocalContextBuilder(ContextBuilder):
    """Customized context builder with trimmed system prompt."""

    def _get_identity(self) -> str:
        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return f"""# nanobot

## Current Time
{now} ({tz})

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable)
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

## Communication

Always be helpful, accurate, and concise. When using tools, think step by step: what you know, what you need, and why you chose this tool.
When you learn something new about the user, update {workspace_path}/memory/MEMORY.md using edit_file.
To recall past events, grep {workspace_path}/memory/HISTORY.md"""

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        parts = []

        parts.append(self._get_identity())

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        # Show skills summary
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

To use a skill, read its SKILL.md file using the read_file tool.

{skills_summary}""")

        return "\n\n---\n\n".join(parts)
