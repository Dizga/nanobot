"""Local overrides for tools."""

from datetime import datetime
from typing import Any

from nanobot.agent.tools.filesystem import WriteFileTool, _resolve_path
from nanobot.agent.tools.cron import CronTool


class LocalWriteFileTool(WriteFileTool):
    """WriteFileTool with append mode support."""

    @property
    def parameters(self) -> dict[str, Any]:
        params = super().parameters
        params["properties"]["append"] = {
            "type": "boolean",
            "description": "If true, append to the file instead of overwriting. Defaults to false.",
        }
        return params

    async def execute(self, path: str, content: str, append: bool = False, **kwargs: Any) -> str:
        try:
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)
            action = "appended" if append else "wrote"
            return f"Successfully {action} {len(content)} bytes to {path}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {str(e)}"


class LocalCronTool(CronTool):
    """CronTool with rich job listing."""

    def _list_jobs(self) -> str:
        jobs = self._cron.list_jobs()
        if not jobs:
            return "No scheduled jobs."
        lines = []
        for j in jobs:
            parts = [f"- **{j.name}** (id: {j.id})"]
            if j.schedule.kind == "at" and j.schedule.at_ms:
                dt = datetime.fromtimestamp(j.schedule.at_ms / 1000)
                parts.append(f"  fires at: {dt.strftime('%Y-%m-%d %H:%M')}")
            elif j.schedule.kind == "every" and j.schedule.every_ms:
                secs = j.schedule.every_ms // 1000
                if secs >= 3600:
                    parts.append(f"  every: {secs // 3600}h {(secs % 3600) // 60}m")
                elif secs >= 60:
                    parts.append(f"  every: {secs // 60}m")
                else:
                    parts.append(f"  every: {secs}s")
            elif j.schedule.kind == "cron" and j.schedule.expr:
                parts.append(f"  cron: {j.schedule.expr}")
            if j.state.next_run_at_ms:
                nxt = datetime.fromtimestamp(j.state.next_run_at_ms / 1000)
                parts.append(f"  next run: {nxt.strftime('%Y-%m-%d %H:%M')}")
            if j.state.last_run_at_ms:
                last = datetime.fromtimestamp(j.state.last_run_at_ms / 1000)
                status = j.state.last_status or "unknown"
                parts.append(f"  last run: {last.strftime('%Y-%m-%d %H:%M')} ({status})")
            parts.append(f"  message: {j.payload.message}")
            lines.append("\n".join(parts))
        return "Scheduled jobs:\n" + "\n".join(lines)
