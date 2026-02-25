"""Local overrides for HeartbeatService.

Upstream's heartbeat uses a two-phase design: a cheap LLM call decides
skip/run, then the full agent loop executes.  Our HEARTBEAT.md contains
*instructions* (check Todoist, review cron, decide whether to nudge the
user), so the full agent loop always runs.  Gating is via active hours
(8am-11pm) and a no-user-activity check.

We don't call super().__init__ because upstream now requires `provider`
and `model` for its Phase-1 LLM call, which we skip entirely.
"""

import asyncio
from datetime import datetime
from pathlib import Path

from loguru import logger


def _is_heartbeat_empty(content: str | None) -> bool:
    """Check if HEARTBEAT.md has no actionable content."""
    if not content:
        return True
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("<!--"):
            continue
        return False
    return True


class LocalHeartbeatService:
    """Heartbeat that always runs the full agent loop with active-hours gating.

    Does NOT inherit from upstream HeartbeatService (whose __init__ signature
    changed to require provider/model for a Phase-1 LLM call we don't use).
    Implements the same public interface: start(), stop(), interval_s, enabled.
    """

    _PROMPT = "Read HEARTBEAT.md in your workspace carefully. Follow every step listed there — do not skip any."

    def __init__(self, *, workspace: Path, agent_loop, hb_channel="cli", hb_chat_id="direct",
                 interval_s=60 * 60, enabled=True):
        self.workspace = workspace
        self.interval_s = interval_s
        self.enabled = enabled
        self._agent = agent_loop
        self._hb_channel = hb_channel
        self._hb_chat_id = hb_chat_id
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def heartbeat_file(self) -> Path:
        return self.workspace / "HEARTBEAT.md"

    def _read_heartbeat_file(self) -> str | None:
        if self.heartbeat_file.exists():
            try:
                return self.heartbeat_file.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    async def start(self) -> None:
        if not self.enabled:
            logger.info("Heartbeat disabled")
            return
        if self._running:
            logger.warning("Heartbeat already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Heartbeat started (every {}s)", self.interval_s)

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error: {}", e)

    async def _tick(self) -> None:
        # Active hours: 8am-11pm
        hour = datetime.now().hour
        if hour < 8 or hour >= 23:
            logger.debug("Heartbeat: skipping (outside active hours, {}:00)", hour)
            return

        content = self._read_heartbeat_file()
        if _is_heartbeat_empty(content):
            logger.debug("Heartbeat: no tasks (HEARTBEAT.md empty)")
            return

        if not self._agent:
            logger.warning("Heartbeat: no agent_loop configured")
            return

        # Skip if no user activity since last heartbeat action
        session_key = f"{self._hb_channel}:{self._hb_chat_id}"
        session = self._agent.sessions.get_or_create(session_key)
        if session.messages and session.messages[-1].get("source") == "message_tool":
            logger.debug("Heartbeat: skipping (no user activity since last run)")
            return

        logger.info("Heartbeat: checking for tasks...")
        try:
            await self._agent.process_direct(
                self._PROMPT,
                channel=self._hb_channel,
                chat_id=self._hb_chat_id,
                session_key=session_key,
                save_session=False,
            )
            logger.info("Heartbeat: completed")
        except Exception as e:
            logger.error("Heartbeat execution failed: {}", e)
