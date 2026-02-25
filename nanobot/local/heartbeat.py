"""Local overrides for HeartbeatService."""

from datetime import datetime

from loguru import logger

from nanobot.heartbeat.service import HeartbeatService, _is_heartbeat_empty, HEARTBEAT_OK_TOKEN


class LocalHeartbeatService(HeartbeatService):
    """HeartbeatService with active hours, stricter prompt, activity guard, and session routing.

    Upstream's heartbeat is designed for static task lists. Our usage is different:
    HEARTBEAT.md contains instructions (check Todoist, review cron, decide whether
    to nudge the user), so the full agent loop runs every tick. Gating is via
    active hours and a no-user-activity check.
    """

    _PROMPT = "Read HEARTBEAT.md in your workspace carefully. Follow every step listed there — do not skip any."

    def __init__(self, *args, agent_loop=None, hb_channel="cli", hb_chat_id="direct", **kwargs):
        super().__init__(*args, **kwargs)
        self._agent = agent_loop
        self._hb_channel = hb_channel
        self._hb_chat_id = hb_chat_id

    async def _tick(self) -> None:
        # Active hours: 8am–11pm
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
            response = await self._agent.process_direct(
                self._PROMPT,
                channel=self._hb_channel,
                chat_id=self._hb_chat_id,
                session_key=session_key,
                save_session=False,
            )
            if HEARTBEAT_OK_TOKEN.replace("_", "") in response.upper().replace("_", ""):
                logger.info("Heartbeat: OK (no action needed)")
            else:
                logger.info("Heartbeat: completed task")
        except Exception as e:
            logger.error("Heartbeat execution failed: {}", e)
