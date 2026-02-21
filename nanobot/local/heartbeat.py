"""Local overrides for HeartbeatService."""

from datetime import datetime

from loguru import logger

from nanobot.heartbeat.service import HeartbeatService


class LocalHeartbeatService(HeartbeatService):
    """HeartbeatService with active hours check (8am-11pm)."""

    async def _tick(self) -> None:
        hour = datetime.now().hour
        if hour < 8 or hour >= 23:
            logger.debug("Heartbeat: skipping (outside active hours, {}:00)", hour)
            return
        await super()._tick()
