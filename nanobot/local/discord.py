"""Local overrides for Discord channel."""

from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.discord import DiscordChannel, DISCORD_API_BASE
from nanobot.config.schema import DiscordConfig


class LocalDiscordChannel(DiscordChannel):
    """Discord channel with DM resolution and empty message guard."""

    def __init__(self, config: DiscordConfig, bus: MessageBus):
        super().__init__(config, bus)
        self._dm_channels: dict[str, str] = {}  # user_id -> dm_channel_id

    async def _get_dm_channel(self, user_id: str) -> str | None:
        """Open or retrieve a DM channel for a user ID."""
        if user_id in self._dm_channels:
            return self._dm_channels[user_id]
        if not self._http:
            return None
        headers = {"Authorization": f"Bot {self.config.token}"}
        try:
            resp = await self._http.post(
                f"{DISCORD_API_BASE}/users/@me/channels",
                headers=headers,
                json={"recipient_id": user_id},
            )
            resp.raise_for_status()
            dm_channel_id = resp.json()["id"]
            self._dm_channels[user_id] = dm_channel_id
            return dm_channel_id
        except Exception as e:
            logger.error("Failed to open DM channel for {}: {}", user_id, e)
            return None

    async def _resolve_channel_id(self, chat_id: str) -> str | None:
        """Resolve a chat_id to a valid Discord channel ID.

        Tries the chat_id as a channel first; falls back to opening a DM.
        """
        headers = {"Authorization": f"Bot {self.config.token}"}
        try:
            resp = await self._http.get(
                f"{DISCORD_API_BASE}/channels/{chat_id}",
                headers=headers,
            )
            if resp.status_code == 200:
                return chat_id
        except Exception:
            pass
        return await self._get_dm_channel(chat_id)

    async def send(self, msg: OutboundMessage) -> None:
        """Send with DM channel resolution, empty message guard, and progress filtering."""
        if not self._http:
            logger.warning("Discord HTTP client not initialized")
            return
        if not msg.content or not msg.content.strip():
            logger.info("Skipping empty message to {}", msg.chat_id)
            return
        if msg.metadata and msg.metadata.get("_progress"):
            logger.debug("Skipping progress message to Discord: {}", msg.content[:80])
            return
        channel_id = await self._resolve_channel_id(msg.chat_id)
        if not channel_id:
            logger.error("Could not resolve Discord channel for {}", msg.chat_id)
            return
        # Delegate to base class with the resolved channel_id
        resolved_msg = OutboundMessage(
            channel=msg.channel,
            chat_id=channel_id,
            content=msg.content,
            reply_to=msg.reply_to,
            media=msg.media,
            metadata=msg.metadata,
        )
        await super().send(resolved_msg)

    async def _handle_message_create(self, payload: dict[str, Any]) -> None:
        """Handle incoming Discord messages with DM session key logic."""
        author = payload.get("author") or {}
        if author.get("bot"):
            return

        sender_id = str(author.get("id", ""))
        channel_id = str(payload.get("channel_id", ""))
        content = payload.get("content") or ""

        if not sender_id or not channel_id:
            return
        if not self.is_allowed(sender_id):
            return

        # For DMs (no guild_id), use sender_id as chat_id for stable session keys.
        # For server channels, keep channel_id so multiple users share one session.
        is_dm = not payload.get("guild_id")
        effective_chat_id = sender_id if is_dm else channel_id

        content_parts = [content] if content else []
        media_paths: list[str] = []
        media_dir = Path.home() / ".nanobot" / "media"

        for attachment in payload.get("attachments") or []:
            url = attachment.get("url")
            filename = attachment.get("filename") or "attachment"
            size = attachment.get("size") or 0
            if not url or not self._http:
                continue
            from nanobot.channels.discord import MAX_ATTACHMENT_BYTES
            if size and size > MAX_ATTACHMENT_BYTES:
                content_parts.append(f"[attachment: {filename} - too large]")
                continue
            try:
                media_dir.mkdir(parents=True, exist_ok=True)
                file_path = media_dir / f"{attachment.get('id', 'file')}_{filename.replace('/', '_')}"
                resp = await self._http.get(url)
                resp.raise_for_status()
                file_path.write_bytes(resp.content)
                media_paths.append(str(file_path))
                content_parts.append(f"[attachment: {file_path}]")
            except Exception as e:
                logger.warning("Failed to download Discord attachment: {}", e)
                content_parts.append(f"[attachment: {filename} - download failed]")

        reply_to = (payload.get("referenced_message") or {}).get("id")

        await self._start_typing(channel_id)

        await self._handle_message(
            sender_id=sender_id,
            chat_id=effective_chat_id,
            content="\n".join(p for p in content_parts if p) or "[empty message]",
            media=media_paths,
            metadata={
                "message_id": str(payload.get("id", "")),
                "guild_id": payload.get("guild_id"),
                "reply_to": reply_to,
                "discord_channel_id": channel_id,
            },
        )
