"""Local overrides for AgentLoop — surgical consolidation, tool call persistence, save_session control."""

import asyncio
import json
from datetime import datetime
from typing import Awaitable, Callable

from loguru import logger

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.filesystem import ReadFileTool, EditFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.session.manager import Session

from nanobot.local.context import LocalContextBuilder
from nanobot.local.tools import LocalWriteFileTool, LocalCronTool


class LocalAgentLoop(AgentLoop):
    """AgentLoop with surgical consolidation, tool call persistence, and save_session control."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace context builder with local version
        self.context = LocalContextBuilder(self.workspace)

    def _register_default_tools(self) -> None:
        super()._register_default_tools()
        # Replace write_file with append-capable version
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(LocalWriteFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        # Replace cron with rich-listing version
        if self.cron_service:
            self.tools.register(LocalCronTool(self.cron_service))

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict], bool, str]:
        """Run the agent loop with tool call tracking for session persistence.

        Returns (final_content, tools_used, messages, used_message_tool, sent_message_content).
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        used_message_tool = False
        sent_message_content = ""

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.has_tool_calls:
                # Stream progress
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    if tool_call.name == "message":
                        used_message_tool = True
                        sent_message_content = tool_call.arguments.get("content", "")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result,
                    )

            else:
                final_content = self._strip_think(response.content)
                if final_content:
                    messages.append({"role": "assistant", "content": final_content})
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )
            messages.append({"role": "assistant", "content": final_content})

        return final_content, tools_used, messages, used_message_tool, sent_message_content

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        save_session: bool = True,
    ) -> OutboundMessage | None:
        """Process message with tool call persistence, consolidation locks, and save_session control."""
        # System messages
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=self.memory_window)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            final_content, _, all_msgs, _, _ = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            lock = self._get_consolidation_lock(session.key)
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated:]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)
                self._prune_consolidation_lock(session.key, lock)

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/help — Show available commands")

        # Background consolidation — trigger by unconsolidated count
        unconsolidated = len(session.messages) - session.last_consolidated
        if unconsolidated >= self.memory_window and session.key not in self._consolidating:
            self._consolidating.add(session.key)
            lock = self._get_consolidation_lock(session.key)

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        await self._consolidate_memory(session)
                finally:
                    self._consolidating.discard(session.key)
                    self._prune_consolidation_lock(session.key, lock)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=self.memory_window)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        # Progress callback
        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, tools_used, all_msgs, used_message_tool, sent_message_content = \
            await self._run_agent_loop(initial_messages, on_progress=on_progress or _bus_progress)

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        # Save to session — for background tasks (save_session=False), only save
        # the actual message sent to the user, not the internal heartbeat prompt.
        if save_session:
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
        elif used_message_tool:
            session.add_message("assistant", sent_message_content, source="message_tool")
            self.sessions.save(session)

        # Suppress duplicate if message tool already sent
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
                return None

        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """Surgical multi-turn consolidation using restricted file tools. Returns True on success."""
        if archive_all:
            old_messages = [m for m in session.messages if m["role"] in ("user", "assistant")]
            keep_count = 0
            logger.info("Memory consolidation (archive_all): {} messages archived", len(old_messages))
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                return True
            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                return True
            old_messages = [m for m in session.messages[session.last_consolidated:-keep_count]
                           if m["role"] in ("user", "assistant")]
            if len(old_messages) < keep_count:
                return True
            logger.info("Memory consolidation: {} total, {} to consolidate, {} keep",
                        len(session.messages), len(old_messages), keep_count)

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        conversation = "\n".join(lines)

        # Restricted tools: only file ops scoped to memory directory
        memory_dir = self.workspace / "memory"
        memory_tools = ToolRegistry()
        memory_tools.register(ReadFileTool(workspace=memory_dir, allowed_dir=memory_dir))
        memory_tools.register(LocalWriteFileTool(workspace=memory_dir, allowed_dir=memory_dir))
        memory_tools.register(EditFileTool(workspace=memory_dir, allowed_dir=memory_dir))

        memory_path = str(memory_dir / "MEMORY.md")
        history_path = str(memory_dir / "HISTORY.md")

        prompt = f"""You are a memory consolidation agent. Review the conversation below and update the memory files using your tools.

## Your tasks

1. **Update MEMORY.md** ({memory_path}): Read it first, then make surgical edits to add new facts about the user (personal info, preferences, habits). Use edit_file for targeted changes. Only update if there's something new worth remembering. Remove facts that are clearly outdated. Don't add temporary or time-sensitive items. Don't add projects, tasks, or goals — Todoist is the source of truth for those.

2. **Append to HISTORY.md** ({history_path}): Add a summary entry (2-5 sentences) starting with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep later. Use write_file with append=true.

If there's nothing worth remembering, just say so and stop.

## Conversation to Process
{conversation}"""

        try:
            messages = [
                {"role": "system", "content": "You are a memory consolidation agent. Use your tools to update memory files. Be concise and surgical."},
                {"role": "user", "content": prompt},
            ]

            for _ in range(10):
                response = await self.provider.chat(
                    messages=messages,
                    tools=memory_tools.get_definitions(),
                    model=self.model,
                )
                if not response.has_tool_calls:
                    break

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": tool_call_dicts,
                })

                for tc in response.tool_calls:
                    logger.debug("Consolidation tool: {}({})", tc.name,
                                 json.dumps(tc.arguments, ensure_ascii=False)[:200])
                    result = await memory_tools.execute(tc.name, tc.arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": result,
                    })

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info("Memory consolidation done: {} messages, last_consolidated={}",
                        len(session.messages), session.last_consolidated)
            return True
        except Exception as e:
            logger.error("Memory consolidation failed: {}", e)
            return False

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        save_session: bool = True,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly with save_session control."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(
            msg, session_key=session_key, on_progress=on_progress, save_session=save_session,
        )
        return response.content if response else ""
