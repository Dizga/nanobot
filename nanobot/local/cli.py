"""Local CLI entry point — monkey-patches upstream classes and replaces gateway command.

This module is the fork's entry point (see pyproject.toml). It:
1. Patches upstream modules so any code that imports them gets local versions
2. Replaces the gateway command with local heartbeat/cron wiring
3. Re-exports the typer app
"""

import asyncio

import typer
from loguru import logger

# ---------------------------------------------------------------------------
# 1. Monkey-patch upstream modules with local subclasses
# ---------------------------------------------------------------------------

import nanobot.agent.loop
import nanobot.cron.service
import nanobot.channels.discord

from nanobot.local.agent import LocalAgentLoop
from nanobot.local.cron import LocalCronService
from nanobot.local.discord import LocalDiscordChannel

nanobot.agent.loop.AgentLoop = LocalAgentLoop
nanobot.cron.service.CronService = LocalCronService
nanobot.channels.discord.DiscordChannel = LocalDiscordChannel

# Wrap _make_provider to add extra_body support
import nanobot.cli.commands as _commands

_upstream_make_provider = _commands._make_provider


def _make_provider_with_extra_body(config):
    """Wrap upstream _make_provider to pass extra_body to LiteLLMProvider."""
    provider = _upstream_make_provider(config)
    # Patch extra_body onto LiteLLM providers that support it
    model = config.agents.defaults.model
    p = config.get_provider(model)
    if p and getattr(p, "extra_body", None) and hasattr(provider, "extra_body"):
        provider.extra_body = p.extra_body
    return provider


_commands._make_provider = _make_provider_with_extra_body

# ---------------------------------------------------------------------------
# 2. Import the app (all upstream commands are already registered)
# ---------------------------------------------------------------------------

from nanobot.cli.commands import app, console, _make_provider
from nanobot import __logo__

# ---------------------------------------------------------------------------
# 3. Replace the gateway command with local wiring
# ---------------------------------------------------------------------------


@app.command()
def gateway(
    port: int = typer.Option(18790, "--port", "-p", help="Gateway port"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Start the nanobot gateway."""
    from nanobot.config.loader import load_config, get_data_dir
    from nanobot.bus.queue import MessageBus
    from nanobot.agent.loop import AgentLoop  # monkey-patched to LocalAgentLoop
    from nanobot.channels.manager import ChannelManager
    from nanobot.session.manager import SessionManager
    from nanobot.cron.service import CronService  # monkey-patched to LocalCronService
    from nanobot.cron.types import CronJob
    from nanobot.local.heartbeat import LocalHeartbeatService

    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    console.print(f"{__logo__} Starting nanobot gateway on port {port}...")

    config = load_config()
    bus = MessageBus()
    provider = _make_provider(config)
    session_manager = SessionManager(config.workspace_path)

    # Cron
    cron_store_path = get_data_dir() / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    # Agent
    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=session_manager,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
    )

    # Cron callback
    async def on_cron_job(job: CronJob) -> str | None:
        response = await agent.process_direct(
            job.payload.message,
            session_key=f"cron:{job.id}",
            channel=job.payload.channel or "cli",
            chat_id=job.payload.to or "direct",
        )
        if job.payload.deliver and job.payload.to:
            from nanobot.bus.events import OutboundMessage
            await bus.publish_outbound(OutboundMessage(
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to,
                content=response or "",
            ))
        return response
    cron.on_job = on_cron_job

    # Heartbeat — route to primary Discord DM session
    hb_channel = "cli"
    hb_chat_id = "direct"
    if config.channels.discord.enabled and config.channels.discord.allow_from:
        hb_channel = "discord"
        hb_chat_id = config.channels.discord.allow_from[0]

    hb_cfg = config.gateway.heartbeat
    heartbeat = LocalHeartbeatService(
        workspace=config.workspace_path,
        agent_loop=agent,
        hb_channel=hb_channel,
        hb_chat_id=hb_chat_id,
        interval_s=hb_cfg.interval_s,
        enabled=hb_cfg.enabled,
    )

    # Channel manager
    channels = ChannelManager(config, bus)

    if channels.enabled_channels:
        console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]Warning: No channels enabled[/yellow]")

    cron_status = cron.status()
    if cron_status["jobs"] > 0:
        console.print(f"[green]✓[/green] Cron: {cron_status['jobs']} scheduled jobs")

    console.print(f"[green]✓[/green] Heartbeat: every {heartbeat.interval_s // 60}m")

    async def run():
        try:
            await cron.start()
            await heartbeat.start()
            await asyncio.gather(
                agent.run(),
                channels.start_all(),
            )
        except KeyboardInterrupt:
            console.print("\nShutting down...")
        finally:
            await agent.close_mcp()
            heartbeat.stop()
            cron.stop()
            agent.stop()
            await channels.stop_all()

    asyncio.run(run())
