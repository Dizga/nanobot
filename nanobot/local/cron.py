"""Local overrides for CronService."""

from loguru import logger

from nanobot.cron.service import CronService, _compute_next_run, _now_ms
from nanobot.cron.types import CronJob


class LocalCronService(CronService):
    """CronService that prevents re-entry during job execution.

    Upstream updates next_run_at_ms AFTER the on_job callback returns.
    If the callback triggers add_job() → _arm_timer() → _on_timer(),
    the executing job still looks "due" and re-executes in a loop.

    Fix: advance next_run_at_ms BEFORE invoking the callback.
    """

    async def _execute_job(self, job: CronJob) -> None:
        start_ms = _now_ms()
        logger.info("Cron: executing job '{}' ({})", job.name, job.id)

        # Advance next_run_at_ms before callback to prevent re-entry
        if job.schedule.kind == "at":
            job.state.next_run_at_ms = None
        else:
            job.state.next_run_at_ms = _compute_next_run(job.schedule, start_ms)

        try:
            response = None
            if self.on_job:
                response = await self.on_job(job)

            job.state.last_status = "ok"
            job.state.last_error = None
            logger.info("Cron: job '{}' completed", job.name)

        except Exception as e:
            job.state.last_status = "error"
            job.state.last_error = str(e)
            logger.error("Cron: job '{}' failed: {}", job.name, e)

        job.state.last_run_at_ms = start_ms
        job.updated_at_ms = _now_ms()

        # Handle one-shot jobs
        if job.schedule.kind == "at":
            if job.delete_after_run:
                self._store.jobs = [j for j in self._store.jobs if j.id != job.id]
            else:
                job.enabled = False
                job.state.next_run_at_ms = None
