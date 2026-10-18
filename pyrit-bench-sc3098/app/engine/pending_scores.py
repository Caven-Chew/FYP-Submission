"""Bounded background scoring on the run's existing asyncio loop."""
import asyncio


class PendingScores:
    """One judge request stream, at most two captured replies awaiting results."""

    def __init__(self):
        self._tasks = []
        self._judge_slot = asyncio.Semaphore(1)

    @property
    def pending_count(self):
        return sum(not task.done() for task in self._tasks)

    async def submit(self, score):
        while self.pending_count >= 2:
            await asyncio.wait([task for task in self._tasks if not task.done()],
                               return_when=asyncio.FIRST_COMPLETED)

        async def run():
            async with self._judge_slot:
                return await score()

        task = asyncio.create_task(run())
        self._tasks.append(task)
        return task

    async def drain(self):
        if self._tasks:
            await asyncio.gather(*self._tasks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # Cooperative Stop must retain and finish scoring already saved replies.
        results = await asyncio.gather(*self._tasks, return_exceptions=True)
        if exc_type is None:
            for result in results:
                if isinstance(result, BaseException):
                    raise result


def prepare_reply_for_close(target, closing=True):
    """Require a final text check on this send before persisting or judging it."""
    from ..targets.browser.target import BrowserUITarget
    if isinstance(target, BrowserUITarget):
        target._confirm_before_close = closing


async def close_after_reply(target):
    """Close after send has confirmed its final text; API targets need no wait."""
    from ..targets.browser.target import BrowserUITarget
    if isinstance(target, BrowserUITarget) and target._session:
        await target.close_conversation()
