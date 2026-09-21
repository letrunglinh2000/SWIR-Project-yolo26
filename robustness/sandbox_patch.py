"""
Import this module FIRST, before `ultralytics`, in every entry-point script
that touches the dataloader label cache (training, validation, or any
DetectionValidator-based attack evaluation).

Why this exists: this specific execution sandbox blocks creation of Windows
named pipes. Ultralytics' label-cache step
(ultralytics/data/dataset.py:cache_labels) unconditionally uses
`multiprocessing.pool.ThreadPool`, and CPython's `Pool.__init__` creates an
internal change-notifier via `self._ctx.SimpleQueue()`, which is a real
OS-level named pipe on Windows -- even though ThreadPool only spawns Python
threads, not processes. This trips regardless of the `--workers` dataloader
setting, since it fires during label caching, not batch loading.

The fix: replace `multiprocessing.pool.ThreadPool` with a drop-in shim backed
by `concurrent.futures.ThreadPoolExecutor` (which uses `queue.SimpleQueue`,
no OS pipes) before any ultralytics module resolves
`from multiprocessing.pool import ThreadPool`.

This is a sandbox-specific workaround, not a correction to the SWIR project's
own code -- on an unrestricted machine this module is a harmless no-op import
(the real ThreadPool would work fine there too).
"""
import multiprocessing.pool
from concurrent.futures import ThreadPoolExecutor


class _SafeThreadPool:
    def __init__(self, processes=None, *a, **kw):
        self._executor = ThreadPoolExecutor(max_workers=processes or 4)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._executor.shutdown(wait=True)
        return False

    def imap(self, func, iterable, chunksize=1):
        return self._executor.map(func, iterable)

    def imap_unordered(self, func, iterable, chunksize=1):
        return self._executor.map(func, iterable)

    def map(self, func, iterable, chunksize=1):
        return list(self._executor.map(func, iterable))

    def close(self):
        pass

    def join(self):
        self._executor.shutdown(wait=True)

    def terminate(self):
        self._executor.shutdown(wait=False, cancel_futures=True)


multiprocessing.pool.ThreadPool = _SafeThreadPool
