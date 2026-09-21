"""
Wrapper that monkeypatches multiprocessing.pool.ThreadPool BEFORE importing
ultralytics, then execs the requested script with the remaining CLI args.

Why: this sandbox blocks creation of Windows named pipes. CPython's
multiprocessing.pool.ThreadPool.__init__ (even though it only spawns Python
threads, not processes) still calls Pool.__init__, which unconditionally
creates an internal change-notifier via self._ctx.SimpleQueue() -> a real
OS-level named pipe on Windows. Ultralytics uses ThreadPool(NUM_THREADS) for
label caching (ultralytics/data/dataset.py: cache_labels), independent of the
--workers dataloader setting, so it always trips this even with workers=0.

Fix: replace multiprocessing.pool.ThreadPool with a drop-in shim backed by
concurrent.futures.ThreadPoolExecutor (which uses queue.SimpleQueue, no pipes)
before any ultralytics module does `from multiprocessing.pool import ThreadPool`.

Usage: python run_patched.py <target_script.py> [args passed to target...]
"""
import sys
import multiprocessing.pool
from concurrent.futures import ThreadPoolExecutor


class SafeThreadPool:
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


multiprocessing.pool.ThreadPool = SafeThreadPool

if __name__ == "__main__":
    target_script = sys.argv[1]
    sys.argv = sys.argv[1:]  # target script sees itself as argv[0]
    with open(target_script, encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, target_script, "exec"), {"__name__": "__main__", "__file__": target_script})
