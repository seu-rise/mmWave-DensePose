"""Terminal progress without changing model execution or flooding GUI journals."""
from contextlib import nullcontext
import math
import sys
import time

from .execution import current_execution


def _duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "--:--"
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class _ProgressStream:
    """Keep tqdm's monitor-thread refreshes out of the execution journal too."""

    def __init__(self, stream, observer):
        self.stream, self.observer = stream, observer

    def _context(self):
        suspend = getattr(self.observer, "suspend_console_capture", None)
        return suspend() if suspend is not None else nullcontext()

    def write(self, text):
        with self._context():
            return self.stream.write(text)

    def flush(self):
        with self._context():
            return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class TerminalProgress:
    """Use tqdm in a TTY and bounded newline snapshots for redirected output.

    tqdm is already present in the compute profiles through torch-geometric.
    A plain-text fallback keeps progress available in minimal installations too.
    Progress output bypasses only the observer's console capture; regular prints,
    warnings and errors continue to appear in both the terminal and GUI journal.
    """

    def __init__(self, total, label, *, stream=None, interval=None):
        self.total = max(0, int(total))
        self.label = label
        output = sys.stderr if stream is None else stream
        self.stream = _ProgressStream(output, current_execution())
        try:
            self.dynamic = bool(self.stream.isatty())
        except (AttributeError, OSError):
            self.dynamic = False
        self.interval = (.25 if self.dynamic else 1.) if interval is None else max(0., float(interval))
        self.current, self.metrics = 0, {}
        self.started, self.last_update = time.monotonic(), -math.inf
        self.bar, self.closed, self.enabled = None, False, True

    def _capture_context(self):
        observer = current_execution()
        suspend = getattr(observer, "suspend_console_capture", None)
        return suspend() if suspend is not None else nullcontext()

    def __enter__(self):
        if self.dynamic:
            try:
                from tqdm import tqdm
            except ImportError:
                self.dynamic = False
            else:
                try:
                    with self._capture_context():
                        self.bar = tqdm(total=self.total, desc=self.label, unit="batch", file=self.stream,
                                        ascii=True, dynamic_ncols=True, mininterval=self.interval, leave=True,
                                        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                                                   "[{elapsed}, ETA {remaining}, {rate_fmt}]{postfix}")
                except (OSError, ValueError):
                    self.enabled = False
        if self.bar is None:
            self._snapshot()
        return self

    def update(self, current, **metrics):
        if not self.enabled or self.closed:
            return
        self.current = max(self.current, min(int(current), self.total))
        self.metrics.update({key: float(value) for key, value in metrics.items()
                             if isinstance(value, (float, int)) and math.isfinite(value)})
        now = time.monotonic()
        due = now - self.last_update >= self.interval or self.current == self.total
        if self.bar is not None:
            try:
                with self._capture_context():
                    if due:
                        self.bar.set_postfix(self.metrics, refresh=False)
                    self.bar.update(self.current - self.bar.n)
                if due:
                    self.last_update = now
            except (OSError, ValueError):
                self.enabled = False
        elif due:
            self._snapshot()

    def set_metrics(self, **metrics):
        self.update(self.current, **metrics)

    def _snapshot(self):
        if not self.enabled:
            return
        now = time.monotonic()
        elapsed = max(0., now - self.started)
        rate = self.current / elapsed if elapsed and self.current else None
        remaining = (self.total - self.current) / rate if rate else None
        fraction = self.current / self.total if self.total else 0.
        blocks = min(20, max(0, int(fraction * 20)))
        bar = "#" * blocks + "-" * (20 - blocks)
        metric_text = " ".join(f"{key}={value:.5g}" for key, value in self.metrics.items())
        rate_text = f"{rate:.2f} batch/s" if rate else "-- batch/s"
        line = (f"{self.label}: {fraction:6.1%}|{bar}| {self.current}/{self.total} "
                f"[elapsed {_duration(elapsed)}, ETA {_duration(remaining)}, {rate_text}]")
        try:
            with self._capture_context():
                self.stream.write(line + (" " + metric_text if metric_text else "") + "\n")
                self.stream.flush()
            self.last_update = now
        except (OSError, ValueError):
            self.enabled = False

    def close(self):
        if self.closed:
            return
        if self.bar is not None:
            try:
                with self._capture_context():
                    self.bar.set_postfix(self.metrics, refresh=False)
                    self.bar.close()
            except (OSError, ValueError):
                self.enabled = False
        elif self.enabled and self.current < self.total:
            self._snapshot()
        self.closed = True

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
