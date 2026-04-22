"""
ProgressDisplay — a `rich`-based live progress bar for the CLIF pipeline.

Design
------
The display is a drop-in wrapper for `sys.stdout`.  Every line written to it
(via `print(...)` anywhere in the process, including parent re-emission of
subprocess stdout) is classified with a few small regexes, and its state
updates a single `rich.live.Live` display composed of:

    CLIF Pipeline  Step 2/4: Table One Generation  ████████░░░░ 25%  00:47  ETA 02:22
       ↳ Stratified Table Ones Complete · peak 24.5 GB

- A filled progress bar proportional to `completed / total_steps`
- `Step N/total: TITLE` badge, updated at each top-level STEP banner
- Elapsed + rich's auto-computed ETA
- A dim sub-phase line pulled from `[Memory Checkpoint: …]` and
  `[…] Memory: X MB | Peak: Y MB | Time: Z s` lines

Non-phase input lines are silently dropped — they're still in the workflow
log file because the Python `logging.FileHandler`s are untouched.

The rich library takes care of:
- Cross-platform TTY rendering (macOS Terminal/iTerm, Linux xterm/tmux,
  Windows Terminal/VS Code/modern cmd.exe)
- ANSI color + cursor control, with graceful fallback to plain lines
  when `sys.stdout` is not a TTY (piped through `tee`, CI logs, etc.)
"""

import re
import time
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text


_STEP_RE = re.compile(r"^STEP\s+(\S+):\s*(.+?)\s*$")
_CHECKPOINT_RE = re.compile(r"^\s*\[Memory Checkpoint:\s*(.+?)\]\s*$")
_TIMING_RE = re.compile(
    r"^\s*\[(.+?)\]\s+Memory:\s*([\d.]+)\s*MB\s*\|\s*Peak:\s*([\d.]+)\s*MB\s*\|\s*Time:\s*([\d.]+)s\s*$"
)
_EQUALS_RE = re.compile(r"^=+$")
_LOG_PREFIX_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\s+-\s+\w+\s+-\s+(.*)$"
)
# CLIAnalysisRunner emits progress lines like "🔍 Running validation for X..."
# and "✅ Completed analysis for X" via ConsoleFormatter. Surfacing these as
# sub-phase updates keeps the live display informative during --validate-only,
# which otherwise never emits [Memory Checkpoint: …] or STEP N: banners.
_RUNNER_PROGRESS_RE = re.compile(
    r"^(?:🔍|✅|ℹ️|⚠️|❌)\s+(.+?)\.{0,3}\s*$"
)
_RUNNER_SECTION_RE = re.compile(r"^Processing\s+(.+?)\s+table\s*$", re.IGNORECASE)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Titles emitted by `run_project.py:print_header` for the four top-level
# workflow steps.  Sub-runners (TableOneRunner, ECDFRunner) emit their own
# internal `STEP 1:` / `STEP 2:` banners for validation/loading phases, which
# match _STEP_RE but MUST NOT advance the top-level bar — they are surfaced
# as sub-phases instead.
_TOP_LEVEL_TITLES = frozenset({
    "CLIF VALIDATION",
    "TABLE ONE GENERATION",
    "WARD TABLE ONE GENERATION",
    "GET ECDF BINS",
})


class ProgressDisplay:
    """rich-backed live progress display, usable as a `sys.stdout` wrapper."""

    def __init__(self, total_steps, out_stream):
        self.total_steps = max(1, int(total_steps))
        self._out_stream = out_stream
        self._console = Console(
            file=out_stream,
            force_terminal=self._safe_isatty(out_stream),
        )
        self._progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold cyan]CLIF Pipeline[/]"),
            TextColumn("[bold]{task.fields[step_title]}[/]"),
            BarColumn(bar_width=28),
            # Use explicit "N of M done" instead of a percentage — avoids the
            # "Step 4/4 · 75%" contradiction where the step counter reads
            # "running last step" but the bar honestly shows "3 of 4
            # completed".  `completed` is the number of *finished* steps.
            TextColumn("{task.completed:.0f} of {task.total} done"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            console=self._console,
        )
        self._task_id = None
        self._step_n = 0
        self._step_title = "—"
        self._sub_phase = "starting…"
        self._peak_mb = 0.0
        self._buffer = ""
        self._banner_state = "idle"
        self._live = None

    # ---- public lifecycle ----
    def start(self):
        if self._task_id is not None:
            return
        self._task_id = self._progress.add_task(
            "",
            total=self.total_steps,
            step_n=0,
            step_title="—",
        )
        self._live = Live(
            self._renderable(),
            console=self._console,
            refresh_per_second=4,
            transient=True,
            # Critical: rich.Live defaults to hijacking sys.stdout/stderr on
            # start, which replaces our ProgressDisplay wrapper and starves the
            # classifier of input.  Leave the streams alone — we've already
            # wrapped stdout upstream, and file logs cover stderr.
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.start()

    def stop(self):
        if self._live is None:
            return
        try:
            # On clean shutdown mark the bar full — we only advance on
            # *start* of the next step, so the final step otherwise never
            # ticks to complete.
            if self._task_id is not None:
                # Preserve an explicit end-state title set by
                # _mark_all_done_and_stop; otherwise apply the default.
                if "All steps complete" not in (self._step_title or ""):
                    self._step_title = "[green]All steps complete[/]"
                if self._sub_phase in (None, "starting…"):
                    self._sub_phase = "done"
                self._progress.update(
                    self._task_id,
                    completed=self.total_steps,
                    step_title=self._step_title,
                )
            self._live.update(self._renderable(), refresh=True)
        finally:
            self._live.stop()
            self._live = None

    # ---- file-like protocol (so `sys.stdout = ProgressDisplay(...)` works) ----
    def write(self, data):
        if not data:
            return 0
        # Once the live display has been torn down (e.g. after the
        # "Launching Web App" sentinel fires), pass writes through to the
        # real stdout.  This lets status messages printed after the
        # progress bar finishes — "share these files when reporting an
        # issue", uvicorn startup notes, error messages — actually reach
        # the user's terminal.
        if self._live is None and self._task_id is not None:
            try:
                return self._out_stream.write(data)
            except Exception:
                return len(data)
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._on_line(line)
        return len(data)

    def flush(self):
        if self._live is None and self._task_id is not None:
            try:
                self._out_stream.flush()
            except Exception:
                pass

    def isatty(self):
        # Anything inspecting our "tty-ness" should see False — we drop most
        # writes, so callers that branch on isatty() (e.g. progress bars)
        # should pick the non-interactive path.
        return False

    @property
    def encoding(self):
        return "utf-8"

    @property
    def buffer(self):
        return None

    def close(self):
        self.stop()

    # ---- rendering ----
    def _renderable(self):
        # Dynamic renderable: rich's Live auto-refresh calls into this object
        # on every tick, so state updates (sub_phase, peak_mb) made silently
        # by the classifier propagate to the bar without an explicit _redraw.
        owner = self

        class _Dynamic:
            def __rich_console__(self_inner, console, options):
                peak = owner._fmt_mem(owner._peak_mb)
                sub = Text(
                    f"   ↳ {owner._sub_phase} · peak {peak}",
                    style="dim",
                )
                yield from console.render(
                    Group(owner._progress, sub), options
                )

        return _Dynamic()

    def _redraw(self):
        # Explicit redraw used only at step transitions; auto-refresh
        # (4 Hz via Live) handles incremental updates.
        if self._live is not None:
            self._live.refresh()

    @staticmethod
    def _safe_isatty(stream):
        try:
            return bool(stream.isatty())
        except Exception:
            return False

    @staticmethod
    def _fmt_mem(mb):
        if mb <= 0:
            return "—"
        if mb >= 1024:
            return f"{mb / 1024:.1f} GB"
        return f"{mb:.0f} MB"

    # ---- classifier ----
    @staticmethod
    def _strip_log_prefix(line):
        m = _LOG_PREFIX_RE.match(line)
        return m.group(1) if m else line

    @staticmethod
    def _strip_ansi(s):
        return _ANSI_RE.sub("", s)

    # Text that signals the workflow is done and the web app is about to
    # start.  When we see any of these, snap the bar to 100% and tear the
    # live display down so uvicorn's output can scroll normally.
    _END_OF_WORKFLOW_MARKERS = (
        "launching web app",
        "🚀 launching web app",
    )

    def _mark_all_done_and_stop(self):
        if self._task_id is not None:
            self._step_n = self.total_steps
            self._step_title = "[green]All steps complete[/]"
            self._sub_phase = "launching web app…"
            self._progress.update(
                self._task_id,
                completed=self.total_steps,
                step_title=self._step_title,
            )
            self._redraw()
        self.stop()

    def _advance_step(self, number, title):
        # Only advance for *known* top-level banners.  Internal sub-runner
        # banners (e.g. TableOneRunner's "STEP 1: VALIDATING CONFIGURATION")
        # are surfaced as sub-phases instead.
        if title not in _TOP_LEVEL_TITLES:
            self._sub_phase = f"Step {number}: {title}"
            self._redraw()
            return

        # Cap step_n at total_steps defensively — guards against duplicate
        # banners (e.g. if a step prints its header twice).
        if self._step_n >= self.total_steps:
            self._step_title = title
            self._sub_phase = "starting…"
            if self._task_id is not None:
                self._progress.update(
                    self._task_id,
                    step_title=title,
                )
            self._redraw()
            return

        self._step_n += 1
        self._step_title = title
        self._sub_phase = "starting…"
        if self._task_id is not None:
            # `completed = step_n - 1` means "N-1 steps are actually done;
            # step N is in flight".  Final advance to `total_steps` happens
            # in `stop()` so the bar visibly finishes at 100%.
            self._progress.update(
                self._task_id,
                completed=max(0, self._step_n - 1),
                step_n=self._step_n,
                step_title=title,
            )
        self._redraw()

    def _on_line(self, raw):
        line = self._strip_log_prefix(raw)
        stripped = line.strip()
        is_equals = (
            bool(stripped)
            and bool(_EQUALS_RE.match(stripped))
            and len(stripped) >= 20
        )

        # ── === banner state machine ──
        if self._banner_state == "expect_title":
            if not stripped:
                return
            if is_equals:
                self._banner_state = "idle"
                return
            # End-of-workflow banner? Snap to 100% and tear down before
            # launch_app's uvicorn output starts scrolling.
            if any(m in stripped.lower() for m in self._END_OF_WORKFLOW_MARKERS):
                self._mark_all_done_and_stop()
                return
            m = _STEP_RE.match(stripped)
            if m:
                self._advance_step(m.group(1), m.group(2).strip())
            else:
                self._sub_phase = stripped
                self._redraw()
            self._banner_state = "expect_close"
            return

        if self._banner_state == "expect_close":
            if is_equals:
                self._banner_state = "idle"
                return
            if not stripped:
                return
            self._banner_state = "idle"

        if is_equals:
            self._banner_state = "expect_title"
            return

        if not stripped:
            return

        # End-of-workflow signal outside of a === banner (the "🚀 Launching
        # Web App..." direct print at run_project.py:915).
        if any(m in stripped.lower() for m in self._END_OF_WORKFLOW_MARKERS):
            self._mark_all_done_and_stop()
            return

        # ── plain-line classifiers ──
        m = _STEP_RE.match(stripped)
        if m:
            self._advance_step(m.group(1), m.group(2).strip())
            return

        # Memory checkpoint + timing lines fire often during a step (every
        # stratum, every bin collection, every waterfall chunk).  We update
        # the state silently here; rich's auto-refresh (4 Hz) picks it up
        # via the dynamic renderable, so the bar still ticks but each event
        # does not force an explicit redraw/write.
        m = _CHECKPOINT_RE.match(stripped)
        if m:
            self._sub_phase = m.group(1).strip()
            return

        m = _TIMING_RE.match(stripped)
        if m:
            peak = float(m.group(3))
            if peak > self._peak_mb:
                self._peak_mb = peak
            self._sub_phase = m.group(1).strip()
            return

        # CLIAnalysisRunner per-table progress. Match after checkpoint/timing
        # so Table One / ECDF steps keep their richer sub-phase text; this is
        # what keeps --validate-only from sitting on "starting…" for minutes.
        clean = self._strip_ansi(stripped)
        m = _RUNNER_SECTION_RE.match(clean)
        if m:
            self._sub_phase = f"processing {m.group(1).lower()}"
            return

        m = _RUNNER_PROGRESS_RE.match(clean)
        if m:
            self._sub_phase = m.group(1).strip()
            return

        # Unclassified input — ignore (still in file log).
