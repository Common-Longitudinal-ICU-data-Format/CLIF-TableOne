"""
Cross-platform stress test harness for:
    uv run python run_project.py --get-ecdf --ward --visualize

Modes:
    default         : run pipeline, measure tree-wide RSS/timing, checksum outputs
    --check-install : run `uv sync` only, classify failure mode, exit
    --compare-to X  : after run, diff output_checksums against baseline JSONL

Usage:
    uv run python dev/stress_test/stress_test.py --label mac-baseline
    uv run python dev/stress_test/stress_test.py --label win-x1 --compare-to output/stress_test/results.jsonl
    uv run python dev/stress_test/stress_test.py --check-install --label win-corp-laptop
"""

import argparse
import contextlib
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.json"
FINAL_OUT = REPO_ROOT / "output" / "final"
STRESS_DIR = REPO_ROOT / "output" / "stress_test"
RESULTS_JSONL = STRESS_DIR / "results.jsonl"

POLARS_WARNING_PATTERNS = re.compile(
    r"Traceback|panicked|PanicException|"
    r"Error:|Warning:|"
    r"DeprecationWarning|FutureWarning|RuntimeWarning|UserWarning|"
    r"ArrowError|MemoryError|OOM|Killed|segfault|"
    r"polars[^-]*?(?:warning|error)",
    re.IGNORECASE,
)


def get_polars_info() -> dict:
    try:
        import polars as pl
    except ImportError:
        return {"version": None, "build_info": None, "error": "polars not installed"}
    info: dict = {"version": pl.__version__, "build_info": None}
    try:
        info["build_info"] = pl.build_info()
    except Exception as e:
        info["build_info"] = {"error": str(e)}
    return info


def get_host_fingerprint() -> dict:
    vm = psutil.virtual_memory()
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "processor": platform.processor() or "",
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(vm.total / 1024**3, 2),
        "python_version": platform.python_version(),
    }


def get_git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def directory_size_bytes(path: Path) -> int:
    if not path or not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def checksum_outputs(root: Path) -> dict:
    """SHA256 every file under root, excluding meta/ (nondeterministic timing reports)."""
    out: dict = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel.startswith("meta/"):
            continue
        try:
            out[rel] = sha256_file(p)
        except OSError as e:
            out[rel] = f"<unreadable: {e}>"
    return out


_STAGE_LINE_RE = re.compile(
    r"^(Step [^\s].+?)\s+(\d{2}:\d{2}:\d{2})\s+(\d{2}:\d{2}:\d{2})\s+(\d{2}:\d{2}:\d{2})\s+"
    r"(?:([\d,.]+)\s+MB|\S+)\s+(\S.*)$"
)


def parse_workflow_report(report_path: Path) -> list:
    if not report_path.exists():
        return []
    text = report_path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        m = _STAGE_LINE_RE.match(line.strip())
        if not m:
            continue
        label, start, end, dur, peak_mb, report = m.groups()
        peak_float = float(peak_mb.replace(",", "")) if peak_mb else None
        rows.append({
            "step": label.strip(),
            "start_offset": start,
            "end_offset": end,
            "duration": dur,
            "peak_rss_mb": peak_float,
            "report": report.strip(),
        })
    return rows


def grep_polars_warnings(log_path: Path, max_lines: int = 200) -> list:
    if not log_path.exists():
        return []
    hits = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if POLARS_WARNING_PATTERNS.search(line):
                hits.append(line.rstrip("\n"))
                if len(hits) >= max_lines:
                    break
    return hits


class RSSPoller:
    """Polls memory usage of a subprocess tree at a fixed interval from a background thread.

    Tracks three summed-across-tree metrics:
      - rss_sum: RSS summed across procs. OVER-counts shared copy-on-write pages
        (ProcessPoolExecutor workers share large pages with the parent), so a
        tree holding 2 GB of shared pandas DataFrames across 10 workers reports
        ~20 GB here. Kept for backward comparison only, not for OOM reasoning.
      - uss_sum: Unique Set Size summed across procs. Excludes all shared pages,
        so it's the conservative lower bound of private memory the tree "owns".
        This is the primary cross-platform memory pressure metric.
      - pss_sum: Proportional Set Size sum — ideal (counts shared pages once,
        proportionally), but Linux-only via /proc/pid/smaps. None on macOS/Windows.
    """

    def __init__(self, root_pid: int, interval: float = 0.5, csv_path: Optional[Path] = None):
        self.root_pid = root_pid
        self.interval = interval
        self.csv_path = csv_path
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_rss_sum_mb = 0.0
        self.peak_uss_sum_mb = 0.0
        self.peak_pss_sum_mb: Optional[float] = None
        self.pss_available = False
        self.peak_per_pid_uss: dict = {}
        self.samples = 0
        # macOS unsigned-Python denies memory_full_info (needs task_for_pid);
        # we fall back to memory_info RSS and set this flag so downstream code
        # can tell "USS is 0" apart from "USS was never observable".
        self.uss_unavailable = False
        self._warned_uss_unavailable = False
        self._csv_file = None
        self._csv_writer = None

    def start(self):
        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["t_sec", "rss_sum_mb", "uss_sum_mb", "pss_sum_mb", "n_procs"])
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._csv_file is not None:
            self._csv_file.close()

    def _run(self):
        t0 = time.time()
        while not self._stop.is_set():
            rss_sum = 0.0
            uss_sum = 0.0
            pss_sum = 0.0
            saw_pss = False
            n_procs = 0
            try:
                root = psutil.Process(self.root_pid)
                procs = [root] + root.children(recursive=True)
            except psutil.NoSuchProcess:
                break
            tick_uss_unavailable = False
            for p in procs:
                try:
                    fi = p.memory_full_info()
                    rss_sum += fi.rss
                    uss_sum += fi.uss
                    maybe_pss = getattr(fi, "pss", None)
                    if maybe_pss is not None:
                        pss_sum += maybe_pss
                        saw_pss = True
                    n_procs += 1
                    uss_mb = fi.uss / 1024 / 1024
                    pid = p.pid
                    if uss_mb > self.peak_per_pid_uss.get(pid, 0):
                        self.peak_per_pid_uss[pid] = uss_mb
                except psutil.AccessDenied:
                    # macOS unsigned Python: memory_full_info needs task_for_pid,
                    # but memory_info (proc_pidinfo PROC_PIDTASKINFO) works.
                    try:
                        bi = p.memory_info()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                    rss_sum += bi.rss
                    n_procs += 1
                    tick_uss_unavailable = True
                    self.uss_unavailable = True
                except psutil.NoSuchProcess:
                    continue

            if tick_uss_unavailable and not self._warned_uss_unavailable:
                print(
                    "[stress_test] macOS: USS/PSS unavailable "
                    "(psutil.memory_full_info AccessDenied). Recording RSS only.",
                    file=sys.stderr,
                    flush=True,
                )
                self._warned_uss_unavailable = True

            rss_mb = rss_sum / 1024 / 1024
            uss_mb_total = uss_sum / 1024 / 1024
            pss_mb_total = pss_sum / 1024 / 1024 if saw_pss else None

            if rss_mb > self.peak_rss_sum_mb:
                self.peak_rss_sum_mb = rss_mb
            if uss_mb_total > self.peak_uss_sum_mb:
                self.peak_uss_sum_mb = uss_mb_total
            if saw_pss:
                self.pss_available = True
                if self.peak_pss_sum_mb is None or pss_mb_total > self.peak_pss_sum_mb:
                    self.peak_pss_sum_mb = pss_mb_total

            self.samples += 1
            if self._csv_writer is not None:
                pss_cell = f"{pss_mb_total:.1f}" if saw_pss else ""
                uss_cell = "" if tick_uss_unavailable else f"{uss_mb_total:.1f}"
                self._csv_writer.writerow([
                    f"{time.time() - t0:.2f}",
                    f"{rss_mb:.1f}",
                    uss_cell,
                    pss_cell,
                    n_procs,
                ])
                if self._csv_file is not None:
                    self._csv_file.flush()
            self._stop.wait(self.interval)


@contextlib.contextmanager
def temp_tables_path(data_dir: Optional[Path]):
    """If data_dir given, rewrite config/config.json to point there; restore on exit."""
    if data_dir is None:
        yield
        return
    backup_text = CONFIG_PATH.read_text(encoding="utf-8")
    cfg = json.loads(backup_text)
    cfg["tables_path"] = str(data_dir)
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    os.replace(tmp, CONFIG_PATH)
    try:
        yield
    finally:
        CONFIG_PATH.write_text(backup_text, encoding="utf-8")


def find_existing_pipeline() -> Optional[psutil.Process]:
    """Return a running run_project.py process that is not a descendant of this harness."""
    my_pid = os.getpid()
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(p.info.get("cmdline") or [])
            if "run_project.py" in cmdline and p.pid != my_pid:
                return p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


STAGE_FLAGS = {
    "validate": ["--validate-only"],
    "tableone": ["--tableone-only"],
    "ward":     ["--ward-only"],
    "ecdf":     ["--get-ecdf-only", "--visualize"],
}


def run_pipeline(
    label: str,
    data_dir: Optional[Path],
    poll_interval: float,
    only_stage: Optional[str] = None,
) -> dict:
    existing = find_existing_pipeline()
    if existing is not None:
        raise RuntimeError(
            f"refusing to launch: run_project.py already running (pid {existing.pid}). "
            f"Kill it or wait for it to finish before running the harness."
        )

    run_dir = STRESS_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    rss_csv = run_dir / "rss_timeseries.csv"

    # Only wipe final/ when running the full pipeline; a single-stage run should
    # leave other stages' outputs intact for incremental exploration.
    if only_stage is None and FINAL_OUT.exists():
        shutil.rmtree(FINAL_OUT)

    env = os.environ.copy()
    env["POLARS_VERBOSE"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")

    if only_stage is not None:
        pipeline_args = STAGE_FLAGS[only_stage]
    else:
        pipeline_args = ["--get-ecdf", "--ward", "--visualize"]
    cmd = ["uv", "run", "python", "run_project.py", *pipeline_args]
    start = time.time()
    start_ts = datetime.now().isoformat(timespec="seconds")

    with open(stdout_path, "w", encoding="utf-8", errors="replace") as so, \
         open(stderr_path, "w", encoding="utf-8", errors="replace") as se:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=so,
            stderr=se,
            env=env,
        )

        poller = RSSPoller(proc.pid, interval=poll_interval, csv_path=rss_csv)
        poller.start()
        try:
            exit_code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                exit_code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                exit_code = proc.wait()
            poller.stop()
            raise
        finally:
            poller.stop()

    elapsed = time.time() - start

    oom_killed = False
    total_ram_mb = psutil.virtual_memory().total / 1024 / 1024
    if platform.system() == "Linux":
        oom_killed = exit_code == -9 or exit_code == 137
    # Fallback order for pressure estimate: PSS (Linux-only, ideal) →
    # USS (private pages, only if observable) → RSS (upper bound, but the
    # only thing we have on macOS unsigned Python).
    if poller.pss_available:
        pressure_peak = poller.peak_pss_sum_mb
        pressure_metric = "pss"
    elif not poller.uss_unavailable:
        pressure_peak = poller.peak_uss_sum_mb
        pressure_metric = "uss"
    else:
        pressure_peak = poller.peak_rss_sum_mb
        pressure_metric = "rss"
    if exit_code != 0 and pressure_peak > 0.85 * total_ram_mb:
        oom_killed = True

    stage_rows = parse_workflow_report(FINAL_OUT / "meta" / "workflow_execution_report.txt")
    checksums = checksum_outputs(FINAL_OUT)
    warnings = grep_polars_warnings(stderr_path)

    return {
        "start_ts": start_ts,
        "elapsed_sec": elapsed,
        "exit_code": exit_code,
        "oom_killed": oom_killed,
        "peak_uss_sum_mb": poller.peak_uss_sum_mb,
        "peak_pss_sum_mb": poller.peak_pss_sum_mb,
        "peak_rss_sum_mb": poller.peak_rss_sum_mb,
        "pss_available": poller.pss_available,
        "uss_unavailable": poller.uss_unavailable,
        "pressure_metric": pressure_metric,
        "peak_uss_per_pid_mb": poller.peak_per_pid_uss,
        "rss_samples": poller.samples,
        "stage_table": stage_rows,
        "output_checksums": checksums,
        "polars_warnings": warnings,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "rss_csv": str(rss_csv),
    }


def infer_ward_isolation(stage_rows: list, tree_peak_mb: float) -> str:
    """Compare measured tree peak against max vs sum of CI+Ward per-step peaks.

    Per-step peak_rss_mb comes from run_project's internal MemoryMonitor which
    uses memory_info().rss (not USS) — so it shares the double-counting caveat.
    This inference is still informative as a directional signal: isolated ~ max,
    regression ~ sum. tree_peak_mb argument should be USS-based (or PSS on Linux)
    for the cleanest comparison.
    """
    ci = next((r["peak_rss_mb"] for r in stage_rows if "CI Table One" in r["step"]), None)
    ward = next((r["peak_rss_mb"] for r in stage_rows if "Ward Table One" in r["step"]), None)
    if ci is None or ward is None or tree_peak_mb <= 0:
        return "unknown (missing per-step peak)"
    max_cw = max(ci, ward)
    sum_cw = ci + ward
    dist_max = abs(tree_peak_mb - max_cw)
    dist_sum = abs(tree_peak_mb - sum_cw)
    verdict = "isolated (peak ~= max)" if dist_max < dist_sum else "NOT isolated (peak ~= sum)"
    return (
        f"{verdict}: tree_peak_uss={tree_peak_mb:.1f} MB, "
        f"max(CI,Ward)={max_cw:.1f} MB, CI+Ward={sum_cw:.1f} MB"
    )


def check_install(label: str) -> dict:
    run_dir = STRESS_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "uv_sync.log"

    start = time.time()
    r = subprocess.run(
        ["uv", "sync"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    elapsed = time.time() - start
    log_path.write_text(
        f"# uv sync\n# exit_code={r.returncode}\n# elapsed={elapsed:.1f}s\n\n"
        f"--- STDOUT ---\n{r.stdout}\n\n--- STDERR ---\n{r.stderr}\n",
        encoding="utf-8",
    )

    combined = (r.stdout + r.stderr).lower()
    if r.returncode == 0:
        classification = "ok"
    elif "polars" in combined and ("wheel" in combined or "no matching distribution" in combined or "build" in combined):
        classification = "polars_wheel_missing"
    elif "rustc" in combined or "cargo" in combined or "maturin" in combined:
        classification = "toolchain_missing"
    else:
        classification = "other_failure"

    return {
        "classification": classification,
        "exit_code": r.returncode,
        "elapsed_sec": elapsed,
        "log_path": str(log_path),
    }


def compare_checksums(baseline_row: dict, current_row: dict) -> list:
    base = baseline_row.get("output_checksums", {})
    curr = current_row.get("output_checksums", {})
    all_keys = sorted(set(base) | set(curr))
    diffs = []
    for k in all_keys:
        b, c = base.get(k), curr.get(k)
        if b != c:
            diffs.append({"file": k, "baseline": b, "current": c})
    return diffs


def load_latest_runtime_row(jsonl_path: Path) -> Optional[dict]:
    if not jsonl_path.exists():
        return None
    last = None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "runtime":
                last = row
    return last


def emit_row(row: dict):
    STRESS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def print_human_summary(row: dict):
    print()
    print("=" * 80)
    print(f"Stress test: {row.get('label')}  |  kind={row.get('kind')}")
    print("=" * 80)
    if row.get("kind") == "install":
        print(f"uv sync exit:       {row.get('exit_code')}")
        print(f"classification:     {row.get('classification')}")
        print(f"elapsed:            {row.get('elapsed_sec'):.1f} s")
        print(f"log:                {row.get('log_path')}")
        return
    h = row.get("host", {})
    print(f"host:               {h.get('hostname')} ({h.get('system')} {h.get('release')})")
    print(f"cpu/ram:            {h.get('cpu_count_logical')} logical, {h.get('total_ram_gb')} GB")
    print(f"polars:             {row.get('polars', {}).get('version')}")
    print(f"data:               {row.get('tables_path')} ({row.get('tables_path_mb'):.1f} MB)")
    print(f"factor:             {row.get('factor', 1)}")
    print(f"wall time:          {row.get('elapsed_sec'):.1f} s")
    print(f"exit code:          {row.get('exit_code')}  (oom_killed={row.get('oom_killed')})")
    uss = row.get("peak_uss_sum_mb", 0.0) or 0.0
    pss = row.get("peak_pss_sum_mb")
    rss_sum = row.get("peak_rss_sum_mb", 0.0) or 0.0
    if row.get("uss_unavailable"):
        print(f"peak USS (private):        N/A (macOS AccessDenied on memory_full_info)")
    else:
        print(f"peak USS (private): {uss:>10,.1f} MB   <- primary memory-pressure metric")
    if pss is not None:
        print(f"peak PSS (Linux):   {pss:>10,.1f} MB")
    print(f"peak RSS sum:       {rss_sum:>10,.1f} MB   (double-counts shared pages, use as upper bound only)")
    print(f"pressure metric:    {row.get('pressure_metric', 'n/a')}")
    print(f"samples:            {row.get('rss_samples')} @ 0.5 s")
    print()
    print("stage timeline:")
    for s in row.get("stage_table", []):
        peak = f"{s['peak_rss_mb']:,.1f} MB" if s["peak_rss_mb"] else "—"
        print(f"  {s['step']:<30} {s['duration']:>10}  peak={peak}")
    print()
    print(f"ward isolation:     {row.get('ward_isolation_inference', 'n/a')}")
    print()
    print(f"output files:       {len(row.get('output_checksums', {}))} checksummed")
    print(f"polars warnings:    {len(row.get('polars_warnings', []))}")
    if row.get("diff_vs_baseline") is not None:
        d = row["diff_vs_baseline"]
        if not d:
            print("baseline diff:      0 files differ (outputs match baseline)")
        else:
            print()
            print(f"!! DIVERGENCE vs baseline: {len(d)} files differ")
            for entry in d[:10]:
                print(f"  - {entry['file']}")
            if len(d) > 10:
                print(f"  ... ({len(d) - 10} more)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True, help="Free-text label for this run")
    ap.add_argument("--data", type=Path, help="Override config tables_path for this run")
    ap.add_argument("--factor", type=int, default=1, help="Row multiplication factor tag (metadata only)")
    ap.add_argument("--poll-interval", type=float, default=0.5, help="RSS poll interval in seconds")
    ap.add_argument("--check-install", action="store_true", help="Run uv sync only; classify failure mode")
    ap.add_argument("--compare-to", type=Path, help="Baseline JSONL to diff output checksums against")
    ap.add_argument(
        "--only",
        choices=sorted(STAGE_FLAGS.keys()),
        help="Run only one pipeline stage (validate | tableone | ward | ecdf). "
             "Default runs the full --get-ecdf --ward --visualize pipeline.",
    )
    args = ap.parse_args()

    host = get_host_fingerprint()
    git_commit = get_git_commit()

    if args.check_install:
        result = check_install(args.label)
        row = {
            "kind": "install",
            "label": args.label,
            "host": host,
            "git_commit": git_commit,
            **result,
        }
        emit_row(row)
        print_human_summary(row)
        return 0 if result["classification"] == "ok" else 1

    effective_tables_path = (
        args.data if args.data is not None
        else Path(json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["tables_path"])
    )
    with temp_tables_path(args.data):
        pipeline = run_pipeline(args.label, args.data, args.poll_interval, only_stage=args.only)

    polars = get_polars_info()
    if pipeline.get("pss_available"):
        tree_peak = pipeline.get("peak_pss_sum_mb")
    elif not pipeline.get("uss_unavailable"):
        tree_peak = pipeline.get("peak_uss_sum_mb")
    else:
        tree_peak = pipeline.get("peak_rss_sum_mb")
    ward_verdict = infer_ward_isolation(pipeline["stage_table"], tree_peak or 0.0)

    row = {
        "kind": "runtime",
        "label": args.label,
        "only_stage": args.only,
        "host": host,
        "git_commit": git_commit,
        "polars": polars,
        "tables_path": str(effective_tables_path),
        "tables_path_mb": directory_size_bytes(effective_tables_path) / 1_048_576,
        "factor": args.factor,
        "ward_isolation_inference": ward_verdict,
        **pipeline,
    }

    if args.compare_to is not None:
        baseline = load_latest_runtime_row(args.compare_to)
        if baseline is None:
            print(f"warning: no runtime row in {args.compare_to}; skipping divergence check", file=sys.stderr)
        else:
            row["diff_vs_baseline"] = compare_checksums(baseline, row)

    emit_row(row)
    print_human_summary(row)
    return 0 if row["exit_code"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
