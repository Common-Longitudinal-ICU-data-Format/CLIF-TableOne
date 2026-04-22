"""
Memory Monitor

Shared utility for tracking process RSS across pipeline steps.
Used by both TableOneRunner and ECDFRunner.

Also provides an opt-in per-phase profiler (see ``start_phase`` /
``end_phase``) that streams each phase's peak RSS to a CSV on disk.
Rows are flushed after every phase so that if the process is killed
(e.g. by the OS OOM-killer), prior phases' data survives on disk.
"""

import csv
import os
import threading
import time
import psutil


class MemoryMonitor:
    """Monitor memory usage during script execution."""

    def __init__(self, csv_path=None, sample_interval=0.5):
        """
        Parameters
        ----------
        csv_path : str or pathlib.Path, optional
            If given, ``start_phase``/``end_phase`` will stream rows to
            this CSV. Parent directory is created if missing.
        sample_interval : float
            Background sampler interval in seconds. A daemon thread polls
            RSS at this rate so per-phase peaks are captured without
            needing invasive sampling calls inside the pipeline code.
        """
        self.process = psutil.Process()
        self.start_time = time.time()
        self.start_memory = self.get_memory_mb()
        self.peak_memory = self.start_memory
        self.checkpoints = []

        # Per-phase profiler state (opt-in via csv_path)
        self._csv_path = str(csv_path) if csv_path is not None else None
        self._csv_fp = None
        self._csv_writer = None
        self._phase_lock = threading.Lock()
        self._current_phase = None  # dict: name, start_time, start_rss, peak_rss
        self._sampler_stop = threading.Event()
        self._sampler_thread = None
        self._sample_interval = sample_interval
        if self._csv_path:
            self._init_csv()
            self._start_sampler()

    # --- basic API (unchanged) -------------------------------------------

    def get_memory_mb(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def checkpoint(self, label):
        """Record a memory checkpoint."""
        current_memory = self.get_memory_mb()
        elapsed_time = time.time() - self.start_time

        if current_memory > self.peak_memory:
            self.peak_memory = current_memory

        self.checkpoints.append({
            'label': label,
            'memory_mb': current_memory,
            'peak_mb': self.peak_memory,
            'elapsed_sec': elapsed_time
        })

        print(f"  [{label}] Memory: {current_memory:.1f} MB | Peak: {self.peak_memory:.1f} MB | Time: {elapsed_time:.1f}s")

    def get_summary(self):
        """Get memory usage summary."""
        end_memory = self.get_memory_mb()
        total_time = time.time() - self.start_time

        return {
            'start_memory_mb': self.start_memory,
            'end_memory_mb': end_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': end_memory - self.start_memory,
            'total_time_sec': total_time,
            'checkpoints': self.checkpoints
        }

    # --- per-phase profiler (opt-in) -------------------------------------

    def _init_csv(self):
        os.makedirs(os.path.dirname(self._csv_path) or '.', exist_ok=True)
        # Append so multiple phases land in one file across a run; write
        # header only if the file is new.
        new_file = not os.path.exists(self._csv_path) or os.path.getsize(self._csv_path) == 0
        self._csv_fp = open(self._csv_path, 'a', encoding='utf-8', newline='')
        self._csv_writer = csv.writer(self._csv_fp)
        if new_file:
            self._csv_writer.writerow([
                'phase', 'start_iso', 'duration_sec',
                'start_rss_mb', 'peak_rss_mb', 'end_rss_mb', 'delta_rss_mb',
                'available_before_mb', 'available_after_mb',
            ])
            self._csv_fp.flush()

    def _start_sampler(self):
        def _run():
            while not self._sampler_stop.wait(self._sample_interval):
                try:
                    rss = self.get_memory_mb()
                except psutil.Error:
                    continue
                if rss > self.peak_memory:
                    self.peak_memory = rss
                with self._phase_lock:
                    if self._current_phase is not None and rss > self._current_phase['peak_rss']:
                        self._current_phase['peak_rss'] = rss

        self._sampler_thread = threading.Thread(target=_run, daemon=True, name='mem-sampler')
        self._sampler_thread.start()

    def start_phase(self, name):
        """Begin a named phase; peak RSS will be tracked until end_phase()."""
        if self._csv_writer is None:
            return  # profiler disabled
        now_rss = self.get_memory_mb()
        with self._phase_lock:
            self._current_phase = {
                'name': name,
                'start_time': time.time(),
                'start_iso': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'start_rss': now_rss,
                'peak_rss': now_rss,
                'available_before_mb': psutil.virtual_memory().available / 1024 / 1024,
            }

    def end_phase(self):
        """End the current phase and flush a row to CSV."""
        if self._csv_writer is None:
            return
        end_rss = self.get_memory_mb()
        end_time = time.time()
        with self._phase_lock:
            phase = self._current_phase
            self._current_phase = None
        if phase is None:
            return
        row = [
            phase['name'],
            phase['start_iso'],
            f"{end_time - phase['start_time']:.2f}",
            f"{phase['start_rss']:.1f}",
            f"{max(phase['peak_rss'], end_rss):.1f}",
            f"{end_rss:.1f}",
            f"{end_rss - phase['start_rss']:+.1f}",
            f"{phase['available_before_mb']:.1f}",
            f"{psutil.virtual_memory().available / 1024 / 1024:.1f}",
        ]
        self._csv_writer.writerow(row)
        # Flush immediately so a later SIGKILL doesn't drop completed rows
        self._csv_fp.flush()
        try:
            os.fsync(self._csv_fp.fileno())
        except OSError:
            pass

    def close(self):
        """Stop the sampler thread and close the CSV."""
        self._sampler_stop.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=2.0)
            self._sampler_thread = None
        if self._csv_fp is not None:
            try:
                self._csv_fp.close()
            except Exception:
                pass
            self._csv_fp = None
            self._csv_writer = None
