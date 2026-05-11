"""
Memory Monitor

Shared utility for tracking process RSS across pipeline steps.
Used by both TableOneRunner and ECDFRunner.
"""

import time
import psutil


class MemoryMonitor:
    """Monitor memory usage during script execution."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.start_memory = self.get_memory_mb()
        self.peak_memory = self.start_memory
        self.checkpoints = []

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
