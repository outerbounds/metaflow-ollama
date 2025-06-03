from metaflow import FlowSpec, step, ollama, kubernetes, pypi
from metaflow.profilers import gpu_profile
import time
import os
import subprocess
import psutil
import json
import signal
from contextlib import contextmanager
from datetime import datetime


@contextmanager
def timeout(duration):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Request timed out after {duration} seconds")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)

    try:
        yield
    finally:
        # Reset the alarm and handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class DeadlockRootCauseAnalysis(FlowSpec):

    @pypi(packages={"ollama": "0.5.1", "psutil": "5.9.5"})
    @kubernetes(
        cpu=32, gpu=4, memory=164_000, shared_memory=128_000, compute_pool="a10g4x"
    )
    @gpu_profile(interval=1)
    @ollama(
        models=["qwen3:0.6b"],  # Use fast model to hit deadlock quickly
        debug=True,
        circuit_breaker_config={
            "failure_threshold": 3,
            "recovery_timeout": 30,
            "reset_timeout": 60,
        },
    )
    @step
    def start(self):
        from ollama import chat
        import gc

        print("[@debug] Starting deadlock root cause analysis...")
        print(
            "[@debug] Target: Trigger deadlock around 650 requests and capture diagnostics"
        )

        # Initialize monitoring
        self.diagnostics = []
        self.resource_snapshots = []

        start_time = time.time()
        request_count = 0
        successful_requests = 0

        # Capture initial state
        self._capture_system_snapshot(request_count, "initial")

        # Run until deadlock or 800 requests (past typical deadlock point)
        max_requests = 800

        while request_count < max_requests:
            try:
                request_start = time.time()
                request_count += 1

                # Add timeout to prevent infinite hang
                with timeout(30):  # 30 second timeout per request
                    response = chat(
                        model="qwen3:0.6b",
                        messages=[
                            {
                                "role": "user",
                                "content": f"Request {request_count}: Count from 1 to 5.",
                            }
                        ],
                        stream=False,
                        options={"max_tokens": 50},
                    )

                successful_requests += 1
                inference_time = time.time() - request_start

                # Capture detailed snapshots every 50 requests
                if request_count % 50 == 0:
                    elapsed_minutes = (time.time() - start_time) / 60
                    print(
                        f"[@debug] Progress: {request_count} requests in {elapsed_minutes:.1f} minutes"
                    )
                    self._capture_system_snapshot(
                        request_count, f"progress_{request_count}"
                    )

                # Intensive monitoring approaching deadlock zone (600-700)
                if 600 <= request_count <= 700 and request_count % 10 == 0:
                    self._capture_detailed_snapshot(request_count, inference_time)

                # Very short delay to stress the system
                time.sleep(0.1)

            except (Exception, TimeoutError) as e:
                error_time = time.time()
                print(f"[@debug] DEADLOCK DETECTED at request {request_count}!")
                print(f"[@debug] Error: {e}")
                print(f"[@debug] Error type: {type(e)}")

                # Capture deadlock state immediately
                deadlock_snapshot = self._capture_deadlock_state(request_count, str(e))

                # Wait a moment then capture "during deadlock" state
                time.sleep(5)
                self._capture_system_snapshot(request_count, "during_deadlock")

                # Try to capture Ollama server logs
                self._capture_ollama_logs()

                print(
                    f"[@debug] Deadlock occurred at request {request_count} (expected ~650)"
                )
                print(f"[@debug] Total successful requests: {successful_requests}")
                break

        # Final analysis
        total_duration = time.time() - start_time

        self.results = {
            "total_requests": request_count,
            "successful_requests": successful_requests,
            "deadlock_occurred": request_count < max_requests,
            "deadlock_at_request": (
                request_count if request_count < max_requests else None
            ),
            "duration_minutes": total_duration / 60,
            "diagnostics": self.diagnostics,
            "resource_snapshots": self.resource_snapshots,
        }

        self._analyze_resource_trends()

        print(f"\n[@debug] ========== ROOT CAUSE ANALYSIS COMPLETE ==========")
        print(f"[@debug] Deadlock occurred: {self.results['deadlock_occurred']}")
        if self.results["deadlock_occurred"]:
            print(
                f"[@debug] Deadlock at request: {self.results['deadlock_at_request']}"
            )
        print(f"[@debug] Total duration: {total_duration/60:.1f} minutes")
        print(f"[@debug] Diagnostics captured: {len(self.diagnostics)} snapshots")
        print(f"[@debug] Resource snapshots: {len(self.resource_snapshots)} points")

        self.next(self.end)

    def _capture_system_snapshot(self, request_count, snapshot_type):
        """Capture comprehensive system resource snapshot"""
        try:
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "request_count": request_count,
                "type": snapshot_type,
                "memory": self._get_memory_info(),
                "gpu": self._get_gpu_info(),
                "processes": self._get_process_info(),
                "disk": self._get_disk_info(),
                "network": self._get_network_info(),
                "file_descriptors": self._get_fd_info(),
            }

            self.resource_snapshots.append(snapshot)

            if (
                snapshot_type in ["initial", "during_deadlock"]
                or request_count % 100 == 0
            ):
                print(
                    f"[@debug] Snapshot {snapshot_type}: RAM {snapshot['memory']['percent']:.1f}%, "
                    f"GPU {snapshot['gpu']['utilization']}%, "
                    f"Ollama processes: {len(snapshot['processes']['ollama'])}"
                )

        except Exception as e:
            print(f"[@debug] Error capturing snapshot: {e}")

    def _capture_detailed_snapshot(self, request_count, inference_time):
        """Capture detailed snapshot during critical zone (600-700 requests)"""
        try:
            detailed = {
                "request_count": request_count,
                "inference_time": inference_time,
                "timestamp": datetime.now().isoformat(),
                "memory_detail": psutil.virtual_memory()._asdict(),
                "gpu_memory": self._get_detailed_gpu_memory(),
                "ollama_processes": self._get_ollama_process_details(),
                "open_files": self._count_open_files(),
                "tcp_connections": self._count_tcp_connections(),
            }

            self.diagnostics.append(detailed)

            print(
                f"[@debug] Detailed snapshot {request_count}: "
                f"RAM {detailed['memory_detail']['percent']:.1f}%, "
                f"Inference {inference_time:.2f}s, "
                f"Open files: {detailed['open_files']}"
            )

        except Exception as e:
            print(f"[@debug] Error in detailed snapshot: {e}")

    def _capture_deadlock_state(self, request_count, error_msg):
        """Capture comprehensive state when deadlock occurs"""
        print(f"[@debug] Capturing deadlock state...")

        deadlock_state = {
            "request_count": request_count,
            "error_message": error_msg,
            "timestamp": datetime.now().isoformat(),
            "system_load": psutil.getloadavg(),
            "memory_full": psutil.virtual_memory()._asdict(),
            "swap_usage": psutil.swap_memory()._asdict(),
            "gpu_full": self._get_detailed_gpu_memory(),
            "disk_usage": psutil.disk_usage("/")._asdict(),
            "process_tree": self._get_full_process_tree(),
            "network_connections": self._get_all_connections(),
            "file_descriptor_limits": self._get_fd_limits(),
        }

        self.diagnostics.append({"type": "deadlock_state", "data": deadlock_state})
        return deadlock_state

    def _get_memory_info(self):
        """Get memory usage information"""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percent": mem.percent,
        }

    def _get_gpu_info(self):
        """Get GPU utilization and memory info"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                gpus = []
                for line in lines:
                    parts = line.split(", ")
                    if len(parts) == 3:
                        gpus.append(
                            {
                                "utilization": int(parts[0]),
                                "memory_used_mb": int(parts[1]),
                                "memory_total_mb": int(parts[2]),
                            }
                        )
                return {
                    "gpus": gpus,
                    "utilization": gpus[0]["utilization"] if gpus else 0,
                }
            else:
                return {"error": "nvidia-smi failed"}
        except Exception as e:
            return {"error": str(e)}

    def _get_detailed_gpu_memory(self):
        """Get detailed GPU memory breakdown"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                detailed = []
                for i, line in enumerate(lines):
                    parts = line.split(", ")
                    if len(parts) == 3:
                        detailed.append(
                            {
                                "gpu_id": i,
                                "used_mb": int(parts[0]),
                                "free_mb": int(parts[1]),
                                "total_mb": int(parts[2]),
                            }
                        )
                return detailed
            else:
                return [{"error": "nvidia-smi failed"}]
        except Exception as e:
            return [{"error": str(e)}]

    def _get_process_info(self):
        """Get information about running processes"""
        ollama_processes = []
        all_processes = []

        for proc in psutil.process_iter(["pid", "name", "memory_info", "cpu_percent"]):
            try:
                info = proc.info
                all_processes.append(info)
                if "ollama" in info["name"].lower():
                    ollama_processes.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return {
            "ollama": ollama_processes,
            "total_processes": len(all_processes),
            "high_memory_processes": [
                p
                for p in all_processes
                if p["memory_info"] and p["memory_info"].rss > 1024 * 1024 * 100
            ],  # >100MB
        }

    def _get_disk_info(self):
        """Get disk usage information"""
        try:
            usage = psutil.disk_usage("/")
            return {
                "total_gb": usage.total / (1024**3),
                "used_gb": usage.used / (1024**3),
                "free_gb": usage.free / (1024**3),
                "percent": (usage.used / usage.total) * 100,
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_network_info(self):
        """Get network connection information"""
        try:
            connections = psutil.net_connections()
            return {
                "total_connections": len(connections),
                "established": len(
                    [c for c in connections if c.status == "ESTABLISHED"]
                ),
                "listen": len([c for c in connections if c.status == "LISTEN"]),
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_fd_info(self):
        """Get file descriptor information"""
        try:
            # Count open file descriptors for the current process
            import os

            proc_fd_dir = f"/proc/{os.getpid()}/fd"
            if os.path.exists(proc_fd_dir):
                fd_count = len(os.listdir(proc_fd_dir))
                return {"open_fds": fd_count}
            else:
                return {"error": "Cannot access /proc/pid/fd"}
        except Exception as e:
            return {"error": str(e)}

    def _get_ollama_process_details(self):
        """Get detailed information about Ollama processes"""
        ollama_details = []
        for proc in psutil.process_iter(
            ["pid", "name", "memory_info", "cpu_percent", "num_threads", "open_files"]
        ):
            try:
                if "ollama" in proc.info["name"].lower():
                    details = proc.info.copy()
                    details["memory_mb"] = proc.info["memory_info"].rss / (1024 * 1024)
                    details["num_open_files"] = len(proc.info.get("open_files", []))
                    ollama_details.append(details)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return ollama_details

    def _count_open_files(self):
        """Count total open files across all processes"""
        try:
            total = 0
            for proc in psutil.process_iter():
                try:
                    total += proc.num_fds()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total
        except Exception as e:
            return -1

    def _count_tcp_connections(self):
        """Count TCP connections"""
        try:
            connections = psutil.net_connections(kind="tcp")
            return len(connections)
        except Exception as e:
            return -1

    def _get_full_process_tree(self):
        """Get full process tree information"""
        try:
            processes = []
            for proc in psutil.process_iter(
                ["pid", "ppid", "name", "memory_info", "num_threads"]
            ):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return processes
        except Exception as e:
            return []

    def _get_all_connections(self):
        """Get all network connections"""
        try:
            return [conn._asdict() for conn in psutil.net_connections()]
        except Exception as e:
            return []

    def _get_fd_limits(self):
        """Get file descriptor limits"""
        try:
            import resource

            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            return {"soft_limit": soft, "hard_limit": hard}
        except Exception as e:
            return {"error": str(e)}

    def _capture_ollama_logs(self):
        """Attempt to capture Ollama server logs"""
        try:
            # Try to get recent ollama logs
            result = subprocess.run(
                ["journalctl", "-u", "ollama", "--since", "5 minutes ago"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                self.diagnostics.append({"type": "ollama_logs", "data": result.stdout})
            else:
                print(f"[@debug] Could not capture ollama logs: {result.stderr}")
        except Exception as e:
            print(f"[@debug] Error capturing logs: {e}")

    def _analyze_resource_trends(self):
        """Analyze resource usage trends leading to deadlock"""
        if len(self.resource_snapshots) < 2:
            return

        print(f"\n[@debug] ========== RESOURCE TREND ANALYSIS ==========")

        # Memory trend
        memory_usage = [s["memory"]["percent"] for s in self.resource_snapshots]
        memory_growth = (
            memory_usage[-1] - memory_usage[0] if len(memory_usage) > 1 else 0
        )
        print(
            f"[@debug] Memory usage: {memory_usage[0]:.1f}% → {memory_usage[-1]:.1f}% (Δ{memory_growth:+.1f}%)"
        )

        # Process count trend
        process_counts = [
            len(s["processes"]["ollama"]) for s in self.resource_snapshots
        ]
        if process_counts:
            print(
                f"[@debug] Ollama processes: {process_counts[0]} → {process_counts[-1]} (Δ{process_counts[-1] - process_counts[0]:+d})"
            )

        # Detailed diagnostics trend
        if self.diagnostics:
            detailed_snapshots = [d for d in self.diagnostics if "memory_detail" in d]
            if len(detailed_snapshots) > 1:
                first = detailed_snapshots[0]
                last = detailed_snapshots[-1]

                ram_growth = (
                    last["memory_detail"]["percent"] - first["memory_detail"]["percent"]
                )
                fd_growth = last.get("open_files", 0) - first.get("open_files", 0)

                print(f"[@debug] Detailed RAM growth: {ram_growth:+.2f}%")
                print(f"[@debug] Open files growth: {fd_growth:+d}")

                if ram_growth > 5:
                    print(f"[@debug] ⚠️  MEMORY LEAK DETECTED: {ram_growth:.1f}% growth")
                if fd_growth > 100:
                    print(
                        f"[@debug] ⚠️  FILE DESCRIPTOR LEAK DETECTED: {fd_growth} growth"
                    )

    @step
    def end(self):
        pass


if __name__ == "__main__":
    DeadlockRootCauseAnalysis()
