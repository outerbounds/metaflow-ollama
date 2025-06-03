import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import socket
import sys
import os
import functools
import json
import requests
from enum import Enum
import threading
from datetime import datetime

from .constants import OLLAMA_SUFFIX
from .exceptions import (
    EmptyOllamaManifestCacheException,
    EmptyOllamaBlobCacheException,
    UnspecifiedRemoteStorageRootException,
    CircuitBreakerInvalidStateException
)


class ProcessStatus:
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    SUCCESSFUL = "SUCCESSFUL"


class CircuitBreakerState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:

    """
    The CircuitBreaker is used in conjunction with the OllamaHealthChecker and OllamaRequestInterceptor 
    to make user commands more robust. 
    
    If commands hit the server at a time when the server isn't healthy, as detected by the OllamaHealthChecker, we want:
        1. Avoid ollama deadlocking as observed in development when under sustained load. It is possible
            that has to do with an aspect of this integration. We observed no resource leaks and 
            consistent deadlocking around 400-700 single-thread requests consistently.  
        2. Reset the server to a healthy state once deadlock (1) is detected.
        3. Send Runtime errors back to user code so they can decide whether to error out or retry the request. 

    This class is responsbile for
        - updating the circuit breaker state,
        - block request flow when circuit state is OPEN,
        - allow request flow when circuit state is CLOSED,
        - facilitate retry execution logic,
        - send logging events to a monitoring dashboard contained in a Metaflow @card.
    """
    def __init__(
        self,
        failure_threshold,
        recovery_timeout,
        reset_timeout,
        debug=False,
        status_card=None,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.reset_timeout = reset_timeout
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_open_time = None
        self.debug = debug
        self.status_card = status_card
        self.lock = threading.Lock()
        self.request_count = 0  # Track total requests for pattern detection

        if self.debug:
            print(
                f"[@ollama] CircuitBreaker initialized: threshold={failure_threshold}, recovery={recovery_timeout}, reset={reset_timeout}"
            )

    def _log_state_change(self, new_state):
        if self.debug:
            print(
                f"[@ollama] Circuit Breaker state change: {self.state.value} -> {new_state.value}"
            )
        self.state = new_state
        self._update_status_card()

    def _update_status_card(self):
        """Update the status card with current circuit breaker state"""
        if self.status_card:
            self.status_card.update_status(
                "circuit_breaker",
                {
                    "state": self.state.value,
                    "failure_count": self.failure_count,
                    "last_failure_time": self.last_failure_time,
                    "last_open_time": self.last_open_time,
                },
            )

    def record_success(self):
        with self.lock:
            self.request_count += 1
            if self.state == CircuitBreakerState.HALF_OPEN:
                self._log_state_change(CircuitBreakerState.CLOSED)
                self.failure_count = 0
            elif self.state == CircuitBreakerState.OPEN:
                # Allow transition to HALF_OPEN on success - server might have recovered
                self._log_state_change(CircuitBreakerState.HALF_OPEN)
                if self.debug:
                    print(
                        f"[@ollama] Success recorded while circuit OPEN. Transitioning to HALF_OPEN for testing."
                    )
            self.failure_count = 0
            self.last_failure_time = None

            if self.debug and self.request_count % 100 == 0:
                print(f"[@ollama] Request count: {self.request_count}")

            self._update_status_card()

    def record_failure(self):
        with self.lock:
            self.request_count += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            if (
                self.failure_count >= self.failure_threshold
                and self.state == CircuitBreakerState.CLOSED
            ):
                self._log_state_change(CircuitBreakerState.OPEN)
                self.last_open_time = time.time()
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # If we fail while testing recovery, go back to OPEN
                self._log_state_change(CircuitBreakerState.OPEN)
                self.last_open_time = time.time()
            if self.debug:
                print(
                    f"[@ollama] Failure recorded. Count: {self.failure_count}, State: {self.state.value}, Total requests: {self.request_count}"
                )
            self._update_status_card()

    def should_attempt_reset(self):
        """Check if we should attempt to reset/restart Ollama based on reset_timeout"""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN and self.last_open_time:
                elapsed_time = time.time() - self.last_open_time
                return elapsed_time > self.reset_timeout
            return False

    def is_request_allowed(self):
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                elapsed_time = time.time() - self.last_open_time
                if elapsed_time > self.recovery_timeout:
                    self._log_state_change(CircuitBreakerState.HALF_OPEN)
                    if self.debug:
                        print(
                            f"[@ollama] Circuit Breaker transitioning to HALF_OPEN after {elapsed_time:.1f}s."
                        )
                    return True  # Allow a single request to test recovery
                else:
                    if self.debug:
                        print(
                            f"[@ollama] Circuit Breaker is OPEN. Not allowing request. Time until HALF_OPEN: {self.recovery_timeout - elapsed_time:.1f}s"
                        )
                    return False
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # In HALF_OPEN, be more restrictive - only allow one request at a time
                if self.debug:
                    print(
                        f"[@ollama] Circuit Breaker is HALF_OPEN. Allowing request to test recovery."
                    )
                return True
            elif self.state == CircuitBreakerState.CLOSED:
                return True
            else:
                raise CircuitBreakerInvalidStateException(f'Circuit breaker is in an invalid state {self.state}.')

    def get_status(self):
        with self.lock:
            status = {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "last_failure_time": self.last_failure_time,
                "last_open_time": self.last_open_time,
            }
            if self.debug:
                print(f"[@ollama] Circuit Breaker status: {status}")
            return status


class TimeoutCommand:
    def __init__(self, command, timeout, debug=False, **kwargs):
        self.command = command
        self.timeout = timeout
        self.debug = debug
        self.input_data = kwargs.pop("input", None)
        self.kwargs = kwargs

    def run(self):
        if self.debug:
            print(
                f"[@ollama] Executing command with timeout {self.timeout}s: {' '.join(self.command)}"
            )
        try:
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **self.kwargs,
            )
            stdout, stderr = process.communicate(
                input=self.input_data, timeout=self.timeout
            )
            return process.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            if self.debug:
                print(
                    f"[@ollama] Command timed out after {self.timeout}s: {' '.join(self.command)}"
                )
            process.kill()
            stdout, stderr = process.communicate()
            return (
                124,
                stdout,
                stderr,
            )  # 124 is the standard exit code for `timeout` command
        except Exception as e:
            if self.debug:
                print(
                    f"[@ollama] Error executing command {' '.join(self.command)}: {e}"
                )
            return 1, "", str(e)


class OllamaHealthChecker:

    """
    The OllamaHealthChecker is assigned a circuit breaker to write health to.
    These health records are crucial for driving the CircuitBreaker system that keeps the Ollama server alive.

    This is kind of the god class. 
    It sees everything about the system behind @ollama including the circuit_breaker amd the ollama_manager.
    """
    def __init__(self, ollama_url, circuit_breaker, ollama_manager, debug=False):
        self.ollama_url = ollama_url
        self.circuit_breaker = circuit_breaker
        self.ollama_manager = ollama_manager
        self.debug = debug
        self._stop_event = threading.Event()
        self._thread = None
        self._interval = 30

    def _check_health(self):
        try:
            health_timeout = self.ollama_manager.timeouts.get("health_check", 5)
            if self.debug:
                print(f"[@ollama] Health check: Pinging {self.ollama_url}/api/tags")
            response = requests.get(
                f"{self.ollama_url}/api/tags", timeout=health_timeout
            )
            if response.status_code == 200:
                self.circuit_breaker.record_success()
                self._update_server_health_status("Healthy")
                if self.debug:
                    print(
                        f"[@ollama] Health check successful. Circuit state: {self.circuit_breaker.state.value}"
                    )
                return True
            else:
                if self.debug:
                    print(
                        f"[@ollama] Health check failed. Status code: {response.status_code}. Circuit state: {self.circuit_breaker.state.value}"
                    )
                self.circuit_breaker.record_failure()
                self._update_server_health_status(
                    f"Unhealthy (HTTP {response.status_code})"
                )
                return False
        except requests.exceptions.RequestException as e:
            if self.debug:
                print(
                    f"[@ollama] Health check exception: {e}. Circuit state: {self.circuit_breaker.state.value}"
                )
            self.circuit_breaker.record_failure()
            self._update_server_health_status(f"Unhealthy ({str(e)[:50]})")
            return False

    def _update_server_health_status(self, status):
        """Update server health status in the status card"""
        if self.ollama_manager.status_card:
            self.ollama_manager.status_card.update_status(
                "server", {"health_status": status, "last_health_check": datetime.now()}
            )

    def _run_health_check_loop(self):
        while not self._stop_event.is_set():
            self._check_health()

            if self.circuit_breaker.should_attempt_reset():
                try:
                    if self.debug:
                        print(
                            "[@ollama] Circuit breaker reset timeout reached. Attempting restart..."
                        )
                    restart_success = self.ollama_manager._attempt_ollama_restart()
                    if restart_success:
                        if self.debug:
                            print("[@ollama] Restart successful via health checker")
                    else:
                        if self.debug:
                            print("[@ollama] Restart failed via health checker")
                except Exception as e:
                    if self.debug:
                        print(
                            f"[@ollama] Error during health checker restart attempt: {e}"
                        )

            self._stop_event.wait(self._interval)

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_health_check_loop, daemon=True
            )
            self._thread.start()
            if self.debug:
                print("[@ollama] OllamaHealthChecker started.")

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=self._interval + 1)
            if self.debug:
                print("[@ollama] OllamaHealthChecker stopped.")


class OllamaRequestInterceptor:

    """
    This class monkey patches the following ollama methods:
    - chat
    - generate
    - embeddings

    Why? 

    So we can wrap ollama API requests with a "circuit breaker" system.
    The system ensures the API server is staying healthy and does not dead lock the system.

    When?

    It is instantiated in the @ollama StepDecorator's task_decorate before the user's code runs.
    Then, after the user's code runs, it cleans up.
    """

    def __init__(self, circuit_breaker, debug=False):
        self.circuit_breaker = circuit_breaker
        self.debug = debug
        self.original_methods = {}
        self._protection_installed = False

    def install_protection(self):
        """Install request protection by monkey-patching the ollama package"""
        if self._protection_installed:
            return

        try:
            import ollama  # Import the actual ollama package

            self.original_methods = {
                "chat": getattr(ollama, "chat", None),
                "generate": getattr(ollama, "generate", None),
                "embeddings": getattr(ollama, "embeddings", None),
            }

            if hasattr(ollama, "chat"):
                ollama.chat = self._protected_chat
            if hasattr(ollama, "generate"):
                ollama.generate = self._protected_generate
            if hasattr(ollama, "embeddings"):
                ollama.embeddings = self._protected_embeddings

            self._protection_installed = True
            if self.debug:
                print(
                    "[@ollama] Request protection installed on ollama package methods"
                )

        except ImportError:
            if self.debug:
                print(
                    "[@ollama] Warning: Could not import ollama package for request protection"
                )
        except Exception as e:
            if self.debug:
                print(f"[@ollama] Error installing request protection: {e}")

    def remove_protection(self):
        """Remove request protection by restoring original methods"""
        if not self._protection_installed:
            return

        try:
            import ollama

            for method_name, original_method in self.original_methods.items():
                if original_method is not None and hasattr(ollama, method_name):
                    setattr(ollama, method_name, original_method)

            self._protection_installed = False
            if self.debug:
                print("[@ollama] Request protection removed")

        except Exception as e:
            if self.debug:
                print(f"[@ollama] Error removing request protection: {e}")

    def _protected_chat(self, *args, **kwargs):
        if not self.circuit_breaker.is_request_allowed():
            raise RuntimeError(
                f"Ollama server is currently unavailable. Circuit Breaker is {self.circuit_breaker.state.value}. "
                "Please wait or check Ollama server status. "
                f"Current status: {self.circuit_breaker.get_status()}"
            )
        try:
            if self.debug:
                # Debug: log model being used in request
                model_name = kwargs.get("model", "unknown")
                if args and isinstance(args[0], dict) and "model" in args[0]:
                    model_name = args[0]["model"]
                print(f"[@ollama] DEBUG: Making chat request with model: {model_name}")
                print(f"[@ollama] DEBUG: Request args: {args}")
                print(f"[@ollama] DEBUG: Request kwargs keys: {list(kwargs.keys())}")

            result = self.original_methods["chat"](*args, **kwargs)
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            if self.debug:
                print(f"[@ollama] Protected chat call failed: {e}")
                print(f"[@ollama] DEBUG: Exception type: {type(e)}")
            self.circuit_breaker.record_failure()
            raise

    def _protected_generate(self, *args, **kwargs):
        if not self.circuit_breaker.is_request_allowed():
            raise RuntimeError(
                f"Ollama server is currently unavailable. Circuit Breaker is {self.circuit_breaker.state.value}. "
                "Please wait or check Ollama server status. "
                f"Current status: {self.circuit_breaker.get_status()}"
            )
        try:
            result = self.original_methods["generate"](*args, **kwargs)
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            if self.debug:
                print(f"[@ollama] Protected generate call failed: {e}")
            self.circuit_breaker.record_failure()
            raise

    def _protected_embeddings(self, *args, **kwargs):
        if not self.circuit_breaker.is_request_allowed():
            raise RuntimeError(
                f"Ollama server is currently unavailable. Circuit Breaker is {self.circuit_breaker.state.value}. "
                "Please wait or check Ollama server status. "
                f"Current status: {self.circuit_breaker.get_status()}"
            )
        try:
            result = self.original_methods["embeddings"](*args, **kwargs)
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            if self.debug:
                print(f"[@ollama] Protected embeddings call failed: {e}")
            self.circuit_breaker.record_failure()
            raise


class OllamaManager:
    """
    A process manager for Ollama runtimes.
    Implements interface @ollama([models=...], ...) has a local, remote, or managed backend.
    """

    def __init__(
        self,
        models,
        backend="local",
        flow_datastore_backend=None,
        remote_storage_root=None,
        force_pull=False,
        cache_update_policy="auto",
        force_cache_update=False,
        debug=False,
        circuit_breaker_config=None,
        timeout_config=None,
        status_card=None,
    ):
        self.models = {}
        self.processes = {}
        self.flow_datastore_backend = flow_datastore_backend
        if self.flow_datastore_backend is not None:
            self.remote_storage_root = self.get_ollama_storage_root(
                self.flow_datastore_backend
            )
        elif remote_storage_root is not None:
            self.remote_storage_root = remote_storage_root
        else:
            raise UnspecifiedRemoteStorageRootException(
                "Can not determine the storage root, as both flow_datastore_backend and remote_storage_root arguments of OllamaManager are None."
            )
        self.force_pull = force_pull

        self.cache_update_policy = cache_update_policy
        if force_cache_update:
            self.cache_update_policy = "force"
        self.cache_status = {}

        self.debug = debug
        self.stats = {}
        self.storage_info = {}
        self.ollama_url = "http://localhost:11434" # Ollama API base URL - TODO: remote server.
        self.status_card = status_card
        self.initialization_start = time.time()

        if backend != "local":
            raise ValueError(
                "OllamaManager only supports the 'local' backend at this time."
            )

        if circuit_breaker_config is None:
            circuit_breaker_config = {
                "failure_threshold": 3,
                "recovery_timeout": 30,
                "reset_timeout": 60,
            }

        if timeout_config is None:
            timeout_config = {
                "pull": 600, 
                "stop": 30,
                "health_check": 5,
                "install": 60,  
                "server_startup": 300, 
            }
        self.timeouts = timeout_config
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_config.get("failure_threshold", 3),
            recovery_timeout=circuit_breaker_config.get("recovery_timeout", 30),
            reset_timeout=circuit_breaker_config.get("reset_timeout", 60),
            debug=self.debug,
            status_card=self.status_card,
        )
        self.health_checker = OllamaHealthChecker(
            self.ollama_url, self.circuit_breaker, self, self.debug
        )

        self._log_event("info", "Starting Ollama initialization")
        self._timeit(self._install_ollama, "install_ollama")
        self._timeit(self._launch_server, "launch_server")
        self.health_checker.start()
        self._collect_version_info()
        self._update_cache_status()

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._pull_model, m) for m in models]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    raise RuntimeError(f"Error pulling one or more models. {e}") from e
        self._update_cache_status()
        
        for m in models:
            f = functools.partial(self._run_model, m)
            self._timeit(f, f"model_{m.lower()}")

        total_init_time = time.time() - self.initialization_start
        self._update_performance("total_initialization_time", total_init_time)
        self._log_event(
            "success", f"Ollama initialization completed in {total_init_time:.1f}s"
        )

    def _collect_version_info(self):
        """Collect version information for Ollama system and Python client"""
        version_info = {}

        try:
            result = subprocess.run(
                ["ollama", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                version_line = result.stdout.strip()
                # Common formats: "ollama version 0.1.0", "0.1.0", "v0.1.0"
                if "version" in version_line.lower():
                    parts = version_line.lower().split("version")
                    if len(parts) > 1:
                        version_info["ollama_system"] = parts[1].strip()
                    else:
                        version_info["ollama_system"] = version_line
                elif version_line.startswith("v"):
                    version_info["ollama_system"] = version_line[
                        1:
                    ]
                else:
                    version_info["ollama_system"] = version_line
            else:
                version_info["ollama_system"] = "Unknown"
        except Exception as e:
            version_info["ollama_system"] = "Error detecting"
            if self.debug:
                print(f"[@ollama] Error getting system version: {e}")

        try:
            import ollama

            if hasattr(ollama, "__version__"):
                version_info["ollama_python"] = ollama.__version__
            else:
                try:
                    import pkg_resources

                    version_info["ollama_python"] = pkg_resources.get_distribution(
                        "ollama"
                    ).version
                except:
                    try:
                        from importlib import metadata

                        version_info["ollama_python"] = metadata.version("ollama")
                    except:
                        version_info["ollama_python"] = "Unknown"
        except ImportError:
            version_info["ollama_python"] = "Not installed"
        except Exception as e:
            version_info["ollama_python"] = "Error detecting"
            if self.debug:
                print(f"[@ollama] Error getting Python client version: {e}")

        if self.status_card:
            self.status_card.update_status("versions", version_info)
            self._log_event(
                "info",
                f"Versions: System {version_info.get('ollama_system', 'Unknown')}, Python {version_info.get('ollama_python', 'Unknown')}",
            )

    def _check_cache_exists(self, m):
        """Check if cache exists for the given model"""
        if self.local_datastore:
            return False

        if m not in self.storage_info:
            return False

        try:
            from metaflow import S3
            from metaflow.plugins.datatools.s3.s3 import MetaflowS3NotFound

            with S3() as s3:
                manifest_exists = s3.get(self.storage_info[m]["manifest_remote"]).exists

                if manifest_exists:
                    self.cache_status[m] = "exists"
                    self._update_cache_status()
                    return True
                else:
                    self.cache_status[m] = "missing"
                    self._update_cache_status()
                    return False

        except Exception as e:
            if self.debug:
                print(f"[@ollama {m}] Error checking cache existence: {e}")
            self.cache_status[m] = "error"
            self._update_cache_status()
            return False

    def _should_update_cache(self, m):
        """Determine if we should update cache for this model based on policy"""
        if self.cache_update_policy == "never":
            return False
        elif self.cache_update_policy == "force":
            return True
        elif self.cache_update_policy == "auto":
            cache_exists = self._check_cache_exists(m)
            return not cache_exists
        else:
            cache_exists = self._check_cache_exists(m)
            return not cache_exists

    def _log_event(self, event_type, message):
        """Log an event to the status card"""
        if self.status_card:
            self.status_card.add_event(event_type, message)
        if self.debug:
            print(f"[@ollama] {event_type.upper()}: {message}")

    def _update_server_status(self, status, **kwargs):
        """Update server status in the status card"""
        if self.status_card:
            update_data = {"status": status}
            update_data.update(kwargs)
            self.status_card.update_status("server", update_data)

    def _update_model_status(self, model_name, **kwargs):
        """Update model status in the status card"""
        if self.status_card:
            current_models = self.status_card.status_data.get("models", {})
            if model_name not in current_models:
                current_models[model_name] = {}
            current_models[model_name].update(kwargs)
            self.status_card.update_status("models", current_models)

    def _update_performance(self, metric, value):
        """Update performance metrics in the status card"""
        if self.status_card:
            self.status_card.update_status("performance", {metric: value})

    def _update_circuit_breaker_status(self):
        """Update circuit breaker status in the status card"""
        if self.status_card:
            cb_status = self.circuit_breaker.get_status()
            self.status_card.update_status("circuit_breaker", cb_status)

    def _update_cache_status(self):
        """Update cache status in the status card"""
        if self.status_card:
            self.status_card.update_status(
                "cache",
                {
                    "policy": self.cache_update_policy,
                    "model_status": self.cache_status.copy(),
                },
            )

    def _timeit(self, f, name):
        t0 = time.time()
        f()
        tf = time.time()
        duration = tf - t0
        self.stats[name] = {"process_runtime": duration}

        if name == "install_ollama":
            self._update_performance("install_time", duration)
        elif name == "launch_server":
            self._update_performance("server_startup_time", duration)

    def _is_port_open(self, host, port, timeout=1):
        """Check if a TCP port is open on a given host."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            try:
                sock.connect((host, port))
                return True
            except socket.error:
                return False

    def _install_ollama(self, max_retries=3):
        self._log_event("info", "Checking for existing Ollama installation")
        try:
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
            if result.returncode == 0:
                self._log_event("success", "Ollama is already installed")
                print("[@ollama] Ollama is already installed.")
                return
        except Exception as e:
            if self.debug:
                print(f"[@ollama] Did not find Ollama installation: {e}")
            if sys.platform == "darwin":
                raise RuntimeError(
                    "On macOS, please install Ollama manually from https://ollama.com/download."
                )

        self._log_event("info", "Installing Ollama...")
        if self.debug:
            print("[@ollama] Installing Ollama...")
        env = os.environ.copy()
        env["CURL_IPRESOLVE"] = "4"

        for attempt in range(max_retries):
            try:
                install_cmd = ["curl", "-fsSL", "https://ollama.com/install.sh"]
                curl_proc = subprocess.run(
                    install_cmd, capture_output=True, text=True, env=env, timeout=120
                )
                if curl_proc.returncode != 0:
                    raise RuntimeError(
                        f"Failed to download Ollama install script: stdout: {curl_proc.stdout}, stderr: {curl_proc.stderr}"
                    )
                sh_proc = subprocess.run(
                    ["sh"],
                    input=curl_proc.stdout,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=self.timeouts.get("install", 60),
                )
                if sh_proc.returncode != 0:
                    raise RuntimeError(
                        f"Ollama installation script failed: stdout: {sh_proc.stdout}, stderr: {sh_proc.stderr}"
                    )
                self._log_event("success", "Ollama installation completed successfully")
                if self.debug:
                    print("[@ollama] Ollama installed successfully.")
                break
            except Exception as e:
                self._log_event(
                    "warning", f"Installation attempt {attempt+1} failed: {str(e)}"
                )
                if self.debug:
                    print(f"[@ollama] Installation attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    self._log_event(
                        "error",
                        f"Ollama installation failed after {max_retries} attempts",
                    )
                    raise RuntimeError(
                        f"Error installing Ollama after {max_retries} attempts: {e}"
                    ) from e

    def _launch_server(self):
        """
        Start the Ollama server process and ensure it's running.
        """
        self._update_server_status("Starting")
        self._log_event("info", "Starting Ollama server...")

        try:
            print("[@ollama] Starting Ollama server...")
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.processes[process.pid] = {
                "p": process,
                "properties": {"type": "api-server", "error_details": None},
                "status": ProcessStatus.RUNNING,
            }

            if self.debug:
                print(f"[@ollama] Started server process with PID {process.pid}.")

            host, port = "127.0.0.1", 11434
            retries = 0
            max_retries = 10
            while (
                not self._is_port_open(host, port, timeout=1) and retries < max_retries
            ):
                if retries == 0:
                    print("[@ollama] Waiting for server to be ready...")
                elif retries % 3 == 0:
                    print(f"[@ollama] Still waiting... ({retries + 1}/{max_retries})")

                returncode = process.poll()
                if returncode is not None:
                    error_details = f"Return code: {returncode}"
                    self.processes[process.pid]["properties"][
                        "error_details"
                    ] = error_details
                    self.processes[process.pid]["status"] = ProcessStatus.FAILED
                    self._update_server_status("Failed", error_details=error_details)
                    self._log_event(
                        "error", f"Ollama server failed to start: {error_details}"
                    )
                    raise RuntimeError(
                        f"Ollama server failed to start. {error_details}"
                    )

                time.sleep(5)
                retries += 1

            if not self._is_port_open(host, port, timeout=1):
                error_details = (
                    f"Ollama server did not start listening on {host}:{port}"
                )
                self.processes[process.pid]["properties"][
                    "error_details"
                ] = error_details
                self.processes[process.pid]["status"] = ProcessStatus.FAILED
                self._update_server_status("Failed", error_details=error_details)
                self._log_event("error", f"Server startup timeout: {error_details}")
                raise RuntimeError(f"Ollama server failed to start. {error_details}")

            returncode = process.poll()
            if returncode is not None:
                error_details = f"Return code: {returncode}"
                self.processes[process.pid]["properties"][
                    "error_details"
                ] = error_details
                self.processes[process.pid]["status"] = ProcessStatus.FAILED
                self._update_server_status("Failed", error_details=error_details)
                self._log_event(
                    "error", f"Server process died unexpectedly: {error_details}"
                )
                raise RuntimeError(f"Ollama server failed to start. {error_details}")

            self._update_server_status("Running", uptime_start=datetime.now())
            self._log_event("success", "Ollama server is ready and listening")
            print("[@ollama] Server is ready.")

        except Exception as e:
            if "process" in locals() and process.pid in self.processes:
                self.processes[process.pid]["status"] = ProcessStatus.FAILED
                self.processes[process.pid]["properties"]["error_details"] = str(e)
            self._update_server_status("Failed", error_details=str(e))
            self._log_event("error", f"Error starting Ollama server: {str(e)}")
            raise RuntimeError(f"Error starting Ollama server: {e}") from e

    def _setup_storage(self, m):
        """
        Configure local and remote storage paths for an Ollama model.
        """
        ollama_model_name_components = m.split(":")
        if len(ollama_model_name_components) == 1:
            model_name = ollama_model_name_components[0]
            tag = "latest"
        elif len(ollama_model_name_components) == 2:
            model_name = ollama_model_name_components[0]
            tag = ollama_model_name_components[1]

        # Find where Ollama actually stores models.
        # This was driving me nuts so I went with the: 
        # "loop through everywhere you've found it before search strategy."
        possible_storage_roots = [
            os.environ.get("OLLAMA_MODELS"),
            "/usr/share/ollama/.ollama/models",
            os.path.expanduser("~/.ollama/models"),
            "/root/.ollama/models",
        ]

        ollama_local_storage_root = None
        for root in possible_storage_roots:
            if root and os.path.exists(root):
                ollama_local_storage_root = root
                break

        if not ollama_local_storage_root:
            # https://github.com/ollama/ollama/blob/main/docs/faq.md#where-are-models-stored
            if sys.platform.startswith("linux"):
                ollama_local_storage_root = "/usr/share/ollama/.ollama/models"
            elif sys.platform == "darwin":
                ollama_local_storage_root = os.path.expanduser("~/.ollama/models")

        if self.debug:
            print(
                f"[@ollama {m}] Using Ollama storage root: {ollama_local_storage_root}."
            )

        blob_local_path = os.path.join(ollama_local_storage_root, "blobs")
        manifest_base_path = os.path.join(
            ollama_local_storage_root,
            "manifests/registry.ollama.ai/library",
            model_name,
        )

        try:
            os.makedirs(blob_local_path, exist_ok=True)
            os.makedirs(manifest_base_path, exist_ok=True)
        except FileExistsError:
            pass

        if not self.local_datastore and self.remote_storage_root is not None:
            blob_remote_key = os.path.join(self.remote_storage_root, "blobs")
            manifest_remote_key = os.path.join(
                self.remote_storage_root,
                "manifests/registry.ollama.ai/library",
                model_name,
                tag,
            )
        else:
            blob_remote_key = None
            manifest_remote_key = None

        self.storage_info[m] = {
            "blob_local_root": blob_local_path,
            "blob_remote_root": blob_remote_key,
            "manifest_local": os.path.join(manifest_base_path, tag),
            "manifest_remote": manifest_remote_key,
            "manifest_content": None,
            "model_name": model_name,
            "tag": tag,
            "storage_root": ollama_local_storage_root,
        }

        if self.debug:
            print(f"[@ollama {m}] Storage paths configured.")

    def _fetch_manifest(self, m):
        """
        Load the manifest file and content, either from local storage or remote cache.
        """
        if self.debug:
            print(f"[@ollama {m}] Checking for cached manifest...")

        def _disk_to_memory():
            with open(self.storage_info[m]["manifest_local"], "r") as f:
                self.storage_info[m]["manifest_content"] = json.load(f)

        if os.path.exists(self.storage_info[m]["manifest_local"]):
            if self.storage_info[m]["manifest_content"] is None:
                _disk_to_memory()
            if self.debug:
                print(f"[@ollama {m}] Manifest found locally.")
        elif self.local_datastore:
            if self.debug:
                print(f"[@ollama {m}] No manifest found in local datastore.")
            return None
        else:
            from metaflow import S3
            from metaflow.plugins.datatools.s3.s3 import MetaflowS3NotFound

            try:
                with S3() as s3:
                    s3obj = s3.get(self.storage_info[m]["manifest_remote"])
                    if not s3obj.exists:
                        raise EmptyOllamaManifestCacheException(
                            f"No manifest in remote storage for model {m}"
                        )

                    if self.debug:
                        print(f"[@ollama {m}] Downloaded manifest from cache.")
                    os.rename(s3obj.path, self.storage_info[m]["manifest_local"])
                    _disk_to_memory()

                    if self.debug:
                        print(
                            f"[@ollama {m}] Manifest found in remote cache, downloaded locally."
                        )
            except (MetaflowS3NotFound, EmptyOllamaManifestCacheException):
                if self.debug:
                    print(
                        f"[@ollama {m}] No manifest found locally or in remote cache."
                    )
                return None

        return self.storage_info[m]["manifest_content"]

    def _fetch_blobs(self, m):
        """
        Fetch missing blobs from remote cache.
        """
        if self.debug:
            print(f"[@ollama {m}] Checking for cached blobs...")

        manifest = self._fetch_manifest(m)
        if not manifest:
            raise EmptyOllamaBlobCacheException(f"No manifest available for model {m}")

        blobs_required = [layer["digest"] for layer in manifest["layers"]]
        missing_blob_info = []

        # Check which, if any, blobs are missing locally.
        for blob_digest in blobs_required:
            blob_filename = blob_digest.replace(":", "-")
            local_blob_path = os.path.join(
                self.storage_info[m]["blob_local_root"], blob_filename
            )

            if not os.path.exists(local_blob_path):
                if self.debug:
                    print(f"[@ollama {m}] Blob {blob_digest} not found locally.")

                remote_blob_path = os.path.join(
                    self.storage_info[m]["blob_remote_root"], blob_filename
                )
                missing_blob_info.append(
                    {
                        "digest": blob_digest,
                        "filename": blob_filename,
                        "remote_path": remote_blob_path,
                        "local_path": local_blob_path,
                    }
                )

        if not missing_blob_info:
            if self.debug:
                print(f"[@ollama {m}] All blobs found locally.")
            return

        if self.debug:
            print(
                f"[@ollama {m}] Downloading {len(missing_blob_info)} missing blobs from cache..."
            )

        remote_urls = [blob_info["remote_path"] for blob_info in missing_blob_info]

        from metaflow import S3

        try:
            with S3() as s3:
                if len(remote_urls) == 1:
                    s3objs = [s3.get(remote_urls[0])]
                else:
                    s3objs = s3.get_many(remote_urls)

                if not isinstance(s3objs, list):
                    s3objs = [s3objs]

                # Move each downloaded blob to correct location
                for i, s3obj in enumerate(s3objs):
                    if not s3obj.exists:
                        blob_info = missing_blob_info[i]
                        raise EmptyOllamaBlobCacheException(
                            f"Blob {blob_info['digest']} not found in remote cache for model {m}"
                        )

                    blob_info = missing_blob_info[i]
                    os.makedirs(os.path.dirname(blob_info["local_path"]), exist_ok=True)
                    os.rename(s3obj.path, blob_info["local_path"])

                    if self.debug:
                        print(f"[@ollama {m}] Downloaded blob {blob_info['filename']}.")

        except Exception as e:
            if self.debug:
                print(f"[@ollama {m}] Error during blob fetch: {e}")
            raise EmptyOllamaBlobCacheException(
                f"Failed to fetch blobs for model {m}: {e}"
            )

        if self.debug:
            print(
                f"[@ollama {m}] Successfully downloaded all missing blobs from cache."
            )

    def _verify_model_available(self, m):
        """
        Verify model is available using Ollama API
        """
        try:
            if self.debug:
                print(f"[@ollama] DEBUG: Verifying model availability for: {m}")

            response = requests.post(
                f"{self.ollama_url}/api/show", json={"model": m}, timeout=10
            )

            available = response.status_code == 200

            if self.debug:
                if available:
                    print(f"[@ollama {m}] ✓ Model is available via API.")
                    # Also list all available models for debugging
                    try:
                        tags_response = requests.get(
                            f"{self.ollama_url}/api/tags", timeout=10
                        )
                        if tags_response.status_code == 200:
                            models = tags_response.json().get("models", [])
                            model_names = [
                                model.get("name", "unknown") for model in models
                            ]
                            print(
                                f"[@ollama] DEBUG: All available models: {model_names}"
                            )
                    except Exception as e:
                        print(f"[@ollama] DEBUG: Could not list models: {e}")
                else:
                    print(
                        f"[@ollama {m}] ✗ Model not available via API (status: {response.status_code})."
                    )
                    try:
                        error_detail = response.text
                        print(f"[@ollama] DEBUG: Error response: {error_detail}")
                    except:
                        pass

            return available

        except Exception as e:
            if self.debug:
                print(f"[@ollama {m}] Error verifying model: {e}")
            return False

    def _register_cached_model_with_ollama(self, m):
        """
        Register a cached model with Ollama using the API.
        """
        try:
            show_response = requests.post(
                f"{self.ollama_url}/api/show", json={"model": m}, timeout=10
            )

            if show_response.status_code == 200:
                if self.debug:
                    print(f"[@ollama {m}] Model already registered with Ollama.")
                return True

            # Try to create/register the model from existing files
            if self.debug:
                print(f"[@ollama {m}] Registering cached model with Ollama...")

            create_response = requests.post(
                f"{self.ollama_url}/api/create",
                json={
                    "model": m,
                    "from": m,  # Use same name - should find existing files
                    "stream": False,
                },
                timeout=60,
            )

            if create_response.status_code == 200:
                result = create_response.json()
                if result.get("status") == "success":
                    if self.debug:
                        print(f"[@ollama {m}] Successfully registered cached model.")
                    return True
                else:
                    if self.debug:
                        print(f"[@ollama {m}] Create response: {result}.")

            # Fallback: try a pull which should be fast if files exist
            if self.debug:
                print(f"[@ollama {m}] Create failed, trying pull to register...")

            pull_response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"model": m, "stream": False},
                timeout=120,
            )

            if pull_response.status_code == 200:
                result = pull_response.json()
                if result.get("status") == "success":
                    if self.debug:
                        print(f"[@ollama {m}] Model registered via pull.")
                    return True

        except requests.exceptions.RequestException as e:
            if self.debug:
                print(f"[@ollama {m}] API registration failed: {e}")
        except Exception as e:
            if self.debug:
                print(f"[@ollama {m}] Error during registration: {e}")

        return False

    def _pull_model(self, m):
        """
        Pull/setup a model, using cache when possible.
        """
        self._update_model_status(m, status="Setting up storage")
        self._log_event("info", f"Setting up model {m}")
        pull_start_time = time.time()

        self._setup_storage(m)
        cache_exists = self._check_cache_exists(m)
        will_update_cache = self._should_update_cache(m)

        if cache_exists:
            if will_update_cache:
                self._log_event(
                    "info",
                    f"Cache exists for {m}, but will be updated due to {self.cache_update_policy} policy",
                )
                print(
                    f"[@ollama {m}] Cache exists but will be updated ({self.cache_update_policy} policy)"
                )
            else:
                self._log_event("info", f"Using existing cache for {m}")
                print(f"[@ollama {m}] Using existing cache")
        else:
            if will_update_cache:
                self._log_event(
                    "info",
                    f"No cache found for {m}, will populate after successful setup",
                )
                print(f"[@ollama {m}] No cache found, will populate cache after setup")
            else:
                self._log_event(
                    "info",
                    f"No cache found for {m}, but cache updates disabled ({self.cache_update_policy} policy)",
                )
                print(
                    f"[@ollama {m}] No cache found, cache updates disabled ({self.cache_update_policy} policy)"
                )

        manifest = None
        try:
            manifest = self._fetch_manifest(m)
        except (EmptyOllamaManifestCacheException, Exception) as e:
            if self.debug:
                print(f"[@ollama {m}] No cached manifest found or error fetching: {e}")
            manifest = None

        # If we don't have a cached manifest or force_pull is True, pull the model
        if self.force_pull or not manifest:
            try:
                self._update_model_status(m, status="Downloading")
                self._log_event("info", f"Downloading model {m}...")
                print(f"[@ollama {m}] Not using cache. Downloading model {m}...")
                result = subprocess.run(
                    ["ollama", "pull", m],
                    capture_output=True,
                    text=True,
                    timeout=self.timeouts.get("pull", 600),
                )
                if result.returncode != 0:
                    self._update_model_status(m, status="Failed")
                    self._log_event(
                        "error", f"Failed to pull model {m}: {result.stderr}"
                    )
                    raise RuntimeError(
                        f"Failed to pull model {m}: stdout: {result.stdout}, stderr: {result.stderr}"
                    )
                pull_time = time.time() - pull_start_time
                self._update_model_status(m, status="Downloaded", pull_time=pull_time)
                self._log_event("success", f"Model {m} downloaded in {pull_time:.1f}s")
                print(f"[@ollama {m}] Model downloaded successfully.")
            except Exception as e:
                self._update_model_status(m, status="Failed")
                self._log_event("error", f"Error pulling model {m}: {str(e)}")
                raise RuntimeError(f"Error pulling Ollama model {m}: {e}") from e
        else:
            try:
                self._update_model_status(m, status="Loading from cache")
                self._log_event("info", f"Loading model {m} from cache")
                self._fetch_blobs(m)
                print(f"[@ollama {m}] Using cached model.")

                if not self._verify_model_available(m):
                    if not self._register_cached_model_with_ollama(m):
                        self._update_model_status(m, status="Failed")
                        self._log_event("error", f"Failed to register cached model {m}")
                        raise RuntimeError(
                            f"Failed to register cached model {m} with Ollama"
                        )

                pull_time = time.time() - pull_start_time
                self._update_model_status(m, status="Cached", pull_time=pull_time)
                self._log_event(
                    "success", f"Model {m} loaded from cache in {pull_time:.1f}s"
                )

            except (EmptyOllamaBlobCacheException, Exception) as e:
                if self.debug:
                    print(f"[@ollama {m}] Cache failed, downloading model...")
                    print(f"[@ollama {m}] Error: {e}")

                try:
                    self._update_model_status(m, status="Downloading (fallback)")
                    self._log_event(
                        "warning", f"Cache failed for {m}, downloading as fallback"
                    )
                    result = subprocess.run(
                        ["ollama", "pull", m],
                        capture_output=True,
                        text=True,
                        timeout=self.timeouts.get("pull", 600),
                    )
                    if result.returncode != 0:
                        self._update_model_status(m, status="Failed")
                        self._log_event("error", f"Fallback pull failed for model {m}")
                        raise RuntimeError(
                            f"Failed to pull model {m}: stdout: {result.stdout}, stderr: {result.stderr}"
                        )
                    pull_time = time.time() - pull_start_time
                    self._update_model_status(
                        m, status="Downloaded (fallback)", pull_time=pull_time
                    )
                    self._log_event(
                        "success",
                        f"Model {m} downloaded via fallback in {pull_time:.1f}s",
                    )
                    print(f"[@ollama {m}] Model downloaded successfully (fallback).")
                except Exception as pull_e:
                    self._update_model_status(m, status="Failed")
                    self._log_event(
                        "error",
                        f"Fallback download failed for model {m}: {str(pull_e)}",
                    )
                    raise RuntimeError(
                        f"Error pulling Ollama model {m} as fallback: {pull_e}"
                    ) from pull_e

        if not self._verify_model_available(m):
            self._update_model_status(m, status="Failed")
            self._log_event("error", f"Model {m} verification failed")
            raise RuntimeError(f"Model {m} is not available to Ollama after setup")

        metadata = self._collect_model_metadata(m)
        self._update_model_status(
            m,
            status="Ready",
            size_formatted=metadata["size_formatted"],
            blob_count=metadata["blob_count"],
        )
        self._log_event("success", f"Model {m} setup complete and verified")
        if self.debug:
            print(f"[@ollama {m}] Model setup complete and verified.")
            if metadata["size_formatted"] != "Unknown":
                print(
                    f"[@ollama {m}] Model size: {metadata['size_formatted']}, Blobs: {metadata['blob_count']}"
                )

    def _run_model(self, m):
        """
        Start the Ollama model as a subprocess and record its status.
        """
        process = None
        try:
            self._update_model_status(m, status="Starting process")
            self._log_event("info", f"Starting model process for {m}")
            if self.debug:
                print(f"[@ollama {m}] Starting model process...")

            # For `ollama run`, we want it to stay running, so no timeout on Popen.
            # The health checker will detect if it becomes unresponsive.
            process = subprocess.Popen(
                ["ollama", "run", m],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.processes[process.pid] = {
                "p": process,
                "properties": {"type": "model", "model": m, "error_details": None},
                "status": ProcessStatus.RUNNING,
            }

            if self.debug:
                print(f"[@ollama {m}] Model process PID: {process.pid}.")

            # We don't want to wait here indefinitely. Check if it failed immediately and continue.
            # The health checker in another thread will monitor long-term responsiveness of the ollama server.
            try:
                process.wait(timeout=1)  # Check if it exited immediately
                returncode = process.poll()
                if (
                    returncode is not None and returncode != 0
                ): 
                    stdout, stderr = process.communicate()
                    error_details = f"Return code: {returncode}, Error: {stderr}"
                    self.processes[process.pid]["properties"][
                        "error_details"
                    ] = error_details
                    self.processes[process.pid]["status"] = ProcessStatus.FAILED
                    self._update_model_status(m, status="Failed")
                    self._log_event(
                        "error",
                        f"Model {m} process failed immediately: {error_details}",
                    )
                    if self.debug:
                        print(
                            f"[@ollama {m}] Process {process.pid} failed immediately: {error_details}."
                        )
                    raise RuntimeError(
                        f"Ollama model {m} failed to start immediately: {error_details}"
                    )
                elif returncode == 0:
                    if self.debug:
                        print(
                            f"[@ollama {m}] Process {process.pid} exited immediately with success. This might be unexpected for a model process."
                        )
                    self.processes[process.pid]["status"] = ProcessStatus.SUCCESSFUL

            except subprocess.TimeoutExpired:
                # This is the expected case: process is running and hasn't exited
                self._update_model_status(m, status="Running")
                self._log_event("success", f"Model {m} process started successfully")
                if self.debug:
                    print(
                        f"[@ollama {m}] Model process {process.pid} is running in background."
                    )
                pass  # Process is still running, woohoo.

        except Exception as e:
            if process and process.pid in self.processes:
                self.processes[process.pid]["status"] = ProcessStatus.FAILED
                self.processes[process.pid]["properties"]["error_details"] = str(e)
            self._update_model_status(m, status="Failed")
            self._log_event("error", f"Error running model {m}: {str(e)}")
            raise RuntimeError(f"Error running Ollama model {m}: {e}") from e

    def terminate_models(self, skip_push_check=None):
        """
        Terminate all processes gracefully and update cache.
        """
        shutdown_start_time = time.time()
        self._log_event("info", "Starting Ollama shutdown sequence")
        print("[@ollama] Shutting down models...")

        # Stop the health checker first
        self.health_checker.stop()

        # Handle backward compatibility for skip_push_check parameter.
        # Get rid of a dumb legacy thing for such a young package. This is to facilitate extreme laziness...
        if skip_push_check is not None:
            if skip_push_check:
                self.cache_update_policy = "never"
                self._log_event(
                    "warning",
                    "Using legacy skip_push_check=True, setting cache policy to 'never'",
                )
            else:
                self.cache_update_policy = "force"
                self._log_event(
                    "warning",
                    "Using legacy skip_push_check=False, setting cache policy to 'force'",
                )

        model_shutdown_results = {}
        for pid, process_info in list(self.processes.items()):
            if process_info["properties"].get("type") == "model":
                model_name = process_info["properties"].get("model")
                model_shutdown_start = time.time()
                shutdown_cause = "graceful"

                self._update_model_status(model_name, status="Stopping")
                self._log_event("info", f"Stopping model {model_name}")
                if self.debug:
                    print(f"[@ollama {model_name}] Stopping model process...")

                try:
                    result = subprocess.run(
                        ["ollama", "stop", model_name],
                        capture_output=True,
                        text=True,
                        timeout=self.timeouts.get("stop", 30),
                    )
                    if result.returncode == 0:
                        process_info["status"] = ProcessStatus.SUCCESSFUL
                        self._update_model_status(model_name, status="Stopped")
                        self._log_event(
                            "success", f"Model {model_name} stopped gracefully"
                        )
                        if self.debug:
                            print(f"[@ollama {model_name}] Stopped successfully.")
                    else:
                        process_info["status"] = ProcessStatus.FAILED
                        shutdown_cause = "force_kill"
                        self._update_model_status(model_name, status="Force stopped")
                        self._log_event(
                            "warning",
                            f"Model {model_name} stop command failed, killing process",
                        )
                        if self.debug:
                            print(
                                f"[@ollama {model_name}] Stop failed: {result.stderr}. Attempting to kill process directly."
                            )
                        try:
                            process_info["p"].terminate()
                            process_info["p"].wait(timeout=5)
                            if process_info["p"].poll() is None:
                                process_info["p"].kill()
                                process_info["p"].wait()
                            process_info["status"] = ProcessStatus.SUCCESSFUL
                            self._update_model_status(model_name, status="Killed")
                            self._log_event(
                                "warning", f"Model {model_name} process killed directly"
                            )
                            if self.debug:
                                print(
                                    f"[@ollama {model_name}] Process killed directly."
                                )
                        except Exception as kill_e:
                            process_info["status"] = ProcessStatus.FAILED
                            shutdown_cause = "failed"
                            self._update_model_status(
                                model_name, status="Failed to stop"
                            )
                            self._log_event(
                                "error",
                                f"Model {model_name} failed to stop: {str(kill_e)}",
                            )
                            print(
                                f"[@ollama {model_name}] Error killing process directly: {kill_e}"
                            )

                except Exception as e:
                    process_info["status"] = ProcessStatus.FAILED
                    shutdown_cause = "failed"
                    self._update_model_status(model_name, status="Failed to stop")
                    self._log_event(
                        "error", f"Model {model_name} shutdown error: {str(e)}"
                    )
                    print(f"[@ollama {model_name}] Error stopping: {e}")

                model_shutdown_time = time.time() - model_shutdown_start
                model_shutdown_results[model_name] = {
                    "shutdown_time": model_shutdown_time,
                    "shutdown_cause": shutdown_cause,
                }
                
                should_update = self._should_update_cache(model_name)
                if should_update:
                    self._log_event(
                        "info",
                        f"Updating cache for {model_name} ({self.cache_update_policy} policy)",
                    )
                    self._update_model_cache(model_name)
                else:
                    cache_reason = f"policy is '{self.cache_update_policy}'"
                    if (
                        self.cache_update_policy == "auto"
                        and self.cache_status.get(model_name) == "exists"
                    ):
                        cache_reason = "cache already exists"
                    self._log_event(
                        "info",
                        f"Skipping cache update for {model_name} ({cache_reason})",
                    )
                    if self.debug:
                        print(
                            f"[@ollama {model_name}] Skipping cache update: {cache_reason}"
                        )

        server_shutdown_cause = "graceful"
        server_shutdown_start = time.time()
        for pid, process_info in list(self.processes.items()):
            if process_info["properties"].get("type") == "api-server":
                self._update_server_status("Stopping")
                self._log_event("info", "Stopping Ollama API server")
                if self.debug:
                    print(f"[@ollama] Stopping API server process PID {pid}.")

                process = process_info["p"]
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server_shutdown_cause = "force_kill"
                        self._log_event(
                            "warning",
                            "API server did not terminate gracefully, killing...",
                        )
                        print(
                            f"[@ollama] API server PID {pid} did not terminate, killing..."
                        )
                        process.kill()
                        process.wait()

                    process_info["status"] = ProcessStatus.SUCCESSFUL
                    self._update_server_status("Stopped")
                    self._log_event(
                        "success", f"API server stopped ({server_shutdown_cause})"
                    )
                    if self.debug:
                        print(f"[@ollama] API server terminated successfully.")
                except Exception as e:
                    process_info["status"] = ProcessStatus.FAILED
                    server_shutdown_cause = "failed"
                    self._update_server_status("Failed to stop")
                    self._log_event("error", f"API server shutdown error: {str(e)}")
                    print(f"[@ollama] Warning: Error terminating API server: {e}")

        total_shutdown_time = time.time() - shutdown_start_time
        server_shutdown_time = time.time() - server_shutdown_start
        self._update_performance("server_shutdown_time", server_shutdown_time)
        self._update_performance("total_shutdown_time", total_shutdown_time)
        self._update_performance("shutdown_cause", server_shutdown_cause)

        for model_name, results in model_shutdown_results.items():
            self._update_performance(
                f"{model_name}_shutdown_time", results["shutdown_time"]
            )
            self._update_performance(
                f"{model_name}_shutdown_cause", results["shutdown_cause"]
            )

        self._log_event(
            "success", f"Ollama shutdown completed in {total_shutdown_time:.1f}s"
        )
        print("[@ollama] All models stopped.")
        if self.debug:
            if hasattr(self, "stats") and self.stats:
                print("[@ollama] Performance summary:")
                for operation, stats in self.stats.items():
                    runtime = stats.get("process_runtime", 0)
                    if runtime > 1:  # Only show operations that took meaningful time
                        print(f"[@ollama]   {operation}: {runtime:.1f}s")

    def _update_model_cache(self, model_name):
        """
        Update the remote cache with model files if needed.
        """
        try:
            manifest = self._fetch_manifest(model_name)
            if not manifest:
                if self.debug:
                    print(
                        f"[@ollama {model_name}] No manifest available for cache update."
                    )
                return

            from metaflow import S3

            cache_up_to_date = True
            key_paths = [
                (
                    self.storage_info[model_name]["manifest_remote"],
                    self.storage_info[model_name]["manifest_local"],
                )
            ]

            with S3() as s3:
                s3objs = s3.list_paths(
                    [self.storage_info[model_name]["blob_remote_root"]]
                )
                for layer in manifest["layers"]:
                    expected_blob_sha = layer["digest"]
                    if expected_blob_sha not in s3objs:
                        cache_up_to_date = False
                        break

                if not cache_up_to_date:
                    blob_count = len(manifest.get("layers", []))
                    print(
                        f"[@ollama {model_name}] Uploading {blob_count} files to cache..."
                    )

                    for layer in manifest["layers"]:
                        blob_filename = layer["digest"].replace(":", "-")
                        key_paths.append(
                            (
                                os.path.join(
                                    self.storage_info[model_name]["blob_remote_root"],
                                    blob_filename,
                                ),
                                os.path.join(
                                    self.storage_info[model_name]["blob_local_root"],
                                    blob_filename,
                                ),
                            )
                        )

                    s3.put_files(key_paths)
                    print(f"[@ollama {model_name}] Cache updated.")
                else:
                    if self.debug:
                        print(f"[@ollama {model_name}] Cache is up to date.")

        except Exception as e:
            if self.debug:
                print(f"[@ollama {model_name}] Error updating cache: {e}")

    def get_ollama_storage_root(self, backend):
        """
        Return the path to the root of the datastore.
        """
        if backend.TYPE == "s3":
            from metaflow.metaflow_config import DATASTORE_SYSROOT_S3

            self.local_datastore = False
            return os.path.join(DATASTORE_SYSROOT_S3, OLLAMA_SUFFIX)
        elif backend.TYPE == "azure":
            from metaflow.metaflow_config import DATASTORE_SYSROOT_AZURE

            self.local_datastore = False
            return os.path.join(DATASTORE_SYSROOT_AZURE, OLLAMA_SUFFIX)
        elif backend.TYPE == "gs":
            from metaflow.metaflow_config import DATASTORE_SYSROOT_GS

            self.local_datastore = False
            return os.path.join(DATASTORE_SYSROOT_GS, OLLAMA_SUFFIX)
        else:
            self.local_datastore = True
            return None

    def _attempt_ollama_restart(self):
        """Attempt to restart Ollama when circuit breaker suggests it"""
        try:
            print("[@ollama] Attempting Ollama restart due to circuit breaker...")

            server_stopped = False
            for pid, process_info in list(self.processes.items()):
                if process_info["properties"].get("type") == "api-server":
                    process = process_info["p"]
                    try:
                        process.terminate()
                        process.wait(timeout=10)
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                        process_info["status"] = ProcessStatus.SUCCESSFUL
                        server_stopped = True
                        if self.debug:
                            print(
                                f"[@ollama] Stopped server process {pid} during restart"
                            )
                    except Exception as e:
                        if self.debug:
                            print(
                                f"[@ollama] Error stopping server {pid} during restart: {e}"
                            )

            if not server_stopped:
                if self.debug:
                    print("[@ollama] No server process found to stop during restart")

            time.sleep(2)
            self._launch_server()
            health_attempts = 3
            for attempt in range(health_attempts):
                if self._verify_server_health():
                    print("[@ollama] Restart successful")
                    return True
                else:
                    if attempt < health_attempts - 1:
                        if self.debug:
                            print(
                                f"[@ollama] Health check failed, attempt {attempt + 1}/{health_attempts}"
                            )
                        time.sleep(5)

            print(
                "[@ollama] Restart failed - server not healthy after multiple attempts"
            )
            return False

        except Exception as e:
            print(f"[@ollama] Restart failed: {e}")
            return False

    def _verify_server_health(self):
        """Quick health check for server availability"""
        try:
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=self.timeouts.get("health_check", 5),
            )
            return response.status_code == 200
        except Exception:
            return False

    def _collect_model_metadata(self, m):
        """
        Collect model metadata including size and blob count from manifest and API
        """
        metadata = {"size_bytes": None, "size_formatted": "Unknown", "blob_count": 0}

        try:
            # get info from manifest (cached models)
            manifest = self._fetch_manifest(m)
            if manifest and "layers" in manifest:
                metadata["blob_count"] = len(manifest["layers"])
                total_size = 0
                for layer in manifest["layers"]:
                    if "size" in layer:
                        total_size += layer["size"]

                if total_size > 0:
                    metadata["size_bytes"] = total_size
                    metadata["size_formatted"] = self._format_bytes(total_size)

            try:
                response = requests.post(
                    f"{self.ollama_url}/api/show", json={"model": m}, timeout=10
                )
                if response.status_code == 200:
                    model_info = response.json()

                    if (
                        "details" in model_info
                        and "parameter_size" in model_info["details"]
                    ):
                        param_size = model_info["details"]["parameter_size"]
                        if self.debug:
                            print(f"[@ollama {m}] Parameter size: {param_size}")

                    # If we get model_info but didn't have manifest info, try to get layer count
                    if metadata["blob_count"] == 0 and "details" in model_info:
                        details = model_info["details"]
                        if "families" in details or "family" in details:
                            metadata["blob_count"] = "API"

            except Exception as api_e:
                if self.debug:
                    print(f"[@ollama {m}] Could not get API metadata: {api_e}")

        except Exception as e:
            if self.debug:
                print(f"[@ollama {m}] Error collecting model metadata: {e}")

        return metadata

    def _format_bytes(self, bytes_count):
        """Format bytes into human-readable string"""
        if bytes_count is None:
            return "Unknown"

        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_count < 1024.0:
                if unit == "B":
                    return f"{int(bytes_count)} {unit}"
                else:
                    return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.1f} PB"