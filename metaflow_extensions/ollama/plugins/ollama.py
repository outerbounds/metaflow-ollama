import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import socket
import sys
import os
import functools


class ProcessStatus:
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    SUCCESSFUL = "SUCCESSFUL"


class OllamaManager:

    """
    A process manager for Ollama runtimes.
    This is run locally, e.g., whether @ollama has a local, remote, or managed backend.
    """

    def __init__(self, models, backend="local", debug=False):
        self.models = {}
        self.processes = {}
        self.debug = debug
        self.stats = {}

        if backend != "local":
            raise ValueError(
                "OllamaManager only supports the 'local' backend at this time."
            )

        self._timeit(self._install_ollama, "install_ollama")
        self._timeit(self._launch_server, "launch_server")

        # Pull models concurrently
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._pull_model, m) for m in models]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    raise RuntimeError(f"Error pulling one or more models: {e}") from e

        # Run models as background processes.
        for m in models:
            f = functools.partial(self._run_model, m)
            self._timeit(f, f"model_{m.lower()}")

    def _timeit(self, f, name):
        t0 = time.time()
        f()
        tf = time.time()
        self.stats[name] = {"process_runtime": tf - t0}

    def _install_ollama(self, max_retries=3):

        try:
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
            if result.returncode == 0:
                if self.debug:
                    print("[@ollama] is already installed.")
                return
        except Exception as e:
            print("[@ollama] Did not find Ollama installation: %s" % e)
            if sys.platform == "darwin":
                raise RuntimeError(
                    "On macOS, please install Ollama manually from https://ollama.com/download"
                )

        env = os.environ.copy()
        env["CURL_IPRESOLVE"] = "4"
        for attempt in range(max_retries):
            try:
                install_cmd = ["curl", "-fsSL", "https://ollama.com/install.sh"]
                curl_proc = subprocess.run(
                    install_cmd, capture_output=True, text=True, env=env
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
                )
                if sh_proc.returncode != 0:
                    raise RuntimeError(
                        f"Ollama installation script failed: stdout: {sh_proc.stdout}, stderr: {sh_proc.stderr}"
                    )
                if self.debug:
                    print("[@ollama] Installed successfully.")
                    break
            except Exception as e:
                print(f"Installation attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise RuntimeError(
                        f"Error installing Ollama after {max_retries} attempts: {e}"
                    ) from e

    def _is_port_open(self, host, port, timeout=1):
        """Check if a TCP port is open on a given host."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            try:
                sock.connect((host, port))
                return True
            except socket.error:
                return False

    def _launch_server(self):
        """
        Start the Ollama server process and ensure it's running.
        This version waits until the server is listening on port 11434.
        """
        try:
            if self.debug:
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
                print(
                    "[@ollama] Started Ollama server process with PID %s" % process.pid
                )

            # Wait until the server is ready (listening on 127.0.0.1:11434)
            host, port = "127.0.0.1", 11434
            retries = 0
            max_retries = 10
            while (
                not self._is_port_open(host, port, timeout=1) and retries < max_retries
            ):
                print(
                    "[@ollama] Waiting for server to be ready... (%d/%d)"
                    % (retries + 1, max_retries)
                )
                time.sleep(1)
                retries += 1

            if not self._is_port_open(host, port, timeout=1):
                error_details = (
                    f"Ollama server did not start listening on {host}:{port}"
                )
                self.processes[process.pid]["properties"][
                    "error_details"
                ] = error_details
                self.processes[process.pid]["status"] = ProcessStatus.FAILED
                raise RuntimeError(f"Ollama server failed to start. {error_details}")

            # Check if the process has unexpectedly terminated
            returncode = process.poll()
            if returncode is not None:
                stdout, stderr = process.communicate()
                error_details = f"Return code: {returncode}, Error: {stderr}"
                self.processes[process.pid]["properties"][
                    "error_details"
                ] = error_details
                self.processes[process.pid]["status"] = ProcessStatus.FAILED
                raise RuntimeError(f"Ollama server failed to start. {error_details}")

            print("[@ollama] Server is ready.")

        except Exception as e:
            if "process" in locals() and process.pid in self.processes:
                self.processes[process.pid]["status"] = ProcessStatus.FAILED
                self.processes[process.pid]["properties"]["error_details"] = str(e)
            raise RuntimeError(f"Error starting Ollama server: {e}") from e

    def _pull_model(self, m):
        try:
            if self.debug:
                print("[@ollama] Pulling model: %s" % m)
            result = subprocess.run(
                ["ollama", "pull", m], capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to pull model {m}: stdout: {result.stdout}, stderr: {result.stderr}"
                )
            if self.debug:
                print("[@ollama] Model %s pulled successfully." % m)
        except Exception as e:
            raise RuntimeError(f"Error pulling Ollama model {m}: {e}") from e

    def _run_model(self, m):
        """
        Start the Ollama model as a subprocess and record its status.
        """
        process = None
        try:
            if self.debug:
                print("[@ollama] Running model: %s" % m)
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
                print("[@ollama] Stored process %s for model %s." % (process.pid, m))

            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

            returncode = process.poll()
            if returncode is not None:
                stdout, stderr = process.communicate()
                if returncode == 0:
                    self.processes[process.pid]["status"] = ProcessStatus.SUCCESSFUL
                    if self.debug:
                        print(
                            "[@ollama] Process %s for model %s exited successfully."
                            % (process.pid, m)
                        )
                else:
                    error_details = f"Return code: {returncode}, Error: {stderr}"
                    self.processes[process.pid]["properties"][
                        "error_details"
                    ] = error_details
                    self.processes[process.pid]["status"] = ProcessStatus.FAILED
                    if self.debug:
                        print(
                            "[@ollama] Process %s for model %s failed: %s"
                            % (process.pid, m, error_details)
                        )
        except Exception as e:
            if process and process.pid in self.processes:
                self.processes[process.pid]["status"] = ProcessStatus.FAILED
                self.processes[process.pid]["properties"]["error_details"] = str(e)
            raise RuntimeError(f"Error running Ollama model {m}: {e}") from e

    def terminate_models(self):
        """
        Terminate all processes gracefully.
        First, stop model processes using 'ollama stop <model>'.
        Then, shut down the API server process.
        """

        for pid, process_info in list(self.processes.items()):
            if process_info["properties"].get("type") == "model":
                model_name = process_info["properties"].get("model")
                if self.debug:
                    print(
                        "[@ollama] Stopping model %s using 'ollama stop'" % model_name
                    )
                try:
                    result = subprocess.run(
                        ["ollama", "stop", model_name], capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        process_info["status"] = ProcessStatus.SUCCESSFUL
                        if self.debug:
                            print(
                                "[@ollama] Model %s stopped successfully." % model_name
                            )
                    else:
                        process_info["status"] = ProcessStatus.FAILED
                        if self.debug:
                            print(
                                "[@ollama] Model %s failed to stop gracefully. Return code: %s, Error: %s"
                                % (model_name, result.returncode, result.stderr)
                            )
                except Exception as e:
                    process_info["status"] = ProcessStatus.FAILED
                    print("[@ollama] Error stopping model %s: %s" % (model_name, e))

        # Then, stop the API server
        for pid, process_info in list(self.processes.items()):
            if process_info["properties"].get("type") == "api-server":
                if self.debug:
                    print(
                        "[@ollama] Stopping API server process with PID %s using process.terminate()"
                        % pid
                    )
                process = process_info["p"]
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(
                            "[@ollama] API server process %s did not terminate in time; killing it."
                            % pid
                        )
                        process.kill()
                        process.wait()
                    returncode = process.poll()
                    if returncode is None or returncode != 0:
                        process_info["status"] = ProcessStatus.FAILED
                        print(
                            "[@ollama] API server process %s terminated with error code %s."
                            % (pid, returncode)
                        )
                    else:
                        process_info["status"] = ProcessStatus.SUCCESSFUL
                        if self.debug:
                            print(
                                "[@ollama] API server process %s terminated successfully."
                                % pid
                            )
                except Exception as e:
                    process_info["status"] = ProcessStatus.FAILED
                    print(
                        "[@ollama] Warning: Error while terminating API server process %s: %s"
                        % (pid, e)
                    )